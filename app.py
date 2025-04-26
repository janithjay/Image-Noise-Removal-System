"""
Web application for image denoising.
"""

import os
import uuid
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from model import DenoisingAutoencoder
from utils.data_loader import DataLoader
from utils.preprocessor import NoiseGenerator
from utils.visualizer import combine_images, calculate_metrics

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
MODEL_PATH = 'saved_models/best_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, 'saved_models']:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """
    Process an image using the trained model.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Paths to the processed images (original, noisy, denoised, combined)
    """
    # Generate unique filenames
    base_filename = uuid.uuid4().hex
    noisy_filename = f"noisy_{base_filename}.png"
    denoised_filename = f"denoised_{base_filename}.png"
    combined_filename = f"combined_{base_filename}.png"
    
    noisy_path = os.path.join(app.config['RESULTS_FOLDER'], noisy_filename)
    denoised_path = os.path.join(app.config['RESULTS_FOLDER'], denoised_filename)
    combined_path = os.path.join(app.config['RESULTS_FOLDER'], combined_filename)
    
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            
        # Load data
        data_loader = DataLoader()
        image = data_loader.load_single_image(image_path)
        
        # Load model
        model = DenoisingAutoencoder.load(MODEL_PATH)
        
        # Add noise to create a noisy version
        noise_gen = NoiseGenerator()
        noisy = noise_gen.add_noise(image)[0]
        
        # Save noisy image
        noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
        noisy_img.save(noisy_path)
        
        # Process image with model
        denoised = model.predict(image)[0]
        
        # Clip values and convert to PIL Image
        denoised = np.clip(denoised, 0, 1)
        denoised_img = Image.fromarray((denoised * 255).astype(np.uint8))
        denoised_img.save(denoised_path)
        
        # Calculate metrics with improved error handling
        try:
            metrics = calculate_metrics(image[0], denoised)
            # Check if metrics calculation failed
            if metrics['PSNR'] == 0 and metrics['SSIM'] == 0:
                print("Warning: Model metrics calculation returned zeros, trying with saved images")
                metrics = calculate_metrics(image_path, denoised_path)
        except Exception as e:
            print(f"Error calculating metrics for model: {str(e)}")
            metrics = {'PSNR': 0, 'SSIM': 0}
        
        # Create combined visualization
        combine_images(
            image_path, 
            noisy_path, 
            denoised_path, 
            save_path=combined_path,
            metrics=metrics
        )
        
        return {
            'original': os.path.basename(image_path),
            'noisy': noisy_filename,
            'denoised': denoised_filename,
            'combined': combined_filename,
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"Critical error in image processing: {str(e)}")
        # Create a default response with error message
        
        # Create a simple error image (solid gray with text)
        error_img = Image.new('RGB', (256, 256), (200, 200, 200))
        # Add text about the error if possible
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(error_img)
            # Try to load a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except IOError:
                font = ImageFont.load_default()
                
            draw.text((10, 120), f"Error processing image", fill=(0, 0, 0), font=font)
        except:
            pass
            
        # Save error images
        error_img.save(noisy_path)
        error_img.save(denoised_path)
        error_img.save(combined_path)
        
        # Return dummy data
        return {
            'original': os.path.basename(image_path),
            'noisy': noisy_filename,
            'denoised': denoised_filename,
            'combined': combined_filename,
            'metrics': {'PSNR': 0, 'SSIM': 0}
        }

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/denoise', methods=['POST'])
def denoise():
    """Handle image upload and denoising."""
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process the image with the neural network model
        result = process_image(file_path)
        
        return render_template(
            'result.html',
            original=url_for('uploaded_file', filename=unique_filename),
            noisy=url_for('result_file', filename=result['noisy']),
            denoised=url_for('result_file', filename=result['denoised']),
            combined=url_for('result_file', filename=result['combined']),
            metrics=result['metrics']
        )
    
    return render_template('index.html', error='File type not allowed')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 