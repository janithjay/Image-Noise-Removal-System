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
from utils.preprocessor import NoiseGenerator, TraditionalDenoiser
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

def process_image(image_path, model_path):
    """
    Process an image using the trained model.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained model
        
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
    
    # Check if model exists
    if not os.path.exists(model_path):
        # If model doesn't exist, add noise manually as a demo
        print("Model not found. Using traditional denoising as a demo.")
        
        # Load and resize image
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        
        # Add noise for demonstration
        noise_gen = NoiseGenerator()
        noisy_array = noise_gen.add_noise(np.expand_dims(img_array, axis=0))[0]
        
        # Save noisy image
        noisy_img = Image.fromarray((noisy_array * 255).astype(np.uint8))
        noisy_img.save(noisy_path)
        
        # Apply traditional denoising
        denoiser = TraditionalDenoiser(method='bilateral')
        denoised_array = denoiser.denoise(noisy_array)
        
        # Convert back to PIL image
        denoised_img = Image.fromarray((denoised_array * 255).astype(np.uint8))
        denoised_img.save(denoised_path)
        
        # Calculate metrics
        metrics = calculate_metrics(img_array, denoised_array)
    else:
        # Load the actual model and process the image
        # Load data
        data_loader = DataLoader()
        image = data_loader.load_single_image(image_path)
        
        # Load model
        model = DenoisingAutoencoder.load(model_path)
        
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
        
        # Calculate metrics
        metrics = calculate_metrics(image[0], denoised)
    
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
        
        # Process the image
        result = process_image(file_path, MODEL_PATH)
        
        # Return results
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