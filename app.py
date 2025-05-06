"""
Web application for image denoising.
"""

import os
import uuid
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
import glob
import time
import tempfile
import shutil
from utils.resolution_adapter import ResolutionAdapter

# Define the CastToFloat32 custom layer if needed for older models
class CastToFloat32(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CastToFloat32, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
    
    def compute_output_shape(self, input_shape):
        return input_shape

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
NEW_MODEL_PATH = 'saved_models/denoising_model_best.keras'  # Path to our new model
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

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

def load_denoising_model():
    """
    Load the denoising model.
    
    Returns:
        Model wrapper with predict method
    """
    try:
        print(f"Loading model: {NEW_MODEL_PATH}")
        # Enable unsafe deserialization since we trust our own models
        tf.keras.config.enable_unsafe_deserialization()
        # Load the model (without compiling)
        model = tf.keras.models.load_model(NEW_MODEL_PATH, compile=False)
        print("Successfully loaded model")
        
        # Wrap in a simple object with predict method
        class ModelWrapper:
            """Wrapper for the model to handle prediction."""
            
            def __init__(self, model):
                self.model = model
                # Get input shape for debugging
                self.input_shape = tuple(model.input.shape[1:])
                print(f"Model input shape: {self.input_shape}")
                # Get output shape for debugging
                self.output_shape = tuple(model.output.shape[1:])
                print(f"Model output shape: {self.output_shape}")
            
            def predict(self, input_data):
                """
                Run prediction on image data.
                
                Args:
                    input_data: Input image as numpy array in uint8 [0-255] or float32 [0-1]
                    
                Returns:
                    Denoised image as numpy array in the same format as input
                """
                # Store original dtype and range
                original_dtype = input_data.dtype
                is_uint8 = original_dtype == np.uint8
                
                # Normalize the input to 0-1 range if needed
                if is_uint8:
                    input_data = input_data.astype(np.float32) / 255.0
                
                # Use model for prediction
                print(f"Running prediction with input shape: {input_data.shape}")
                denoised = self.model.predict(input_data)
                print(f"Prediction output shape: {denoised.shape}")
                
                # Ensure output is in 0-1 range
                if denoised.max() > 1.0:
                    denoised = denoised / 255.0
                
                # Convert back to original format if needed
                if is_uint8:
                    denoised = (denoised * 255).astype(np.uint8)
                
                return denoised
        
        # Create the model wrapper
        wrapper = ModelWrapper(model)
        
        # Create resolution adapter with the model wrapper
        adapter = ResolutionAdapter(wrapper, target_size=None, overlap=0.25)
        
        return adapter
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model when the app starts
denoising_model = load_denoising_model()

def process_image(image_path):
    """
    Process an image file by denoising it.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Path to the denoised image file
    """
    try:
        print(f"Processing image: {image_path}")
        
        # Load image using PIL
        img = Image.open(image_path)
        
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Check if model is available
        global denoising_model
        if denoising_model is None:
            print("No model available, loading model...")
            denoising_model = load_denoising_model()
            if denoising_model is None:
                raise ValueError("Could not load denoising model")
        
        # Denoise image using resolution adapter
        print("Denoising image...")
        denoised_img = denoising_model.process_image(img)
        
        # Generate a unique filename for the result
        basename = os.path.basename(image_path)
        filename, ext = os.path.splitext(basename)
        result_filename = f"{filename}_denoised{ext}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        # Save the denoised image
        denoised_img.save(result_path)
        print(f"Saved denoised image to: {result_path}")
        
        return result_path
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    """Render the main page."""
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
        try:
            # Secure the filename to prevent malicious input
            filename = secure_filename(file.filename)
            
            # Add a UUID to ensure uniqueness
            unique_filename = f"{uuid.uuid4()}_{filename}"
            
            # Create full path for the uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the uploaded file
            file.save(upload_path)
            print(f"Saved uploaded file to: {upload_path}")
            
            # Process the image
            result_path = process_image(upload_path)
            
            if result_path:
                # Get filenames for templates
                result_filename = os.path.basename(result_path)
                upload_filename = unique_filename
                
                return render_template('index.html', 
                                       original=upload_filename, 
                                       denoised=result_filename, 
                                       success=True)
            else:
                return render_template('index.html', 
                                       error='Error processing image')
                
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', error=f'Error: {str(e)}')
    
    return render_template('index.html', error='Invalid file format')

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