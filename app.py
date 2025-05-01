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
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model as KerasModel
import glob
import time
import tempfile
import shutil

from model import DenoisingAutoencoder
from utils.data_loader import DataLoader, NoiseGenerator
from utils.visualizer import combine_images, calculate_metrics

# Define custom loss functions
def ssim_loss(y_true, y_pred):
    """Structural Similarity Index loss function."""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def perceptual_loss(y_true, y_pred):
    """Perceptual loss based on VGG16 feature maps."""
    # This is a simplified version since we're only using it for model loading
    return K.mean(K.square(y_true - y_pred))

def edge_preservation_loss(y_true, y_pred):
    """Edge preservation loss using gradient differences."""
    # This is a simplified version since we're only using it for model loading
    return K.mean(K.abs(y_true - y_pred))

def combined_loss(alpha=0.5, beta=0.3, gamma=0.1, delta=0.1):
    """Combined loss function with weights."""
    def loss(y_true, y_pred):
        mse_loss = K.mean(K.square(y_true - y_pred))
        ss_loss = ssim_loss(y_true, y_pred)
        # Use simplified versions for model loading
        percep_loss = K.mean(K.square(y_true - y_pred))
        edge_loss = K.mean(K.abs(y_true - y_pred))
        
        return alpha * mse_loss + beta * ss_loss + gamma * percep_loss + delta * edge_loss
    return loss

# Define the CastToFloat32 custom layer
class CastToFloat32(Layer):
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
MODEL_PATH = 'converted_models/fixed_model.keras'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, 'saved_models']:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Temp path for the fallback model if available
FALLBACK_MODEL_PATH = 'saved_models/best_model.h5'

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def try_load_model(model_path):
    """
    Try to load a model from the given path.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model or None if loading failed
    """
    try:
        # Define custom objects dictionary
        custom_objects = {
            'CastToFloat32': CastToFloat32,
            'ssim_loss': ssim_loss,
            'perceptual_loss': perceptual_loss,
            'edge_preservation_loss': edge_preservation_loss,
            'combined_loss': combined_loss(0.5, 0.3, 0.1, 0.1),
            'mse': tf.keras.losses.MeanSquaredError()
        }
        
        print(f"Attempting to load model: {model_path}")
        
        # DIRECT KERAS LOADING for newer converted models
        if model_path.endswith('.keras'):
            try:
                print("Trying to load .keras format model...")
                # Enable unsafe deserialization since we trust our own models
                tf.keras.config.enable_unsafe_deserialization()
                model = tf.keras.models.load_model(model_path, compile=False)
                print("Successfully loaded .keras model")
                
                # Wrap in a simple object with predict method
                class SimpleModelWrapper:
                    """Wrapper for the model to handle prediction."""
                    
                    def __init__(self, model):
                        self.model = model
                    
                    def predict(self, image_data):
                        """
                        Run prediction on image data with enhanced color preservation.
                        
                        Args:
                            image_data: Input image as numpy array in uint8 [0-255] or float32 [0-1]
                            
                        Returns:
                            Denoised image as numpy array in float32 [0-1] range
                        """
                        # Save original input for residual connection
                        original_input = image_data.copy()
                        
                        # Normalize the input to 0-1 range if needed
                        if image_data.dtype == np.uint8:
                            image_data = image_data.astype(np.float32) / 255.0
                        
                        # Convert original to [0-1] range if it's not already
                        if original_input.dtype == np.uint8:
                            original_input = original_input.astype(np.float32) / 255.0
                        
                        # Make sure we have a batch dimension
                        need_reshape = len(image_data.shape) == 3
                        if need_reshape:
                            image_data = np.expand_dims(image_data, 0)
                            original_input = np.expand_dims(original_input, 0)
                        
                        # Get the model's prediction
                        model_output = self.model.predict(image_data)
                        
                        # For models with outputs in 0-255 range
                        if model_output.max() > 1.0:
                            model_output = model_output / 255.0
                        
                        # ENHANCED COLOR PRESERVATION:
                        # Use a channel-specific approach that maintains luminance from the model output
                        # but preserves chrominance from the original image
                        
                        # 1. Calculate luminance (brightness) from model output
                        # Using standard RGB to luminance conversion weights
                        luminance_weights = np.array([0.299, 0.587, 0.114])
                        
                        # Calculate luminance from model output (per pixel)
                        model_luminance = np.sum(model_output * luminance_weights, axis=-1, keepdims=True)
                        original_luminance = np.sum(original_input * luminance_weights, axis=-1, keepdims=True)
                        
                        # 2. Calculate chrominance (color) components from original
                        # We'll use a simple approach: color offset from luminance
                        original_chrominance = original_input - original_luminance
                        
                        # 3. IMPROVED COLOR PRESERVATION - Hybrid approach combining multiple techniques
                        
                        # A) Direct luminance+chrominance approach (strong color preservation)
                        denoised_v1 = model_luminance + original_chrominance
                        
                        # B) Weighted average approach (balanced)
                        # Increase weight for original to 85% to better preserve colors
                        alpha = 0.85  # Weight for original color (increased from 0.75)
                        beta = 0.15   # Weight for model color (decreased from 0.25)
                        denoised_v2 = (original_input * alpha) + (model_output * beta)
                        
                        # C) Adaptive blending based on noise estimation
                        # Estimate noise by comparing local variance
                        # Higher variance difference = more likely to be noise
                        noise_estimation = np.abs(np.std(original_input, axis=-1, keepdims=True) - 
                                                np.std(model_output, axis=-1, keepdims=True))
                        # Normalize to 0-1 range for blending
                        max_noise = np.max(noise_estimation)
                        if max_noise > 0:
                            noise_mask = noise_estimation / max_noise
                        else:
                            noise_mask = noise_estimation
                        
                        # DETECT IMAGE TYPE - portraits need different handling than landscapes
                        # Check for potential portrait/person image
                        # People/portraits usually have:
                        # 1. Moderate red channel for skin tones
                        # 2. Less extreme color variation than landscapes
                        # 3. More central focus
                        
                        # Simple skin tone detector (moderate red, moderate green, lower blue)
                        skin_tone_pixels = np.logical_and(
                            np.logical_and(
                                original_input[:,:,:,0] > 0.4, 
                                original_input[:,:,:,0] < 0.9
                            ),
                            np.logical_and(
                                original_input[:,:,:,1] > 0.3,
                                original_input[:,:,:,2] < 0.7
                            )
                        )
                        
                        # Calculate percentage of skin tone pixels
                        skin_percentage = np.mean(skin_tone_pixels)
                        
                        # Detect if this is likely a portrait/person image
                        is_portrait = skin_percentage > 0.08  # >8% skin tone indicates person
                        is_warm_image = np.mean(original_input[:,:,:,0]) > np.mean(original_input[:,:,:,2]) * 1.5
                        
                        # Create initial denoised output
                        denoised = denoised_v1 * (1 - noise_mask) + denoised_v2 * noise_mask
                        
                        if is_portrait:
                            # COMPLETE PORTRAIT OVERHAUL - Focus on maximum color preservation for portraits
                            # Create a completely new approach that prioritizes original color preservation
                            
                            # STEP 1: Create a facial region mask with a strong center focus
                            h, w = original_input.shape[1:3]
                            center_x, center_y = w // 2, h // 2
                            face_mask = np.zeros((h, w, 1), dtype=np.float32)
                            face_radius = min(w, h) * 0.4  # Larger face region for better coverage
                            
                            # Create a smooth circular mask for face region
                            for i in range(h):
                                for j in range(w):
                                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                                    if dist < face_radius:
                                        # Strong effect in core face region with smooth falloff
                                        face_mask[i, j, 0] = 1.0 - 0.5 * (dist / face_radius)
                                    else:
                                        # Gradual falloff outside face radius
                                        face_mask[i, j, 0] = max(0, 0.5 - 0.5 * (dist - face_radius) / (max(w, h) * 0.3))

                            # STEP 2: Identify and enhance skin regions
                            # Convert skin boolean mask to float32 for blending
                            skin_mask = skin_tone_pixels.astype(np.float32)
                            if len(skin_mask.shape) == 3:
                                skin_mask = np.expand_dims(skin_mask, axis=-1)
                            
                            # Combine face mask with skin mask for targeted processing
                            combined_mask = np.maximum(face_mask, skin_mask * 0.8)
                            
                            # STEP 3: Color channel adjustments based on portrait characteristics
                            # Create a color-corrected version of the denoised output
                            corrected_output = np.copy(denoised)
                            
                            # Apply skin tone correction throughout with a MUCH stronger red boost
                            corrected_output[:,:,:,0] = denoised[:,:,:,0] * 1.4  # Significantly boost red (was 1.2)
                            corrected_output[:,:,:,1] = denoised[:,:,:,1] * 0.9  # Reduce green more (was 0.95)
                            corrected_output[:,:,:,2] = denoised[:,:,:,2] * 0.8  # Reduce blue more (was 0.85)
                            
                            # DIRECT COLOR PRESERVATION: For strong skin tone areas, use original colors
                            # Calculate strong skin tone mask (more red, less blue)
                            strong_skin = np.logical_and(
                                np.logical_and(
                                    original_input[:,:,:,0] > 0.5,  # Higher red threshold
                                    original_input[:,:,:,0] > original_input[:,:,:,2] * 1.5  # Red much higher than blue
                                ),
                                original_input[:,:,:,1] > 0.35  # Enough green for skin
                            ).astype(np.float32)
                            
                            # Add spatial dimension if needed
                            if len(strong_skin.shape) == 3:
                                strong_skin = np.expand_dims(strong_skin, axis=-1)
                            
                            # STEP 4: Create multiple blending options for different regions
                            
                            # First create a hybrid that uses original colors but denoised luminance
                            lum_weights = np.array([0.299, 0.587, 0.114])
                            
                            # Get luminance from denoised image (which has better noise characteristics)
                            denoised_luma = np.sum(denoised * lum_weights, axis=-1, keepdims=True)
                            
                            # Get color offset from original (chrominance only)
                            original_luma = np.sum(original_input * lum_weights, axis=-1, keepdims=True)
                            original_chroma = original_input - original_luma
                            
                            # Create hybrid using denoised luminance but original chrominance
                            hybrid = denoised_luma + original_chroma
                            
                            # SPECIAL SKIN HANDLING: For areas with very strong skin tones, 
                            # directly use original values for maximum preservation
                            # Create skin-preserved variant that directly uses original RGB values in skin areas
                            skin_preserved = original_input.copy()
                            
                            # Apply noise reduction to skin_preserved by using denoised luminance
                            # but keeping original colors in strong skin areas
                            skin_luma_blend = 0.7  # Use 70% denoised luminance, 30% original
                            skin_preserved_luma = original_luma * (1 - skin_luma_blend) + denoised_luma * skin_luma_blend
                            skin_preserved = skin_preserved_luma + original_chroma
                            
                            # Blend skin_preserved into hybrid based on strong_skin mask
                            hybrid = hybrid * (1 - strong_skin) + skin_preserved * strong_skin
                            
                            # Blend hybrid into the corrected output based on mask
                            # For faces and skin, use up to 90% original colors
                            # For non-face/skin regions, use more denoised content
                            blend_strength = combined_mask * 0.9  # Up to 90% original in important areas
                            
                            # Apply the primary blend
                            portrait_result = corrected_output * (1 - blend_strength) + hybrid * blend_strength
                            
                            # STEP 5: Final contrast and saturation adjustment
                            # Apply channel-specific contrast stretch
                            for c in range(3):
                                min_val = np.min(portrait_result[:,:,:,c])
                                max_val = np.max(portrait_result[:,:,:,c])
                                if max_val > min_val:
                                    # Apply mild contrast enhancement
                                    portrait_result[:,:,:,c] = (portrait_result[:,:,:,c] - min_val) / (max_val - min_val)
                            
                            # Set the final result
                            denoised = np.clip(portrait_result, 0, 1)
                            
                        elif is_warm_image:
                            # Add warmth preservation for Buddha-like sunset images
                            # Create warmth adjustment based on original color ratios
                            warmth = np.zeros_like(denoised)
                            
                            # Calculate warmth from original (red-blue ratio)
                            red_blue_ratio = np.clip(original_input[:,:,:,0] / (original_input[:,:,:,2] + 0.01), 0, 5)
                            
                            # Apply subtle correction to maintain warm colors
                            warmth_strength = 0.1  # Subtle adjustment
                            warmth[:,:,:,0] = denoised[:,:,:,0] * (1 + warmth_strength * red_blue_ratio)  # Boost red
                            warmth[:,:,:,1] = denoised[:,:,:,1]  # Keep green as is
                            warmth[:,:,:,2] = denoised[:,:,:,2] * (1 - warmth_strength * 0.5)  # Slightly reduce blue
                            
                            # Blend with regular denoised output
                            denoised = np.clip(warmth, 0, 1)
                        
                        # Apply a final contrast enhancement to avoid flat areas
                        # Find min/max per channel for contrast stretching
                        mins = np.min(denoised, axis=(0, 1, 2), keepdims=True)
                        maxs = np.max(denoised, axis=(0, 1, 2), keepdims=True)
                        
                        # Only apply if we have enough range to work with
                        if np.any(maxs - mins > 0.1):
                            # Apply contrast stretch but preserve some headroom
                            denoised = (denoised - mins) / (maxs - mins + 1e-10)
                        
                        # Clip to valid range [0-1]
                        denoised = np.clip(denoised, 0, 1)
                        
                        # Remove batch dimension if it was added
                        if need_reshape:
                            denoised = denoised[0]
                        
                        return denoised
                
                return SimpleModelWrapper(model)
            except Exception as e:
                print(f"Error loading .keras model: {str(e)}")
                
        # APPROACH 1: Try the DenoisingAutoencoder wrapper first
        try:
            print("Trying DenoisingAutoencoder wrapper...")
            model = DenoisingAutoencoder.load(model_path, custom_objects=custom_objects)
            print("Successfully loaded model with DenoisingAutoencoder wrapper")
            return model
        except Exception as e1:
            print(f"First loading attempt failed: {str(e1)}")
        
        # APPROACH 2: Try direct loading without compilation
        try:
            print("Trying direct loading without compilation...")
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print("Successfully loaded model directly")
            
            # Wrap in a simple object with predict method
            class SimpleModelWrapper:
                def __init__(self, keras_model):
                    self.model = keras_model
                
                def predict(self, x):
                    return self.model.predict(x)
            
            return SimpleModelWrapper(model)
        except Exception as e2:
            print(f"Second loading attempt failed: {str(e2)}")
        
        # APPROACH 3: If it's an HDF5 file and previous attempts failed, try to rebuild the model from weights
        if model_path.endswith('.h5'):
            try:
                print("Trying to rebuild the model and load weights...")
                # Create a new instance of the denoising model
                new_model = DenoisingAutoencoder(input_shape=(256, 256, 3))
                new_model.build()
                
                # Open the HDF5 file and extract weights without loading the full model
                import h5py
                with h5py.File(model_path, 'r') as f:
                    # Check if 'model_weights' group exists
                    if 'model_weights' in f:
                        print("Found weights in the HDF5 file, loading weights only...")
                        # Load weights using low-level API
                        new_model.model.load_weights(model_path)
                        print("Successfully loaded weights")
                        return new_model
            except Exception as e3:
                print(f"Third loading attempt failed: {str(e3)}")
        
        # APPROACH 4: Try loading using temporary SavedModel conversion
        try:
            print("Trying SavedModel conversion...")
            import tempfile
            import os
            
            # Import h5py to check if the file is a valid HDF5 file
            import h5py
            
            # Check if the file is a valid HDF5 file
            try:
                with h5py.File(model_path, 'r') as f:
                    # Check if model_weights exists in the file
                    if 'model_weights' in f:
                        print("Valid HDF5 file with model weights found")
                        
                        # Create a temporary directory
                        temp_dir = tempfile.mkdtemp()
                        temp_model_path = os.path.join(temp_dir, 'temp_model')
                        
                        # Create a model with the same architecture
                        input_shape = (256, 256, 3)  # Default input shape
                        new_model = DenoisingAutoencoder(input_shape=input_shape)
                        new_model.build()
                        
                        # Try to load the weights
                        try:
                            new_model.model.load_weights(model_path)
                            print("Successfully loaded weights from HDF5 file")
                            
                            # Save the model in SavedModel format
                            new_model.model.save(temp_model_path, save_format='tf')
                            
                            # Load the saved model
                            model = tf.keras.models.load_model(
                                temp_model_path,
                                custom_objects=custom_objects,
                                compile=False
                            )
                            
                            print("Successfully loaded model from temporary SavedModel")
                            
                            # Cleanup
                            import shutil
                            shutil.rmtree(temp_dir)
                            
                            # Wrap the model
                            return SimpleModelWrapper(model)
                        except Exception as inner_e:
                            print(f"Failed to load weights from HDF5: {str(inner_e)}")
            except Exception as h5_e:
                print(f"Not a valid HDF5 file or error reading file: {str(h5_e)}")
        except Exception as e4:
            print(f"Fourth loading attempt failed: {str(e4)}")
        
        # FALLBACK: Create a new model if all loading attempts fail
        try:
            print("All loading attempts failed. Creating new model as fallback...")
            fallback_model = DenoisingAutoencoder(input_shape=(256, 256, 3))
            fallback_model.build()
            
            # Wrap in a class that adds a warning to predictions
            class FallbackModelWrapper:
                def __init__(self, autoencoder):
                    self.model = autoencoder.model
                
                def predict(self, x):
                    print("WARNING: Using untrained fallback model for prediction")
                    return self.model.predict(x)
            
            return FallbackModelWrapper(fallback_model)
        except Exception as e5:
            print(f"Fallback model creation failed: {str(e5)}")
        
        return None
    except Exception as e:
        print(f"Error during model loading process: {str(e)}")
        return None

def find_best_model():
    """
    Find the best available model by trying to load all models in saved_models directory.
    
    Returns:
        Loaded model or None if no model could be loaded
    """
    # First try the converted model path (new format)
    model = try_load_model(MODEL_PATH)
    if model:
        print(f"Successfully loaded converted model from {MODEL_PATH}")
        return model
    
    # Then try the fallback model path (original Kaggle model)
    model = try_load_model(FALLBACK_MODEL_PATH)
    if model:
        print(f"Successfully loaded fallback model from {FALLBACK_MODEL_PATH}")
        return model
    
    print(f"Default models could not be loaded, trying alternatives...")
    
    # Get all model files sorted by modification time (newest first)
    model_files = glob.glob('saved_models/*.h5') + glob.glob('converted_models/*.keras') + glob.glob('converted_models/saved_model')
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Try each model file until one loads successfully
    for model_file in model_files:
        if model_file != MODEL_PATH and model_file != FALLBACK_MODEL_PATH:  # Skip the default we already tried
            model = try_load_model(model_file)
            if model:
                print(f"Successfully loaded alternative model: {model_file}")
                return model
    
    print("Could not load any model, will use fallback processing")
    return None

def process_image(image_path):
    """
    Process an uploaded image - create noisy version, denoise it, and generate comparison images.
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        Dictionary with paths to the processed images and metrics
    """
    # Generate unique filenames for the results
    noisy_filename = f"noisy_{uuid.uuid4().hex}.png"
    denoised_filename = f"denoised_{uuid.uuid4().hex}.png"
    combined_filename = f"combined_{uuid.uuid4().hex}.png"
    
    # Define paths for the result files
    noisy_path = os.path.join(app.config['RESULTS_FOLDER'], noisy_filename)
    denoised_path = os.path.join(app.config['RESULTS_FOLDER'], denoised_filename)
    combined_path = os.path.join(app.config['RESULTS_FOLDER'], combined_filename)
    
    # Ensure TARGET_SIZE is defined (256x256)
    TARGET_SIZE = (256, 256)
    
    try:
        # Load and preprocess the image
        original_img = Image.open(image_path).convert('RGB')
        
        # Resize the image to the target size
        original_img = original_img.resize(TARGET_SIZE, Image.LANCZOS)
        
        # Save the original and resized image
        original_img.save(image_path)
        
        # Load the best available model
        model = find_best_model()
        
        # Process only if a model is available
        if model:
            # Convert the image to a numpy array and normalize to 0-1
            image = np.array(original_img).astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            # Validate the image shape
            if image.shape != (1, TARGET_SIZE[0], TARGET_SIZE[1], 3):
                raise ValueError(f"Image has incorrect shape {image.shape}, expected (1, {TARGET_SIZE[0]}, {TARGET_SIZE[1]}, 3)")
            
            # Add realistic noise to create a noisy version
            try:
                # Try to use NoiseGenerator, but don't let it crash the whole process
                noise_gen = NoiseGenerator(noise_type='gaussian', 
                                          noise_params={'mean': 0, 'std': 0.08})
                # Add synthetic noise
                noisy = noise_gen.add_noise(image)
                # Save the noisy image
                noisy_img = Image.fromarray((noisy[0] * 255).astype(np.uint8))
                noisy_img.save(noisy_path)
            except Exception as noise_err:
                print(f"Error adding noise: {str(noise_err)}. Using simpler noise approach.")
                # Fall back to simple noise addition if NoiseGenerator fails
                noise = np.random.normal(0, 0.1, image.shape)
                noisy = np.clip(image + noise, 0, 1)
                noisy_img = Image.fromarray((noisy[0] * 255).astype(np.uint8))
                noisy_img.save(noisy_path)
            
            # Process image with model - ensure input has the right shape
            print(f"Processing image with shape: {image.shape}")
            
            # Our model expects input in 0-1 range
            denoised = model.predict(image)
            
            # Print denoised image stats for debugging
            if isinstance(denoised, np.ndarray):
                print(f"Denoised array: min={np.min(denoised)}, max={np.max(denoised)}, shape={denoised.shape}")
                
                # Extract the image from batch dimension if needed
                if len(denoised.shape) == 4:
                    denoised = denoised[0]
                
                # If denoised values are close to flat (gray), print a warning
                std_values = np.std(denoised, axis=(0, 1))
                print(f"Standard deviation per channel: {std_values}")
                if np.all(std_values < 0.01):
                    print("WARNING: Denoised image appears to be almost flat (low variation)")
                
                # Clip values between 0-1 and convert to uint8
                denoised = np.clip(denoised, 0, 1)
                denoised_uint8 = (denoised * 255).astype(np.uint8)
                print(f"Denoised uint8 array: min={np.min(denoised_uint8)}, max={np.max(denoised_uint8)}")
                
                # Convert to PIL image and save
                denoised_img = Image.fromarray(denoised_uint8)
            else:
                print(f"Unexpected denoised output type: {type(denoised)}")
                # Create a fallback gray image
                denoised_img = Image.new('RGB', TARGET_SIZE, (128, 128, 128))
        else:
            # Fallback: Basic image processing without the model
            print("Using fallback image processing without model")
            
            # Simple denoising using PIL's built-in filters
            # Convert to grayscale and back to reduce some noise
            denoised_img = original_img.convert('L').convert('RGB')
            
            # Apply a slight blur to reduce noise
            from PIL import ImageFilter
            denoised_img = denoised_img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Also create a noisy version for comparison
            noise = np.random.normal(0, 15, np.array(original_img).shape).astype(np.int32)
            noisy = np.clip(np.array(original_img) + noise, 0, 255).astype(np.uint8)
            noisy_img = Image.fromarray(noisy)
            noisy_img.save(noisy_path)
        
        # Save the denoised image
        denoised_img.save(denoised_path)
        
        # Create combined visualization
        try:
            # Calculate metrics if possible
            metrics = calculate_metrics(image_path, denoised_path)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            metrics = {'PSNR': 0, 'SSIM': 0}
        
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
        error_img = Image.new('RGB', TARGET_SIZE, (200, 200, 200))
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

def calculate_metrics(original_path, denoised_path):
    """
    Calculate image quality metrics between original and denoised images.
    
    Args:
        original_path: Path to the original image
        denoised_path: Path to the denoised image
        
    Returns:
        Dictionary with PSNR and SSIM metrics
    """
    try:
        # Load images
        original_img = Image.open(original_path).convert('RGB')
        denoised_img = Image.open(denoised_path).convert('RGB')
        
        # Ensure both images are the same size
        if original_img.size != denoised_img.size:
            print(f"Resizing images for metric calculation. Original: {original_img.size}, Denoised: {denoised_img.size}")
            # Resize original to match denoised image size
            original_img = original_img.resize(denoised_img.size, Image.LANCZOS)
        
        # Convert to numpy arrays and normalize
        original = np.array(original_img).astype(np.float32) / 255.0
        denoised = np.array(denoised_img).astype(np.float32) / 255.0
        
        # Calculate PSNR
        mse = np.mean((original - denoised) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        # Calculate SSIM
        ssim = tf.reduce_mean(tf.image.ssim(
            tf.convert_to_tensor([original]), 
            tf.convert_to_tensor([denoised]), 
            max_val=1.0
        )).numpy()
        
        return {
            'PSNR': float(psnr),
            'SSIM': float(ssim)
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'PSNR': 0,
            'SSIM': 0
        }

if __name__ == '__main__':
    app.run(debug=True) 