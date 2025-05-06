"""
Main module for image noise removal system.
Provides command-line functionality to process images.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from utils.resolution_adapter import ResolutionAdapter

def load_denoising_model(model_path='saved_models/denoising_model_best.keras'):
    """
    Load the denoising model from the specified path.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded TensorFlow model
    """
    print(f"Loading model from {model_path}")
    try:
        # Enable unsafe deserialization since we trust our own models
        tf.keras.config.enable_unsafe_deserialization()
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully")
        
        # Create a model wrapper
        class ModelWrapper:
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
                    input_data: Input image as numpy array
                    
                Returns:
                    Denoised image as numpy array
                """
                # Use model for prediction
                print(f"Running prediction with input shape: {input_data.shape}")
                output = self.model.predict(input_data)
                print(f"Prediction output shape: {output.shape}")
                return output
                
        # Create and return the resolution adapter
        return ResolutionAdapter(ModelWrapper(model), target_size=None, overlap=0.25)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def denoise_image(image_path, output_path=None):
    """
    Denoise a single image using the trained model.
    
    Args:
        image_path: Path to the image to denoise
        output_path: Path to save the denoised image (optional)
    
    Returns:
        Path to the denoised image
    """
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Load model (resolution adapter)
    model = load_denoising_model()
    if model is None:
        raise ValueError("Failed to load denoising model")
    
    # Process image using resolution adapter
    print("Processing image...")
    denoised_img = model.process_image(img)
    
    # Set default output path if not provided
    if output_path is None:
        filename, ext = os.path.splitext(image_path)
        output_path = f"{filename}_denoised{ext}"
    
    # Save the denoised image
    print(f"Saving denoised image to: {output_path}")
    denoised_img.save(output_path)
    
    # Display original and denoised images side by side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(denoised_img))
    plt.title('Denoised Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return output_path

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Image Noise Removal System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Denoise command
    denoise_parser = subparsers.add_parser('denoise', help='Denoise an image')
    denoise_parser.add_argument('image_path', help='Path to the input image')
    denoise_parser.add_argument('--output', '-o', help='Path to save the denoised image')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'denoise':
        try:
            denoised_path = denoise_image(args.image_path, args.output)
            print(f"Image denoised successfully and saved to: {denoised_path}")
        except Exception as e:
            print(f"Error denoising image: {e}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 