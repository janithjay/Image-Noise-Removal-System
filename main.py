"""
Main module for image noise removal system.
Provides functionality to train a model and process images.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image
import cv2

from model import DenoisingAutoencoder, Trainer
from utils.data_loader import DataLoader
from utils.preprocessor import NoiseGenerator, ImagePreprocessor, TraditionalDenoiser
from utils.visualizer import combine_images, calculate_metrics
from utils.enhanced_denoiser import advanced_denoising, ensemble_denoising, detail_preserving_denoising, enhance_image_details

def train_model(data_dir, model_save_dir, epochs=None):
    """
    Train a denoising autoencoder model.
    
    Args:
        data_dir: Directory containing training images
        model_save_dir: Directory to save the trained model
        epochs: Number of training epochs (overrides config)
    """
    # Load data
    data_loader = DataLoader()
    print(f"Loading clean images from {data_dir}")
    clean_images = data_loader.load_images_from_directory(data_dir)
    
    if len(clean_images) == 0:
        print(f"No images found in {data_dir}")
        return
    
    print(f"Loaded {len(clean_images)} images")
    
    # Generate noisy images
    print("Generating noisy images")
    noise_gen = NoiseGenerator()
    noisy_images = noise_gen.add_noise(clean_images)
    
    # Prepare train/validation data
    print("Preparing training and validation data")
    x_train, x_val, y_train, y_val = data_loader.prepare_train_val_data(
        clean_images, noisy_images
    )
    
    # Create and compile model
    print("Creating model")
    model = DenoisingAutoencoder()
    model.summary()
    
    # Train model
    print("Training model")
    trainer = Trainer(model, save_dir=model_save_dir)
    
    if epochs is not None:
        trainer.epochs = epochs
    
    history = trainer.train(x_train, y_train, x_val, y_val)
    
    # Plot results
    print("Plotting results")
    results_dir = os.path.join(model_save_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    trainer.plot_results(
        x_val[:5], y_val[:5], 
        save_path=os.path.join(results_dir, 'sample_results.png')
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    
    print(f"Model saved to {model_save_dir}")
    print(f"Results saved to {results_dir}")

def denoise_image(image_path, model_path, output_path=None):
    """
    Denoise a single image using a trained model.
    
    Args:
        image_path: Path to the noisy image
        model_path: Path to the trained model
        output_path: Path to save denoised image (optional)
    
    Returns:
        Denoised image as PIL Image
    """
    # Load and preprocess image
    data_loader = DataLoader()
    preprocessor = ImagePreprocessor()
    
    print(f"Loading image: {image_path}")
    # Load image as original clean image
    original_img = Image.open(image_path)
    original_array = img_to_array(original_img) / 255.0
    
    # Convert to model input format
    input_img = preprocessor.preprocess_image(original_array)
    input_img = np.expand_dims(input_img, axis=0)
    
    # Generate noisy version
    print("Generating noisy version")
    noise_gen = NoiseGenerator()
    noisy_array = noise_gen.add_noise(input_img)[0]
    
    # Save the noisy image
    noisy_path = image_path.rsplit('.', 1)[0] + '_noisy.' + image_path.rsplit('.', 1)[1]
    noisy_img = array_to_img(noisy_array)
    noisy_img.save(noisy_path)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = DenoisingAutoencoder.load(model_path)
    
    # Denoise image
    print("Denoising image")
    denoised_array = model.predict(input_img)[0]
    
    # Clip values to valid range
    denoised_array = np.clip(denoised_array, 0, 1)
    
    # Convert to PIL Image
    denoised_img = array_to_img(denoised_array)
    
    # Set default output path if not provided
    if not output_path:
        output_path = image_path.rsplit('.', 1)[0] + '_denoised.' + image_path.rsplit('.', 1)[1]
    
    # Save denoised image
    print(f"Saving denoised image to: {output_path}")
    denoised_img.save(output_path)
    
    # Calculate metrics
    metrics = calculate_metrics(input_img[0], denoised_array)
    print(f"PSNR: {metrics['PSNR']:.2f} dB, SSIM: {metrics['SSIM']:.4f}")
    
    # Create and save combined visualization
    combined_path = image_path.rsplit('.', 1)[0] + '_combined.' + image_path.rsplit('.', 1)[1]
    print(f"Creating combined visualization: {combined_path}")
    combine_images(
        image_path,
        noisy_path,
        output_path,
        save_path=combined_path,
        metrics=metrics
    )
    
    # Display combined image
    combined_img = Image.open(combined_path)
    plt.figure(figsize=(15, 10))
    plt.imshow(np.array(combined_img))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return denoised_img

def enhanced_denoise_image(image_path, output_path=None, method='ensemble', strength=8):
    """
    Denoise an image using enhanced methods that don't require training.
    
    Args:
        image_path: Path to the image to denoise
        output_path: Path to save the denoised image (optional)
        method: Denoising method ('nlm', 'bilateral', 'tv', 'wavelet', 'combined', 'ensemble', 'detail-preserving')
        strength: Denoising strength (1-10)
        
    Returns:
        Denoised image as PIL Image
    """
    # Load image
    print(f"Loading image: {image_path}")
    original_img = Image.open(image_path)
    original_array = img_to_array(original_img) / 255.0
    
    # Add noise if needed
    print("Generating noisy version")
    noise_gen = NoiseGenerator()
    noisy_array = noise_gen.add_noise(np.expand_dims(original_array, axis=0))[0]
    
    # Save the noisy image
    noisy_path = image_path.rsplit('.', 1)[0] + '_noisy.' + image_path.rsplit('.', 1)[1]
    noisy_img = array_to_img(noisy_array)
    noisy_img.save(noisy_path)
    
    # Apply denoising
    print(f"Applying {method} denoising with strength {strength}")
    if method == 'ensemble':
        denoised_array = ensemble_denoising(noisy_array)
    elif method == 'detail-preserving':
        denoised_array = detail_preserving_denoising(noisy_array, strength=strength)
    else:
        denoised_array = advanced_denoising(noisy_array, method=method, strength=strength)
    
    # Enhance details and colors
    print("Enhancing image details")
    denoised_array = enhance_image_details(denoised_array, sharpness=1.5, saturation=1.2)
    
    # Convert to PIL Image
    denoised_img = array_to_img(denoised_array)
    
    # Set default output path if not provided
    if not output_path:
        output_path = image_path.rsplit('.', 1)[0] + '_enhanced_denoised.' + image_path.rsplit('.', 1)[1]
    
    # Save denoised image
    print(f"Saving denoised image to: {output_path}")
    denoised_img.save(output_path)
    
    # Calculate metrics
    metrics = calculate_metrics(original_array, denoised_array)
    print(f"PSNR: {metrics['PSNR']:.2f} dB, SSIM: {metrics['SSIM']:.4f}")
    
    # Create and save combined visualization
    combined_path = image_path.rsplit('.', 1)[0] + '_enhanced_combined.' + image_path.rsplit('.', 1)[1]
    print(f"Creating combined visualization: {combined_path}")
    combine_images(
        image_path,
        noisy_path,
        output_path,
        save_path=combined_path,
        metrics=metrics
    )
    
    # Display combined image
    combined_img = Image.open(combined_path)
    plt.figure(figsize=(15, 10))
    plt.imshow(np.array(combined_img))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return denoised_img

def main():
    """
    Main function to parse command line arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Image Noise Removal System')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data_dir', required=True, help='Directory containing training images')
    train_parser.add_argument('--model_dir', default='saved_models', help='Directory to save model')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    
    # Denoise command
    denoise_parser = subparsers.add_parser('denoise', help='Denoise an image using the trained model')
    denoise_parser.add_argument('--image', required=True, help='Path to the image to denoise')
    denoise_parser.add_argument('--model', required=True, help='Path to the trained model')
    denoise_parser.add_argument('--output', help='Path to save the denoised image')
    
    # Enhanced denoise command (no training required)
    enhanced_parser = subparsers.add_parser('enhance', help='Denoise an image using enhanced methods (no training required)')
    enhanced_parser.add_argument('--image', required=True, help='Path to the image to denoise')
    enhanced_parser.add_argument('--method', default='ensemble', choices=['nlm', 'bilateral', 'tv', 'wavelet', 'combined', 'ensemble', 'detail-preserving'],
                                help='Denoising method')
    enhanced_parser.add_argument('--strength', type=int, default=8, choices=range(1, 11),
                                help='Denoising strength (1-10)')
    enhanced_parser.add_argument('--output', help='Path to save the denoised image')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for path in ['saved_models', 'results']:
        if not os.path.exists(path):
            os.makedirs(path)
    
    if args.command == 'train':
        train_model(args.data_dir, args.model_dir, args.epochs)
    elif args.command == 'denoise':
        denoise_image(args.image, args.model, args.output)
    elif args.command == 'enhance':
        enhanced_denoise_image(args.image, args.output, args.method, args.strength)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 