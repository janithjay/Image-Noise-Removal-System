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

from model import DenoisingAutoencoder, Trainer
from utils.data_loader import DataLoader
from utils.preprocessor import NoiseGenerator, ImagePreprocessor

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
    noisy_img = data_loader.load_single_image(image_path)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = DenoisingAutoencoder.load(model_path)
    
    # Denoise image
    print("Denoising image")
    denoised_img = model.predict(noisy_img)[0]
    
    # Clip values to valid range
    denoised_img = np.clip(denoised_img, 0, 1)
    
    # Convert to PIL Image
    denoised_pil = array_to_img(denoised_img)
    
    # Save denoised image if output path provided
    if output_path:
        print(f"Saving denoised image to: {output_path}")
        denoised_pil.save(output_path)
    
    # Display original and denoised
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(noisy_img[0])
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_img)
    plt.title('Denoised Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return denoised_pil

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
    denoise_parser = subparsers.add_parser('denoise', help='Denoise an image')
    denoise_parser.add_argument('--image', required=True, help='Path to the noisy image')
    denoise_parser.add_argument('--model', required=True, help='Path to the trained model')
    denoise_parser.add_argument('--output', help='Path to save the denoised image')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for path in ['saved_models', 'results']:
        if not os.path.exists(path):
            os.makedirs(path)
    
    if args.command == 'train':
        train_model(args.data_dir, args.model_dir, args.epochs)
    elif args.command == 'denoise':
        denoise_image(args.image, args.model, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 