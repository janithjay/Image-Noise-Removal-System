"""
Script for fine-tuning the image denoising model.

This script provides options for fine-tuning an existing model with control over:
- Learning rate
- Number of epochs
- Batch size
- Data augmentation
- Specific noise types
- Visualization
"""

import os
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import gc
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Import project modules
from config import TRAINING_CONFIG, PATHS, MODEL_CONFIG
from model.autoencoder import DenoisingAutoencoder
from model.trainer import Trainer
from utils.data_loader import DataLoader, NoiseGenerator
from utils.visualizer import (
    calculate_metrics, 
    generate_comparison_grid, 
    visualize_filters, 
    visualize_feature_maps
)

def parse_args():
    """Parse command line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(description='Fine-tune the denoising model')
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the model to fine-tune')
    
    # Optional arguments
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for fine-tuning (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for fine-tuning (default: 20)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--noise-types', type=str, default='gaussian',
                        help='Comma-separated list of noise types to use for training '
                             '(choices: gaussian, poisson, salt_pepper, speckle, all)')
    parser.add_argument('--augmentation', action='store_true',
                        help='Enable data augmentation for training')
    parser.add_argument('--visualize-layers', action='store_true',
                        help='Visualize model layers and feature maps')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory to save results (default: PATHS["RESULTS"])')
    parser.add_argument('--image-dir', type=str, default='clean_images',
                        help='Directory containing clean images for training (default: clean_images)')
    parser.add_argument('--max-images', type=int, default=800,
                        help='Maximum number of images to use for training (default: 800)')
    
    return parser.parse_args()

def load_images(image_dir, max_images, target_size):
    """
    Load images from a directory.
    
    Args:
        image_dir: Directory containing images
        max_images: Maximum number of images to load
        target_size: Target size for resizing images (height, width)
        
    Returns:
        Numpy array of loaded images
    """
    if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
        print(f"Directory {image_dir} not found. Using synthetic data.")
        return None
    
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No images found in {image_dir}. Using synthetic data.")
        return None
    
    # Limit number of images
    if len(image_files) > max_images:
        print(f"Found {len(image_files)} images, limiting to {max_images}.")
        image_files = image_files[:max_images]
    else:
        print(f"Found {len(image_files)} images.")
    
    # Load and preprocess images
    images = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
            images.append(img_array)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    if not images:
        print("Failed to load any images. Using synthetic data.")
        return None
    
    return np.array(images)

def generate_synthetic_data(num_samples, input_shape):
    """
    Generate synthetic data for training.
    
    Args:
        num_samples: Number of samples to generate
        input_shape: Shape of each sample (height, width, channels)
        
    Returns:
        Numpy array of synthetic images
    """
    print(f"Generating {num_samples} synthetic images of shape {input_shape}")
    return np.random.rand(num_samples, *input_shape)

def get_noise_types(noise_arg):
    """
    Parse noise types from argument.
    
    Args:
        noise_arg: Comma-separated string of noise types or 'all'
        
    Returns:
        List of noise types
    """
    all_noise_types = ['gaussian', 'poisson', 'salt_pepper', 'speckle']
    
    if noise_arg.lower() == 'all':
        return all_noise_types
    
    noise_types = [n.strip() for n in noise_arg.split(',')]
    valid_noise_types = [n for n in noise_types if n in all_noise_types]
    
    if not valid_noise_types:
        print(f"No valid noise types found in '{noise_arg}'. Using 'gaussian' as default.")
        return ['gaussian']
    
    return valid_noise_types

def main():
    """Main function for fine-tuning the model."""
    # Parse command line arguments
    args = parse_args()
    
    print(f"Starting fine-tuning of model: {args.model_path}")
    
    # Set up GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        print("No GPU found. Using CPU for training (this will be slower)")
    
    # Clean up memory
    gc.collect()
    
    # Setup directories
    os.makedirs(PATHS['MODELS'], exist_ok=True)
    results_dir = args.results_dir or PATHS['RESULTS']
    os.makedirs(results_dir, exist_ok=True)
    
    # Define target size from MODEL_CONFIG
    input_shape = MODEL_CONFIG['input_shape']
    target_size = (input_shape[0], input_shape[1])
    
    # Load the model to fine-tune
    try:
        print(f"Loading model from {args.model_path}")
        model = DenoisingAutoencoder.load(args.model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load training data
    clean_images = load_images(args.image_dir, args.max_images, target_size)
    
    # If no images were loaded, generate synthetic data
    if clean_images is None:
        clean_images = generate_synthetic_data(400, input_shape)
    
    # Setup noise generator with specified noise types
    noise_types = get_noise_types(args.noise_types)
    print(f"Using noise types: {', '.join(noise_types)}")
    
    noise_gen = NoiseGenerator(noise_type=noise_types[0])
    noisy_images = []
    for noise_type in noise_types:
        noise_gen.noise_type = noise_type
        noisy_batch = noise_gen.add_noise(clean_images)
        noisy_images.append(noisy_batch)
    
    # If multiple noise types were used, average them
    if len(noise_types) > 1:
        noisy_images = np.mean(noisy_images, axis=0)
    else:
        noisy_images = noisy_images[0]
    
    # Split into training and validation sets
    val_split = TRAINING_CONFIG['VALIDATION_SPLIT']
    split_idx = int(len(clean_images) * (1 - val_split))
    
    x_train = noisy_images[:split_idx]  # Noisy images as input
    y_train = clean_images[:split_idx]   # Clean images as target
    
    x_val = noisy_images[split_idx:]     # Noisy images as input for validation
    y_val = clean_images[split_idx:]     # Clean images as target for validation
    
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    # Initialize trainer
    trainer = Trainer(model)
    trainer.batch_size = args.batch_size
    trainer.epochs = args.epochs
    trainer.use_data_augmentation = args.augmentation
    
    if args.augmentation:
        print("Data augmentation is enabled")
    
    # Fine-tune the model
    print(f"Fine-tuning for {args.epochs} epochs with learning rate {args.learning_rate}")
    history = trainer.fine_tune(
        x_train, y_train, 
        x_val, y_val,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    
    # Plot and save training history
    history_path = os.path.join(results_dir, 'fine_tuning_history.png')
    trainer.plot_training_history(history, save_path=history_path)
    
    # Save the fine-tuned model
    fine_tuned_model_path = os.path.join(PATHS['MODELS'], 'fine_tuned_model.h5')
    trainer.model.save(fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")
    
    # Generate comparison visualization
    results_path = os.path.join(results_dir, 'fine_tuning_results.png')
    num_examples = min(3, len(x_val))
    trainer.plot_results(x_val[:num_examples], y_val[:num_examples], save_path=results_path, samples=num_examples)
    
    # Generate detailed grid visualization
    grid_path = os.path.join(results_dir, 'detailed_results.png')
    y_pred = trainer.predict(x_val[:9])  # Get predictions for more samples
    generate_comparison_grid(
        y_val[:9],     # Original
        x_val[:9],     # Noisy
        y_pred,        # Denoised
        save_path=grid_path,
        grid_size=(3, 3)
    )
    
    # Visualize model layers if requested
    if args.visualize_layers:
        print("Generating model visualizations...")
        
        # Find a convolutional layer to visualize
        conv_layer_name = None
        for layer in model.model.layers:
            if 'conv' in layer.name.lower() and len(layer.get_weights()) > 0:
                conv_layer_name = layer.name
                break
        
        if conv_layer_name:
            # Visualize filters
            filters_path = os.path.join(results_dir, 'filters_visualization.png')
            visualize_filters(model.model, conv_layer_name, save_path=filters_path)
            
            # Visualize feature maps for a sample image
            if len(x_val) > 0:
                feature_maps_path = os.path.join(results_dir, 'feature_maps.png')
                visualize_feature_maps(model.model, x_val[0], conv_layer_name, save_path=feature_maps_path)
    
    # Calculate overall metrics on validation set
    print("\nValidation Metrics After Fine-tuning:")
    metrics_sum = {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
    
    # Predict on entire validation set
    val_predictions = trainer.predict(x_val)
    
    # Calculate average metrics
    for i in range(len(y_val)):
        metrics = calculate_metrics(y_val[i], val_predictions[i])
        for key in metrics_sum:
            metrics_sum[key] += metrics.get(key, 0)
    
    # Calculate averages
    for key in metrics_sum:
        metrics_sum[key] /= len(y_val)
        print(f"Average {key}: {metrics_sum[key]:.4f}")
    
    print(f"\nFine-tuning complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main() 