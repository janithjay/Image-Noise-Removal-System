"""
Script to train the image denoising model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import gc
import argparse

# Import configuration
from config import TRAINING_CONFIG, PATHS
from model.autoencoder import DenoisingAutoencoder  
from model.trainer import Trainer
from utils.data_loader import DataLoader, NoiseGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train or fine-tune the denoising model')
    parser.add_argument('--fine-tune', action='store_true', 
                        help='Fine-tune an existing model instead of training from scratch')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the model to fine-tune (required if --fine-tune is used)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate for fine-tuning (defaults to 10%% of the original)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs for training or fine-tuning')
    return parser.parse_args()

def load_training_data():
    """Load training data from clean_images directory or generate synthetic data."""
    print("Loading and preparing data...")
    data_loader = DataLoader()
    
    # Define image size from config
    input_shape = TRAINING_CONFIG['INPUT_SHAPE']
    target_size = (input_shape[0], input_shape[1])
    
    # Load images from clean_images directory if available
    clean_images_folder = "clean_images"
    if os.path.exists(clean_images_folder) and os.path.isdir(clean_images_folder):
        print(f"Loading images from {clean_images_folder}...")
        
        # Get list of image files
        image_files = [f for f in os.listdir(clean_images_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            print(f"No image files found in {clean_images_folder}, using synthetic data instead")
            use_synthetic_data = True
        else:
            use_synthetic_data = False
            print(f"Found {len(image_files)} images")
            
            # Limit number of images to prevent memory issues
            max_images = 800  # Limit to prevent memory issues
            if len(image_files) > max_images:
                print(f"Using only {max_images} images to prevent memory issues")
                image_files = image_files[:max_images]
            
            # Load and preprocess images
            x_clean = []
            for img_file in image_files:
                img_path = os.path.join(clean_images_folder, img_file)
                try:
                    # Load and resize image
                    img = load_img(img_path, target_size=target_size)
                    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
                    x_clean.append(img_array)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
            
            # Convert to numpy array
            if x_clean:
                x_clean = np.array(x_clean)
                print(f"Loaded {len(x_clean)} images, shape: {x_clean.shape}")
            else:
                print("Failed to load any images, using synthetic data instead")
                use_synthetic_data = True
    else:
        print(f"{clean_images_folder} directory not found, using synthetic data instead")
        use_synthetic_data = True
    
    # Generate synthetic data if needed
    if use_synthetic_data:
        print("Creating synthetic dataset for testing...")
        
        # Create a smaller dataset for faster training
        num_samples = 400  # Number of synthetic samples
        
        # Generate clean images (random patterns for testing)
        x_clean = np.random.rand(num_samples, *input_shape)
    
    # Generate corresponding noisy images
    print("Generating noisy versions of images...")
    noise_gen = NoiseGenerator()
    x_noisy = noise_gen.add_noise(x_clean)
    
    # Split into training and validation - clean images are targets (y), noisy are inputs (x)
    val_split = TRAINING_CONFIG['VALIDATION_SPLIT']
    split_idx = int(len(x_clean) * (1 - val_split))
    
    # For training: noisy images are inputs, clean are targets
    x_train = x_noisy[:split_idx]  # Noisy images as input
    y_train = x_clean[:split_idx]  # Clean images as target
    
    # For validation
    x_val = x_noisy[split_idx:]    # Noisy images as input
    y_val = x_clean[split_idx:]    # Clean images as target
    
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    return x_train, y_train, x_val, y_val

def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    print("Starting model training...")
    
    # Check for GPU availability and configure TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configure TensorFlow to use only first GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Allow memory growth for the GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"GPU error: {e}")
    else:
        print("No GPU found. Using CPU for training (this will be slower)")
    
    # Clean up memory before starting
    gc.collect()
    
    # Set up directories
    os.makedirs(PATHS['TRAINING_DATA'], exist_ok=True)
    os.makedirs(PATHS['MODELS'], exist_ok=True)
    os.makedirs(PATHS['RESULTS'], exist_ok=True)
    
    # Load training data
    x_train, y_train, x_val, y_val = load_training_data()
    
    # Set custom epochs if provided
    epochs = args.epochs if args.epochs else TRAINING_CONFIG['EPOCHS']
    
    # Fine-tuning or training from scratch
    if args.fine_tune:
        if not args.model_path:
            print("Error: --model-path is required when using --fine-tune")
            return
            
        print(f"Fine-tuning existing model: {args.model_path}")
        
        # Load existing model
        try:
            model = DenoisingAutoencoder.load(args.model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
            
        # Initialize trainer with loaded model
        trainer = Trainer(model)
        
        # Override epochs if specified
        if args.epochs:
            trainer.epochs = args.epochs
            
        # Fine-tune the model
        learning_rate = args.learning_rate if args.learning_rate else TRAINING_CONFIG['LEARNING_RATE'] * 0.1
        history = trainer.fine_tune(
            x_train, y_train, 
            x_val, y_val,
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        # Plot training history
        history_path = os.path.join(PATHS['RESULTS'], 'fine_tuning_history.png')
        trainer.plot_training_history(history, save_path=history_path)
        
        # Save fine-tuned model
        model_path = os.path.join(PATHS['MODELS'], 'fine_tuned_model.h5')
        trainer.model.save(model_path)
        print(f"Fine-tuned model saved to {model_path}")
    else:
        print("Training model from scratch")
        
        # Initialize model with architecture from config
        print("Initializing model...")
        model = DenoisingAutoencoder(input_shape=TRAINING_CONFIG['INPUT_SHAPE'])
        model.build()
        model.compile(learning_rate=TRAINING_CONFIG['LEARNING_RATE'])
        
        # Initialize trainer
        trainer = Trainer(model)
        
        # Override epochs if specified
        if args.epochs:
            trainer.epochs = args.epochs
        
        # Train model
        print(f"Training model for {trainer.epochs} epochs...")
        history = trainer.train(x_train, y_train, x_val, y_val)
        
        # Plot training history
        history_path = os.path.join(PATHS['RESULTS'], 'training_history.png')
        trainer.plot_training_history(history, save_path=history_path)
        
        # Save model
        model_path = os.path.join(PATHS['MODELS'], 'final_model.h5')
        trainer.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Clean up memory
    gc.collect()
    
    # Generate visualizations with multiple examples
    print("Generating result visualizations...")
    results_path = os.path.join(PATHS['RESULTS'], 'denoising_results.png')
    
    # Show more examples in the visualization (limited by available validation samples)
    num_examples = min(3, len(x_val))
    trainer.plot_results(x_val[:num_examples], y_val[:num_examples], save_path=results_path, samples=num_examples)
    
    print(f"Training complete. Results saved to {results_path}")

if __name__ == "__main__":
    main() 