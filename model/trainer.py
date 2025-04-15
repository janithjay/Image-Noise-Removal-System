"""
Trainer module for handling model training and evaluation.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image

from config import TRAINING_CONFIG, PATHS
from utils.visualizer import combine_images, calculate_metrics

class Trainer:
    """
    Handles training and evaluation of the model.
    """
    def __init__(self, model, save_dir=None):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            save_dir: Directory to save model checkpoints and results
        """
        self.model = model
        self.save_dir = save_dir or PATHS['model_save_path']
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.epochs = TRAINING_CONFIG['epochs']
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def _get_callbacks(self, checkpoint_path):
        """
        Get training callbacks.
        
        Args:
            checkpoint_path: Path to save model checkpoints
            
        Returns:
            List of callbacks
        """
        callbacks = [
            # Save the best model
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, x_train, y_train, x_val, y_val):
        """
        Train the model.
        
        Args:
            x_train: Training inputs
            y_train: Training targets
            x_val: Validation inputs
            y_val: Validation targets
            
        Returns:
            Training history
        """
        # Define checkpoint path
        checkpoint_path = os.path.join(self.save_dir, 'best_model.h5')
        
        # Get callbacks
        callbacks = self._get_callbacks(checkpoint_path)
        
        # Train the model
        history = self.model.model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        self.model.model.save(os.path.join(self.save_dir, 'final_model.h5'))
        
        return history
    
    def evaluate(self, x_test, y_test):
        """
        Evaluate the model on the test set.
        
        Args:
            x_test: Test inputs
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        return self.model.model.evaluate(x_test, y_test, verbose=1)
    
    def predict(self, x):
        """
        Make predictions with the model.
        
        Args:
            x: Input data
            
        Returns:
            Model predictions
        """
        return self.model.model.predict(x)
    
    def plot_results(self, x, y_true, save_path=None):
        """
        Plot original, noisy, and denoised images with metrics using the combined visualization.
        
        Args:
            x: Noisy images
            y_true: Ground truth clean images
            save_path: Path to save the plot
        """
        # Make predictions
        y_pred = self.predict(x)
        
        # Select a few random samples
        n_samples = min(3, len(x))
        indices = np.random.choice(len(x), n_samples, replace=False)
        
        # Create a directory for individual result images
        results_dir = os.path.dirname(save_path) if save_path else os.path.join(self.save_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create a figure for all samples
        plt.figure(figsize=(15, 5 * n_samples))
        
        for i, idx in enumerate(indices):
            # Save the images to temporary files
            temp_original_path = os.path.join(results_dir, f'temp_original_{idx}.png')
            temp_noisy_path = os.path.join(results_dir, f'temp_noisy_{idx}.png')
            temp_denoised_path = os.path.join(results_dir, f'temp_denoised_{idx}.png')
            temp_combined_path = os.path.join(results_dir, f'temp_combined_{idx}.png')
            
            # Convert to PIL images and save
            original_img = array_to_img(y_true[idx])
            noisy_img = array_to_img(x[idx])
            denoised_img = array_to_img(np.clip(y_pred[idx], 0, 1))
            
            original_img.save(temp_original_path)
            noisy_img.save(temp_noisy_path)
            denoised_img.save(temp_denoised_path)
            
            # Calculate metrics
            metrics = calculate_metrics(y_true[idx], y_pred[idx])
            
            # Create combined image
            combine_images(
                temp_original_path,
                temp_noisy_path,
                temp_denoised_path,
                save_path=temp_combined_path,
                metrics=metrics
            )
            
            # Display combined image
            plt.subplot(n_samples, 1, i + 1)
            combined_img = plt.imread(temp_combined_path)
            plt.imshow(combined_img)
            plt.axis('off')
            plt.title(f'Sample {idx} (PSNR: {metrics["PSNR"]:.2f} dB, SSIM: {metrics["SSIM"]:.4f})')
        
        plt.tight_layout()
        
        # Save the figure with all samples
        if save_path:
            plt.savefig(save_path)
        
        # Clean up temporary files
        for idx in indices:
            temp_paths = [
                os.path.join(results_dir, f'temp_original_{idx}.png'),
                os.path.join(results_dir, f'temp_noisy_{idx}.png'),
                os.path.join(results_dir, f'temp_denoised_{idx}.png'),
                os.path.join(results_dir, f'temp_combined_{idx}.png')
            ]
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
        
        plt.show()
        
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath) 