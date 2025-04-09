"""
Trainer module for handling model training and evaluation.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import TRAINING_CONFIG, PATHS

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
        Plot original noisy images, ground truth, and predictions.
        
        Args:
            x: Noisy images
            y_true: Ground truth clean images
            save_path: Path to save the plot
        """
        # Make predictions
        y_pred = self.predict(x)
        
        # Select a few random samples
        n_samples = min(5, len(x))
        indices = np.random.choice(len(x), n_samples, replace=False)
        
        plt.figure(figsize=(15, 5 * n_samples))
        
        for i, idx in enumerate(indices):
            # Original noisy image
            plt.subplot(n_samples, 3, i * 3 + 1)
            plt.imshow(x[idx])
            plt.title('Noisy')
            plt.axis('off')
            
            # Ground truth
            plt.subplot(n_samples, 3, i * 3 + 2)
            plt.imshow(y_true[idx])
            plt.title('Ground Truth')
            plt.axis('off')
            
            # Prediction
            plt.subplot(n_samples, 3, i * 3 + 3)
            plt.imshow(y_pred[idx])
            plt.title('Denoised')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath) 