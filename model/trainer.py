"""
Trainer module for handling model training and evaluation.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import gc
import time
from tensorflow.keras.optimizers import Adam

# Use absolute import to correctly find the config module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        self.save_dir = save_dir or PATHS['MODELS']
        self.batch_size = TRAINING_CONFIG['BATCH_SIZE']
        self.epochs = TRAINING_CONFIG['EPOCHS']
        self.use_data_augmentation = TRAINING_CONFIG.get('DATA_AUGMENTATION', False)
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Create logs directory for TensorBoard
        self.logs_dir = os.path.join(PATHS['LOGS'], f"run_{int(time.time())}")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
    
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
                patience=TRAINING_CONFIG['EARLY_STOPPING_PATIENCE'],
                restore_best_weights=True,
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(
                log_dir=self.logs_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        
        # Add learning rate scheduler if enabled
        if TRAINING_CONFIG.get('SCHEDULER', False):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=TRAINING_CONFIG.get('SCHEDULER_FACTOR', 0.5),
                    patience=TRAINING_CONFIG.get('SCHEDULER_PATIENCE', 3),
                    min_lr=1e-6,
                    verbose=1
                )
            )
            
        return callbacks
    
    def _create_data_generators(self, x_train, y_train):
        """
        Create data generators for training with augmentation.
        
        Args:
            x_train: Training inputs (noisy images)
            y_train: Training targets (clean images)
            
        Returns:
            Data generator for training
        """
        # Common augmentation parameters - keep it simple for denoising tasks
        data_gen_args = dict(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Create generators for inputs and targets with the same seed
        seed = 42
        
        # Generator for noisy images (inputs)
        image_datagen = ImageDataGenerator(**data_gen_args)
        image_datagen.fit(x_train, augment=True, seed=seed)
        
        # Generator for clean images (targets) - must use same transformations
        target_datagen = ImageDataGenerator(**data_gen_args)
        target_datagen.fit(y_train, augment=True, seed=seed)
        
        # Create generator that yields both inputs and targets
        image_generator = image_datagen.flow(x_train, seed=seed, batch_size=self.batch_size)
        target_generator = target_datagen.flow(y_train, seed=seed, batch_size=self.batch_size)
        
        # Combined generator
        def combined_generator():
            while True:
                x_batch = image_generator.next()
                y_batch = target_generator.next()
                yield x_batch, y_batch
                
        return combined_generator(), image_generator.n // self.batch_size
    
    def fine_tune(self, x_train, y_train, x_val, y_val, learning_rate=None, epochs=None):
        """
        Fine-tune the model with a lower learning rate.
        
        Args:
            x_train: Training inputs
            y_train: Training targets
            x_val: Validation inputs
            y_val: Validation targets
            learning_rate: Optional custom learning rate for fine-tuning
            epochs: Optional number of epochs for fine-tuning
            
        Returns:
            Training history
        """
        # Set a lower learning rate if not provided
        if learning_rate is None:
            learning_rate = TRAINING_CONFIG['LEARNING_RATE'] * 0.1
        
        # Set number of epochs if not provided
        if epochs is None:
            epochs = max(10, self.epochs // 2)  # Default to half the regular epochs
            
        print(f"Fine-tuning for {epochs} epochs with learning rate {learning_rate}")
        
        # Check if the model is a DenoisingAutoencoder or just a standard Keras model
        if hasattr(self.model, 'compile') and callable(getattr(self.model, 'compile')):
            # For our custom DenoisingAutoencoder, use the custom compile method
            self.model.compile(learning_rate=learning_rate)
        else:
            # If it's a plain Keras model, compile it directly
            optimizer = Adam(learning_rate=learning_rate)
            # Get the current loss function
            loss_function = self.model.model.loss if hasattr(self.model, 'model') else 'mse'
            self.model.model.compile(optimizer=optimizer, loss=loss_function)
        
        # Define checkpoint path for fine-tuned model
        checkpoint_path = os.path.join(self.save_dir, 'fine_tuned_model.h5')
        
        # Get callbacks
        callbacks = self._get_callbacks(checkpoint_path)
        
        # Clean up memory before training
        gc.collect()
        
        # Use data augmentation if enabled
        if self.use_data_augmentation:
            print("Using data augmentation for fine-tuning")
            train_gen, steps_per_epoch = self._create_data_generators(x_train, y_train)
            
            # Train with data augmentation
            history = self.model.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without data augmentation
            history = self.model.model.fit(
                x_train, y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        
        # Save the fine-tuned model
        self.model.model.save(checkpoint_path)
        
        # Clear memory after training
        gc.collect()
        
        return history
    
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
        
        # Use the class's epochs value, which might be overridden
        print(f"Training for {self.epochs} epochs with batch size {self.batch_size}")
        
        # Clean up memory before training
        gc.collect()
        
        # Use data augmentation if enabled
        if self.use_data_augmentation:
            print("Using data augmentation for training")
            train_gen, steps_per_epoch = self._create_data_generators(x_train, y_train)
            
            # Train with data augmentation
            history = self.model.model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Train without data augmentation
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
        
        # Clear memory after training
        gc.collect()
        
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
    
    def plot_results(self, x, y_true, save_path=None, samples=3):
        """
        Plot original, noisy, and denoised images with metrics.
        
        Args:
            x: Noisy images
            y_true: Ground truth clean images
            save_path: Path to save the plot
            samples: Number of samples to display
        """
        # Make predictions
        y_pred = self.predict(x)
        
        # Limit samples to available data
        n_samples = min(samples, len(x))
        
        # Create a figure with 3 rows (original, noisy, denoised) and n_samples columns
        fig, axes = plt.subplots(3, n_samples, figsize=(4*n_samples, 10))
        
        # If only one sample, make sure axes is 2D
        if n_samples == 1:
            axes = axes.reshape(3, 1)
        
        for i in range(n_samples):
            # Get the images
            original = np.clip(y_true[i], 0, 1)
            noisy = np.clip(x[i], 0, 1)
            denoised = np.clip(y_pred[i], 0, 1)
            
            # Calculate metrics
            metrics = calculate_metrics(original, denoised)
            psnr = metrics['PSNR']
            ssim = metrics.get('SSIM', 0)
            
            # Plot original image
            axes[0, i].imshow(original)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Plot noisy image
            axes[1, i].imshow(noisy)
            axes[1, i].set_title('Noisy')
            axes[1, i].axis('off')
            
            # Plot denoised image
            axes[2, i].imshow(denoised)
            axes[2, i].set_title(f'Denoised\nPSNR: {psnr:.2f}, SSIM: {ssim:.2f}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.close()  # Close the figure to save memory
        
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history (loss curves).
        
        Args:
            history: Training history
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot learning rate if available
        if 'lr' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.ylabel('Learning Rate')
            plt.xlabel('Epoch')
            plt.yscale('log')
        
        plt.tight_layout()
        
        # Save the figure
        if save_path:
            plt.savefig(save_path)
            print(f"Training history saved to {save_path}")
        
        plt.close()  # Close the figure to save memory 