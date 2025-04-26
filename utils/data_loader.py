"""
Utilities for loading and processing image data.
"""

import os
import numpy as np
from keras.utils import img_to_array, load_img
from keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf

from config import DATA_CONFIG

class DataLoader:
    """
    Handles loading and preprocessing of image data.
    """
    def __init__(self, target_size=None):
        """
        Initialize the data loader.
        
        Args:
            target_size: Size to resize images to (height, width)
        """
        self.target_size = target_size or DATA_CONFIG['target_size']
    
    def load_images_from_directory(self, directory, max_samples=None):
        """
        Load images from a directory.
        
        Args:
            directory: Directory containing images
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            numpy array of loaded images
        """
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        
        # Limit the number of samples if specified
        if max_samples is not None and max_samples < len(image_paths):
            image_paths = image_paths[:max_samples]
        
        # Load and preprocess images
        images = []
        for path in image_paths:
            img = load_img(path, target_size=self.target_size)
            img_array = img_to_array(img)
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            images.append(img_array)
        
        return np.array(images)
    
    def load_single_image(self, image_path):
        """
        Load a single image from a file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        img = load_img(image_path, target_size=self.target_size)
        img_array = img_to_array(img)
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def prepare_train_val_data(self, clean_images, noisy_images=None, test_size=0.2, random_state=42):
        """
        Split data into training and validation sets.
        
        Args:
            clean_images: Clean images (targets)
            noisy_images: Noisy images (inputs), if None will be generated
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (x_train, x_val, y_train, y_val)
        """
        # If noisy images are not provided, generate them
        if noisy_images is None:
            from .preprocessor import NoiseGenerator
            noise_gen = NoiseGenerator()
            noisy_images = noise_gen.add_noise(clean_images)
        
        # Split data
        x_train, x_val, y_train, y_val = train_test_split(
            noisy_images, clean_images, test_size=test_size, random_state=random_state
        )
        
        return x_train, x_val, y_train, y_val

class NoiseGenerator:
    """
    Class for adding various types of noise to images.
    """
    def __init__(self, noise_type=None, noise_params=None):
        """
        Initialize the noise generator.
        
        Args:
            noise_type: Type of noise to add ('gaussian', 'salt_pepper', 'speckle', 'poisson')
            noise_params: Parameters for the noise
        """
        self.noise_type = noise_type or DATA_CONFIG['default_noise']
        self.noise_params = noise_params or self._get_default_params()
    
    def _get_default_params(self):
        """Get default parameters for the selected noise type."""
        if self.noise_type == 'gaussian':
            return DATA_CONFIG['gaussian_params']
        elif self.noise_type == 'salt_pepper':
            return DATA_CONFIG['salt_pepper_params']
        elif self.noise_type == 'speckle':
            return DATA_CONFIG['speckle_params']
        else:
            # Default to Gaussian
            return DATA_CONFIG['gaussian_params']
    
    def add_gaussian_noise(self, images, mean=0, std=0.1):
        """
        Add Gaussian noise to images.
        
        Args:
            images: Numpy array of images (batch, height, width, channels)
            mean: Mean of the Gaussian noise
            std: Standard deviation of the Gaussian noise
            
        Returns:
            Noisy images
        """
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, images.shape)
        # Add noise to images
        noisy_images = images + noise
        # Clip to valid range
        return np.clip(noisy_images, 0, 1)
    
    def add_salt_pepper_noise(self, images, amount=0.05):
        """
        Add salt and pepper noise to images.
        
        Args:
            images: Numpy array of images (batch, height, width, channels)
            amount: Fraction of the image to be replaced with noise
            
        Returns:
            Noisy images
        """
        noisy_images = np.copy(images)
        # Salt mode
        salt_mask = np.random.random(images.shape) < amount / 2
        noisy_images[salt_mask] = 1
        
        # Pepper mode
        pepper_mask = np.random.random(images.shape) < amount / 2
        noisy_images[pepper_mask] = 0
        
        return noisy_images
    
    def add_speckle_noise(self, images, mean=0, var=0.1):
        """
        Add speckle noise to images.
        
        Args:
            images: Numpy array of images (batch, height, width, channels)
            mean: Mean of the Gaussian noise
            var: Variance of the Gaussian noise
            
        Returns:
            Noisy images
        """
        # Generate speckle noise (multiplicative noise)
        noise = np.random.normal(mean, var**0.5, images.shape)
        # Add noise to images
        noisy_images = images + images * noise
        # Clip to valid range
        return np.clip(noisy_images, 0, 1)
    
    def add_poisson_noise(self, images, lambda_val=30):
        """
        Add Poisson noise to images.
        
        Args:
            images: Numpy array of images (batch, height, width, channels)
            lambda_val: Lambda parameter for Poisson distribution
            
        Returns:
            Noisy images
        """
        # Scale images to suitable range for Poisson noise
        scaled_images = lambda_val * images
        # Generate Poisson noise
        noisy_images = np.random.poisson(scaled_images) / lambda_val
        # Clip to valid range
        return np.clip(noisy_images, 0, 1)
    
    def add_noise(self, images, noise_type=None, noise_params=None):
        """
        Add noise to images based on specified noise type.
        
        Args:
            images: Numpy array of images (batch, height, width, channels)
            noise_type: Type of noise to add (if None, use the instance's default)
            noise_params: Parameters for the noise (if None, use the instance's default)
            
        Returns:
            Noisy images
        """
        noise_type = noise_type or self.noise_type
        noise_params = noise_params or self.noise_params
        
        if noise_type == 'gaussian':
            mean = noise_params.get('mean', 0)
            std = noise_params.get('std', 0.1)
            return self.add_gaussian_noise(images, mean, std)
        
        elif noise_type == 'salt_pepper':
            amount = noise_params.get('amount', 0.05)
            return self.add_salt_pepper_noise(images, amount)
        
        elif noise_type == 'speckle':
            mean = noise_params.get('mean', 0)
            var = noise_params.get('var', 0.1)
            return self.add_speckle_noise(images, mean, var)
        
        elif noise_type == 'poisson':
            lambda_val = noise_params.get('lambda_val', 30)
            return self.add_poisson_noise(images, lambda_val)
        
        else:
            # Default to Gaussian noise
            return self.add_gaussian_noise(images)