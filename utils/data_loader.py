"""
Utilities for loading and processing image data.
"""

import os
import numpy as np
from keras.utils import img_to_array, load_img
from keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split

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