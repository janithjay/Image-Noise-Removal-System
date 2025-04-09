"""
Image preprocessing utilities, including noise generation and removal.
"""

import numpy as np
import cv2
from skimage.util import random_noise
from skimage.restoration import denoise_bilateral, denoise_wavelet

from config import DATA_CONFIG

class NoiseGenerator:
    """
    Generate different types of noise to add to images.
    """
    def __init__(self, noise_type=None):
        """
        Initialize the noise generator.
        
        Args:
            noise_type: Type of noise to generate ('gaussian', 'salt_pepper', 'speckle')
        """
        self.noise_type = noise_type or DATA_CONFIG['default_noise']
        self.gaussian_params = DATA_CONFIG['gaussian_params']
        self.salt_pepper_params = DATA_CONFIG['salt_pepper_params']
        self.speckle_params = DATA_CONFIG['speckle_params']
    
    def add_gaussian_noise(self, images):
        """
        Add Gaussian noise to images.
        
        Args:
            images: Images to add noise to
            
        Returns:
            Noisy images
        """
        mean = self.gaussian_params['mean']
        std = self.gaussian_params['std']
        
        # Generate Gaussian noise
        noisy_images = []
        for img in images:
            noisy = random_noise(img, mode='gaussian', mean=mean, var=std**2, clip=True)
            noisy_images.append(noisy)
        
        return np.array(noisy_images)
    
    def add_salt_pepper_noise(self, images):
        """
        Add salt and pepper noise to images.
        
        Args:
            images: Images to add noise to
            
        Returns:
            Noisy images
        """
        amount = self.salt_pepper_params['amount']
        
        # Generate salt and pepper noise
        noisy_images = []
        for img in images:
            noisy = random_noise(img, mode='s&p', amount=amount, clip=True)
            noisy_images.append(noisy)
        
        return np.array(noisy_images)
    
    def add_speckle_noise(self, images):
        """
        Add speckle noise to images.
        
        Args:
            images: Images to add noise to
            
        Returns:
            Noisy images
        """
        mean = self.speckle_params['mean']
        var = self.speckle_params['var']
        
        # Generate speckle noise
        noisy_images = []
        for img in images:
            noisy = random_noise(img, mode='speckle', mean=mean, var=var, clip=True)
            noisy_images.append(noisy)
        
        return np.array(noisy_images)
    
    def add_noise(self, images):
        """
        Add noise to images based on the specified noise type.
        
        Args:
            images: Images to add noise to
            
        Returns:
            Noisy images
        """
        if self.noise_type == 'gaussian':
            return self.add_gaussian_noise(images)
        elif self.noise_type == 'salt_pepper':
            return self.add_salt_pepper_noise(images)
        elif self.noise_type == 'speckle':
            return self.add_speckle_noise(images)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")


class ImagePreprocessor:
    """
    Handles image preprocessing such as resizing, normalization, etc.
    """
    def __init__(self, target_size=None):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Size to resize images to (height, width)
        """
        self.target_size = target_size or DATA_CONFIG['target_size']
    
    def preprocess_image(self, image):
        """
        Preprocess a single image.
        
        Args:
            image: Image to preprocess
            
        Returns:
            Preprocessed image
        """
        # Resize if necessary
        if image.shape[0] != self.target_size[0] or image.shape[1] != self.target_size[1]:
            image = cv2.resize(image, self.target_size[::-1])  # cv2.resize expects (width, height)
        
        # Ensure the image is normalized to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
    
    def preprocess_batch(self, images):
        """
        Preprocess a batch of images.
        
        Args:
            images: Batch of images to preprocess
            
        Returns:
            Preprocessed images
        """
        return np.array([self.preprocess_image(img) for img in images])


class TraditionalDenoiser:
    """
    Apply traditional denoising methods to images.
    
    This class implements common traditional methods for noise reduction,
    useful for comparison with deep learning approaches.
    """
    def __init__(self, method='bilateral'):
        """
        Initialize the denoiser.
        
        Args:
            method: Denoising method ('bilateral', 'wavelet', 'median')
        """
        self.method = method
    
    def denoise(self, image):
        """
        Apply denoising to an image.
        
        Args:
            image: Image to denoise
            
        Returns:
            Denoised image
        """
        if self.method == 'bilateral':
            # Bilateral filtering for RGB images
            return denoise_bilateral(image, sigma_color=0.1, sigma_spatial=15, channel_axis=-1)
        
        elif self.method == 'wavelet':
            # Wavelet denoising
            return denoise_wavelet(image, method='BayesShrink', mode='soft', rescale_sigma=True, channel_axis=-1)
        
        elif self.method == 'median':
            # Median filtering
            # Convert to 0-255 range for OpenCV
            img_255 = (image * 255).astype(np.uint8)
            denoised = cv2.medianBlur(img_255, 5)
            # Convert back to 0-1 range
            return denoised / 255.0
        
        else:
            raise ValueError(f"Unsupported denoising method: {self.method}")
    
    def denoise_batch(self, images):
        """
        Apply denoising to a batch of images.
        
        Args:
            images: Batch of images to denoise
            
        Returns:
            Denoised images
        """
        return np.array([self.denoise(img) for img in images])