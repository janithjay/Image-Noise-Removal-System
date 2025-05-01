"""
Resolution adapter for image denoising.
Allows processing images at different resolutions than what the model was trained on.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image
import math

class ResolutionAdapter:
    """
    Adapter for processing images at different resolutions.
    Handles dividing large images into patches, processing them, and reassembling.
    """
    
    def __init__(self, model, target_size=None, overlap=0.25):
        """
        Initialize the resolution adapter.
        
        Args:
            model: The denoising model
            target_size: Target size for output images (height, width), or None to preserve input size
            overlap: Overlap ratio between patches (0.0-0.5, recommended 0.25)
        """
        self.model = model
        self.target_size = target_size
        self.overlap = overlap
        
        # Get the model's expected input shape
        self.model_input_shape = tuple(model.model.input.shape[1:3])
    
    def _split_into_patches(self, image, patch_size):
        """
        Split an image into overlapping patches.
        
        Args:
            image: Input image as numpy array (H,W,C)
            patch_size: Size of each patch (height, width)
            
        Returns:
            List of patches and their positions
        """
        h, w = image.shape[:2]
        patch_h, patch_w = patch_size
        
        # Calculate stride (accounting for overlap)
        stride_h = int(patch_h * (1 - self.overlap))
        stride_w = int(patch_w * (1 - self.overlap))
        
        # Calculate number of patches in each dimension
        n_h = math.ceil((h - patch_h) / stride_h) + 1
        n_w = math.ceil((w - patch_w) / stride_w) + 1
        
        # Adjust last stride to ensure we cover the entire image
        last_stride_h = (h - patch_h) if (h - patch_h) > 0 else 0
        last_stride_w = (w - patch_w) if (w - patch_w) > 0 else 0
        
        if n_h > 1:
            last_stride_h = last_stride_h / (n_h - 1)
        if n_w > 1:
            last_stride_w = last_stride_w / (n_w - 1)
        
        patches = []
        positions = []
        
        for i in range(n_h):
            for j in range(n_w):
                # Calculate position
                y = min(int(i * last_stride_h), h - patch_h)
                x = min(int(j * last_stride_w), w - patch_w)
                
                # Extract patch
                patch = image[y:y+patch_h, x:x+patch_w]
                
                # Only include full-sized patches
                if patch.shape[:2] == (patch_h, patch_w):
                    patches.append(patch)
                    positions.append((y, x))
        
        return patches, positions
    
    def _merge_patches(self, patches, positions, output_shape):
        """
        Merge processed patches back into a single image.
        
        Args:
            patches: List of processed patches
            positions: List of patch positions (y, x)
            output_shape: Shape of the output image (height, width, channels)
            
        Returns:
            Merged image
        """
        h, w, c = output_shape
        patch_h, patch_w = patches[0].shape[:2]
        
        # Create weight mask for blending patches
        y_ramp = np.ones((patch_h, 1))
        if self.overlap > 0:
            fade_len = int(patch_h * self.overlap)
            y_ramp[:fade_len, 0] = np.linspace(0, 1, fade_len)
            y_ramp[-fade_len:, 0] = np.linspace(1, 0, fade_len)
        
        x_ramp = np.ones((1, patch_w))
        if self.overlap > 0:
            fade_len = int(patch_w * self.overlap)
            x_ramp[0, :fade_len] = np.linspace(0, 1, fade_len)
            x_ramp[0, -fade_len:] = np.linspace(1, 0, fade_len)
        
        # Create 2D weight mask by multiplying ramps
        weight_mask = y_ramp @ x_ramp
        
        # Initialize output and weight accumulator
        result = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros((h, w, 1), dtype=np.float32)
        
        # Add weighted patches
        for patch, (y, x) in zip(patches, positions):
            # Expand weight mask to match patch channels
            patch_weight = np.repeat(weight_mask[:, :, np.newaxis], patch.shape[2], axis=2)
            
            # Apply weight to patch
            weighted_patch = patch * patch_weight
            
            # Add to result and accumulate weights
            result[y:y+patch_h, x:x+patch_w] += weighted_patch
            weights[y:y+patch_h, x:x+patch_w] += patch_weight
        
        # Normalize by accumulated weights
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        result = result / (weights + epsilon)
        
        return result
    
    def process_image(self, image):
        """
        Process an image at any resolution.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Denoised image as PIL Image
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            # Resize if target size is specified
            if self.target_size is not None:
                image = image.resize(self.target_size[::-1])  # PIL uses (width, height)
            
            # Convert to numpy array
            image_array = img_to_array(image) / 255.0
        else:
            # Already numpy array
            image_array = image.copy()
            
            # Resize if target size is specified
            if self.target_size is not None:
                image_array = tf.image.resize(
                    image_array, self.target_size).numpy()
        
        # Save original shape
        original_shape = image_array.shape
        
        # If the image is smaller than model input, handle directly
        if original_shape[0] <= self.model_input_shape[0] and original_shape[1] <= self.model_input_shape[1]:
            # Pad if necessary
            pad_h = max(0, self.model_input_shape[0] - original_shape[0])
            pad_w = max(0, self.model_input_shape[1] - original_shape[1])
            
            if pad_h > 0 or pad_w > 0:
                padding = [(0, pad_h), (0, pad_w), (0, 0)]
                padded = np.pad(image_array, padding, mode='reflect')
                # Process padded image
                batch = np.expand_dims(padded, axis=0)
                denoised_batch = self.model.model.predict(batch)
                # Unpad result
                denoised_array = denoised_batch[0, :original_shape[0], :original_shape[1]]
            else:
                # Process directly
                batch = np.expand_dims(image_array, axis=0)
                denoised_batch = self.model.model.predict(batch)
                denoised_array = denoised_batch[0]
        else:
            # Split into patches
            patches, positions = self._split_into_patches(image_array, self.model_input_shape)
            
            # Process each patch
            processed_patches = []
            for patch in patches:
                # Ensure patch shape matches model input
                if patch.shape[:2] != self.model_input_shape:
                    patch = tf.image.resize(patch, self.model_input_shape).numpy()
                
                # Process patch
                batch = np.expand_dims(patch, axis=0)
                denoised_batch = self.model.model.predict(batch)
                processed_patches.append(denoised_batch[0])
            
            # Merge patches
            denoised_array = self._merge_patches(processed_patches, positions, original_shape)
        
        # Clip to valid range
        denoised_array = np.clip(denoised_array, 0, 1)
        
        # Convert back to PIL Image
        return array_to_img(denoised_array)
    
    def process_batch(self, images):
        """
        Process a batch of images.
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            List of denoised images as PIL Images
        """
        return [self.process_image(img) for img in images] 