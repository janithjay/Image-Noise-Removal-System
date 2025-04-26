"""
Visualization utilities for displaying results.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import tensorflow as tf
import cv2

def combine_images(original_path, noisy_path, denoised_path, save_path=None, metrics=None):
    """
    Combine original, noisy, and denoised images side by side with relevant information.
    
    Args:
        original_path: Path to the original image
        noisy_path: Path to the noisy image
        denoised_path: Path to the denoised image
        save_path: Path to save the combined image (if None, will not save)
        metrics: Dictionary of metrics to display (e.g., {'PSNR': 32.5, 'SSIM': 0.92})
        
    Returns:
        Combined image as PIL Image object
    """
    # Load images preserving alpha channel if present
    original = Image.open(original_path)
    noisy = Image.open(noisy_path)
    denoised = Image.open(denoised_path)
    
    # Check if any image has an alpha channel
    has_alpha = any(img.mode == 'RGBA' for img in [original, noisy, denoised])
    target_mode = 'RGBA' if has_alpha else 'RGB'
    
    # Convert all images to the same mode for consistency
    original = original.convert(target_mode)
    noisy = noisy.convert(target_mode)
    denoised = denoised.convert(target_mode)
    
    # Resize images to the same size if needed
    target_height = 256
    aspect_ratio = original.width / original.height
    target_width = int(target_height * aspect_ratio)
    
    original = original.resize((target_width, target_height))
    noisy = noisy.resize((target_width, target_height))
    denoised = denoised.resize((target_width, target_height))
    
    # Create a new image with space for titles and metrics
    padding = 20
    text_height = 30
    header_height = text_height + padding
    metrics_height = 50 if metrics else 0
    
    total_width = (target_width + padding) * 3
    total_height = target_height + header_height + metrics_height + padding
    
    # Create background with appropriate mode
    background_color = (255, 255, 255, 255) if has_alpha else (255, 255, 255)
    combined = Image.new(target_mode, (total_width, total_height), background_color)
    
    # Paste images into the combined image
    combined.paste(original, (padding, header_height))
    combined.paste(noisy, (target_width + padding * 2, header_height))
    combined.paste(denoised, (target_width * 2 + padding * 3, header_height))
    
    # Add text
    draw = ImageDraw.Draw(combined)
    try:
        # Try to load a font, fall back to default if not available
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Add titles
    draw.text((padding + target_width // 2 - 30, padding), "Original Image", fill=(0, 0, 0), font=font)
    draw.text((target_width + padding * 2 + target_width // 2 - 30, padding), "Noisy Image", fill=(0, 0, 0), font=font)
    draw.text((target_width * 2 + padding * 3 + target_width // 2 - 30, padding), "Denoised Image", fill=(0, 0, 0), font=font)
    
    # Add metrics if provided
    if metrics:
        metrics_y = header_height + target_height + padding // 2
        metrics_text = " | ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        draw.text((total_width // 2 - 100, metrics_y), metrics_text, fill=(0, 0, 0), font=font)
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        combined.save(save_path)
    
    return combined

def calculate_metrics(original, denoised):
    """
    Calculate comprehensive image quality metrics between original and denoised images.
    
    Args:
        original: Path to the original image or numpy array
        denoised: Path to the denoised image or numpy array
        
    Returns:
        Dictionary with metrics (PSNR, SSIM, MSE, MAE, LPIPS)
    """
    try:
        # Handle input that could be either file paths or numpy arrays
        if isinstance(original, str):
            try:
                original = np.array(Image.open(original).convert('RGB'))
            except Exception as e:
                print(f"Error loading original image: {str(e)}")
                return {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
                
        if isinstance(denoised, str):
            try:
                denoised = np.array(Image.open(denoised).convert('RGB'))
            except Exception as e:
                print(f"Error loading denoised image: {str(e)}")
                return {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
                
        # Check for valid image arrays
        if original is None or denoised is None:
            print("One or both images are None")
            return {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
            
        if original.size == 0 or denoised.size == 0:
            print("One or both images are empty")
            return {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
            
        # Check for shape mismatch
        if original.shape != denoised.shape:
            print(f"Shape mismatch: original {original.shape}, denoised {denoised.shape}")
            # Attempt to resize if dimensions don't match
            try:
                from skimage.transform import resize
                denoised = resize(denoised, original.shape)
            except Exception as e:
                print(f"Resize error: {str(e)}")
                return {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}
    
        # Ensure images are in the correct range
        if original.max() > 1.0:
            original = original / 255.0
        if denoised.max() > 1.0:
            denoised = denoised / 255.0
        
        # Check for NaN or Inf values
        if np.isnan(original).any() or np.isinf(original).any() or np.isnan(denoised).any() or np.isinf(denoised).any():
            print("NaN or Inf values detected in images")
            original = np.nan_to_num(original)
            denoised = np.nan_to_num(denoised)
        
        # Calculate PSNR
        try:
            psnr = peak_signal_noise_ratio(original, denoised, data_range=1.0)
        except Exception as e:
            print(f"PSNR calculation error: {str(e)}")
            psnr = 0
        
        # Calculate SSIM
        try:
            ssim = structural_similarity(
                original, denoised, 
                data_range=1.0, 
                channel_axis=2 if original.ndim == 3 else None
            )
        except Exception as e:
            print(f"SSIM calculation error: {str(e)}")
            ssim = 0
            
        # Calculate MSE
        try:
            mse = mean_squared_error(original.flatten(), denoised.flatten())
        except Exception as e:
            print(f"MSE calculation error: {str(e)}")
            mse = 0
            
        # Calculate MAE
        try:
            mae = np.mean(np.abs(original - denoised))
        except Exception as e:
            print(f"MAE calculation error: {str(e)}")
            mae = 0
            
        # Calculate NIQE (No-Reference metric) if cv2 is available
        niqe = 0
        try:
            if cv2 is not None:
                denoised_uint8 = (denoised * 255).astype(np.uint8)
                if len(denoised_uint8.shape) == 3 and denoised_uint8.shape[2] == 3:
                    denoised_gray = cv2.cvtColor(denoised_uint8, cv2.COLOR_RGB2GRAY)
                    # Note: This is a placeholder as cv2 doesn't have NIQE built-in
                    # For real NIQE calculation, you'd need to implement the algorithm or use a library
                    niqe = 0  # Placeholder
        except Exception as e:
            print(f"NIQE calculation error: {str(e)}")
            
        return {
            'PSNR': psnr,
            'SSIM': ssim,
            'MSE': mse,
            'MAE': mae
        }
        
    except Exception as e:
        print(f"Unexpected error in metrics calculation: {str(e)}")
        return {'PSNR': 0, 'SSIM': 0, 'MSE': 0, 'MAE': 0}

def generate_comparison_grid(original_images, noisy_images, denoised_images, save_path=None, grid_size=(3, 3)):
    """
    Generate a grid of comparison images showing original, noisy, and denoised versions.
    
    Args:
        original_images: List of original image arrays
        noisy_images: List of noisy image arrays
        denoised_images: List of denoised image arrays
        save_path: Path to save the grid image
        grid_size: Tuple of (rows, cols) for the grid layout
        
    Returns:
        Grid image as numpy array
    """
    rows, cols = grid_size
    num_images = min(len(original_images), rows * cols)
    
    if num_images == 0:
        print("No images to display")
        return None
    
    # Calculate grid dimensions
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 6, rows * 3))
    
    # Flatten axes for easier indexing if necessary
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Fill the grid with images
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_images:
                # Original image
                ax_orig = axes[i, j*3]
                ax_orig.imshow(np.clip(original_images[idx], 0, 1))
                ax_orig.set_title('Original')
                ax_orig.axis('off')
                
                # Noisy image
                ax_noisy = axes[i, j*3 + 1]
                ax_noisy.imshow(np.clip(noisy_images[idx], 0, 1))
                ax_noisy.set_title('Noisy')
                ax_noisy.axis('off')
                
                # Denoised image
                ax_denoised = axes[i, j*3 + 2]
                ax_denoised.imshow(np.clip(denoised_images[idx], 0, 1))
                metrics = calculate_metrics(original_images[idx], denoised_images[idx])
                ax_denoised.set_title(f'Denoised\nPSNR: {metrics["PSNR"]:.2f}')
                ax_denoised.axis('off')
            else:
                # Hide unused subplots
                for k in range(3):
                    axes[i, j*3 + k].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison grid saved to {save_path}")
    
    return fig

def visualize_filters(model, layer_name, save_path=None):
    """
    Visualize filters from a specific convolutional layer of the model.
    
    Args:
        model: Keras model
        layer_name: Name of the convolutional layer to visualize
        save_path: Path to save the filter visualization
        
    Returns:
        Figure with filter visualizations
    """
    try:
        # Get the specified layer
        layer = None
        for l in model.layers:
            if l.name == layer_name:
                layer = l
                break
        
        if layer is None:
            print(f"Layer '{layer_name}' not found")
            return None
        
        # Get the weights
        weights = layer.get_weights()[0]
        
        # For convolutional layer weights: [kernel_height, kernel_width, input_channels, output_channels]
        if len(weights.shape) != 4:
            print(f"Layer {layer_name} doesn't have 4D weights (not a standard conv layer)")
            return None
        
        # Normalize weights for visualization
        weights = weights - weights.min()
        weights = weights / weights.max() if weights.max() > 0 else weights
        
        # Number of filters to display
        n_filters = min(64, weights.shape[3])
        n_channels = min(3, weights.shape[2])
        
        # Create a grid of filter visualizations
        rows = int(np.sqrt(n_filters))
        cols = int(np.ceil(n_filters / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < n_filters:
                    # For RGB input, show colored filters
                    if n_channels == 3:
                        filter_img = weights[:, :, :, idx]
                    else:
                        # For grayscale, just use the first channel
                        filter_img = weights[:, :, 0, idx]
                        
                    # Plot the filter
                    if rows == 1 and cols == 1:
                        ax = axes
                    elif rows == 1:
                        ax = axes[j]
                    elif cols == 1:
                        ax = axes[i]
                    else:
                        ax = axes[i, j]
                        
                    ax.imshow(filter_img)
                    ax.axis('off')
                else:
                    if rows == 1 and cols == 1:
                        axes.axis('off')
                    elif rows == 1:
                        axes[j].axis('off')
                    elif cols == 1:
                        axes[i].axis('off')
                    else:
                        axes[i, j].axis('off')
        
        plt.suptitle(f'Filters from layer: {layer_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Filter visualization saved to {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error visualizing filters: {str(e)}")
        return None

def visualize_feature_maps(model, image, layer_name, save_path=None, max_features=16):
    """
    Visualize feature maps from a specific layer when processing an image.
    
    Args:
        model: Keras model
        image: Input image as numpy array (should be in the correct shape for the model)
        layer_name: Name of the layer to visualize feature maps from
        save_path: Path to save the feature map visualization
        max_features: Maximum number of feature maps to display
        
    Returns:
        Figure with feature map visualizations
    """
    try:
        # Create a model that outputs the feature maps from the specified layer
        layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get feature maps
        feature_maps = layer_model.predict(image)
        
        # Get number of feature maps
        n_features = min(max_features, feature_maps.shape[-1])
        
        # Create a grid to display feature maps
        rows = int(np.sqrt(n_features))
        cols = int(np.ceil(n_features / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < n_features:
                    # Get feature map
                    feature = feature_maps[0, :, :, idx]
                    
                    # Normalize for visualization
                    feature = feature - feature.min()
                    feature = feature / feature.max() if feature.max() > 0 else feature
                    
                    # Plot the feature map
                    if rows == 1 and cols == 1:
                        ax = axes
                    elif rows == 1:
                        ax = axes[j]
                    elif cols == 1:
                        ax = axes[i]
                    else:
                        ax = axes[i, j]
                        
                    ax.imshow(feature, cmap='viridis')
                    ax.set_title(f'Feature {idx}')
                    ax.axis('off')
                else:
                    if rows == 1 and cols == 1:
                        axes.axis('off')
                    elif rows == 1:
                        axes[j].axis('off')
                    elif cols == 1:
                        axes[i].axis('off')
                    else:
                        axes[i, j].axis('off')
        
        plt.suptitle(f'Feature Maps from layer: {layer_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Feature map visualization saved to {save_path}")
        
        return fig
        
    except Exception as e:
        print(f"Error visualizing feature maps: {str(e)}")
        return None 