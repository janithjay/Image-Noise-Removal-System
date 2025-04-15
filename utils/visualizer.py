"""
Visualization utilities for displaying results.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

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
    # Load images
    original = Image.open(original_path)
    noisy = Image.open(noisy_path)
    denoised = Image.open(denoised_path)
    
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
    
    combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
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

def calculate_metrics(original_img, denoised_img):
    """
    Calculate image quality metrics between original and denoised images.
    
    Args:
        original_img: Original image as numpy array (0-1 range)
        denoised_img: Denoised image as numpy array (0-1 range)
        
    Returns:
        Dictionary with metrics (PSNR, SSIM)
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    # Ensure images are in the correct range
    if original_img.max() > 1.0:
        original_img = original_img / 255.0
    if denoised_img.max() > 1.0:
        denoised_img = denoised_img / 255.0
    
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(original_img, denoised_img, data_range=1.0)
    
    # Calculate SSIM
    ssim = structural_similarity(
        original_img, denoised_img, 
        data_range=1.0, 
        multichannel=True if original_img.ndim == 3 else False,
        channel_axis=2 if original_img.ndim == 3 else None
    )
    
    return {
        'PSNR': psnr,
        'SSIM': ssim
    } 