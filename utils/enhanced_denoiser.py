"""
Enhanced denoising methods that don't require training or use pre-trained models.
"""

import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise
import tensorflow as tf
from PIL import Image
import io

def bilateral_filter(img, strength=8, detail_preservation=1.0, iterations=1):
    """
    Apply bilateral filter with fine-tuned parameters.
    
    Args:
        img: Input image
        strength: Overall denoising strength (1-10)
        detail_preservation: Higher values preserve more details (0.5-2.0)
        iterations: Number of filtering passes
    
    Returns:
        Denoised image
    """
    # Convert to proper format if needed
    if img.max() <= 1.0:
        img_255 = (img * 255).astype(np.uint8)
    else:
        img_255 = img.astype(np.uint8)
    
    # Calculate adaptive parameters
    base_d = max(5, min(9, int(5 + strength * 0.5)))  # 5-9 based on strength
    base_sigma_color = strength * 10 / detail_preservation  # Higher detail_preservation = lower sigma_color
    base_sigma_space = strength / 2  # More conservative spatial sigma
    
    result = img_255.copy()
    
    # Apply multiple iterations with progressively finer parameters
    for i in range(iterations):
        # Reduce filter strength with each iteration
        iteration_factor = 1.0 if iterations == 1 else (iterations - i) / iterations
        d = max(3, int(base_d * iteration_factor))
        sigma_color = base_sigma_color * iteration_factor
        sigma_space = base_sigma_space * iteration_factor
        
        try:
            # Apply bilateral filter
            result = cv2.bilateralFilter(
                result, d, sigma_color, sigma_space, borderType=cv2.BORDER_REFLECT
            )
        except Exception as e:
            print(f"Error in bilateral filter iteration {i+1}: {str(e)}")
            break
    
    # Return in the same range as input
    if img.max() <= 1.0:
        return result.astype(np.float32) / 255.0
    else:
        return result

def advanced_denoising(img_array, method='bilateral', strength=8):
    """
    Apply advanced denoising methods to an image.
    
    Args:
        img_array: Input image as numpy array
        method: Denoising method ('bilateral', 'nlm', 'tv', 'wavelet', 'combined')
        strength: Denoising strength (1-10)
        
    Returns:
        Denoised image as numpy array
    """
    # Ensure correct range for image
    if img_array.max() <= 1.0:
        img_255 = (img_array * 255).astype(np.uint8)
    else:
        img_255 = img_array.astype(np.uint8)
    
    # Handle alpha channel properly
    has_alpha = len(img_255.shape) == 3 and img_255.shape[2] == 4
    alpha_channel = None
    
    # Ensure image is in RGB format with 3 channels
    if len(img_255.shape) == 2 or img_255.shape[2] == 1:
        # Convert grayscale to RGB if needed
        img_255 = cv2.cvtColor(img_255, cv2.COLOR_GRAY2BGR)
    elif has_alpha:
        # Save alpha channel and convert RGBA to RGB
        alpha_channel = img_255[:, :, 3]
        img_255 = cv2.cvtColor(img_255, cv2.COLOR_RGBA2BGR)
    
    # Apply denoising based on method
    try:
        if method == 'bilateral':
            # Use enhanced bilateral filter with adaptive parameters
            # For normal use: 1 iteration
            # For fine detail preservation: lower strength, higher detail_preservation
            detail_level = 1.0  # Default balance
            iterations = 1      # Default single pass
            
            # For higher strength, consider multiple passes with gentler settings
            if strength > 8:
                iterations = 2
            
            result = bilateral_filter(
                img_255, 
                strength=strength,
                detail_preservation=detail_level,
                iterations=iterations
            )
        elif method == 'nlm':
            # Non-local means denoising
            h = strength * 5  # Filter strength
            search_window = 21  # Search window size
            block_size = 7  # Block size
            
            result = cv2.fastNlMeansDenoisingColored(
                img_255, None, h, h, block_size, search_window
            )
        elif method == 'tv':
            # Total Variation denoising
            from skimage.restoration import denoise_tv_chambolle
            
            try:
                # Normalize to 0-1 range for skimage
                weight = strength / 30  # Denoising weight
                result = denoise_tv_chambolle(img_array, weight=weight, channel_axis=2)
                result = (result * 255).astype(np.uint8)
            except Exception as e:
                print(f"Error in TV denoising: {str(e)}")
                # Fallback to Gaussian blur
                result = cv2.GaussianBlur(img_255, (5, 5), 1.5)
            
        elif method == 'wavelet':
            # Wavelet denoising
            from skimage.restoration import denoise_wavelet
            
            # Adjust sigma based on strength
            sigma = strength / 50
            
            try:
                # Try the current version approach
                result = denoise_wavelet(img_array, sigma=sigma, mode='soft', 
                                     channel_axis=2, convert2ycbcr=True,
                                     method='BayesShrink')
                result = (result * 255).astype(np.uint8)
            except (TypeError, Exception) as e:
                print(f"Error in wavelet denoising: {str(e)}")
                try:
                    # Fall back to simpler version if there are parameter issues
                    result = denoise_wavelet(img_array, sigma=sigma, mode='soft')
                    result = (result * 255).astype(np.uint8)
                except Exception:
                    # Fallback to Gaussian blur
                    result = cv2.GaussianBlur(img_255, (5, 5), 1.5)
            
        elif method == 'combined':
            # Apply multiple methods for better results
            try:
                # First apply bilateral filter
                bilateral = cv2.bilateralFilter(img_255, 9, strength * 10, strength)
                
                # Then apply NLM
                result = np.zeros_like(bilateral)
                h = strength * 2
                for i in range(3):
                    result[:,:,i] = cv2.fastNlMeansDenoising(bilateral[:,:,i], None, h, 5, 7)
                
                # Optional: Enhance contrast using CLAHE
                if strength > 5:
                    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    l = clahe.apply(l)
                    lab = cv2.merge((l,a,b))
                    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            except cv2.error as e:
                print(f"Error in combined denoising: {str(e)}")
                # Fallback to Gaussian blur
                result = cv2.GaussianBlur(img_255, (5, 5), 1.5)
        else:
            raise ValueError(f"Unsupported denoising method: {method}")
        
        # Restore alpha channel if it was present
        if has_alpha and alpha_channel is not None:
            # Convert denoised result back to RGBA
            if img_array.max() <= 1.0:
                # For float images (0-1)
                result_rgba = np.zeros((result.shape[0], result.shape[1], 4), dtype=np.float32)
                result_rgba[:, :, :3] = result
                result_rgba[:, :, 3] = alpha_channel.astype(np.float32) / 255.0
                return result_rgba
            else:
                # For uint8 images (0-255)
                result_rgba = np.zeros((result.shape[0], result.shape[1], 4), dtype=np.uint8)
                result_rgba[:, :, :3] = result
                result_rgba[:, :, 3] = alpha_channel
                return result_rgba
        
        # Return in the same range as input
        if img_array.max() <= 1.0:
            return result.astype(np.float32) / 255.0
        else:
            return result
    except Exception as e:
        print(f"Error in advanced_denoising: {str(e)}")
        # Return the input image as a last resort
        return img_array

def ensemble_denoising(img_array, methods=None, weights=None):
    """
    Apply multiple denoising methods and combine the results.
    
    Args:
        img_array: Image as numpy array (0-1 range)
        methods: List of denoising methods
        weights: List of weights for each method
        
    Returns:
        Denoised image as numpy array
    """
    try:
        # Ensure image is in a valid format
        if img_array is None or img_array.size == 0:
            raise ValueError("Empty or invalid image input")
            
        # Check for NaN or infinity values
        if np.isnan(img_array).any() or np.isinf(img_array).any():
            # Replace NaN/inf with zeros
            img_array = np.nan_to_num(img_array)
        
        # Convert RGBA to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # Create a white background
            if img_array.max() <= 1.0:
                background = np.ones(img_array.shape[:2] + (3,))
                alpha = img_array[:, :, 3:4]
                img_array = alpha * img_array[:, :, :3] + (1 - alpha) * background
            else:
                background = np.ones(img_array.shape[:2] + (3,)) * 255
                alpha = img_array[:, :, 3:4] / 255.0
                img_array = alpha * img_array[:, :, :3] + (1 - alpha) * background
                
        # Make sure image has 3 channels if it's grayscale
        if len(img_array.shape) == 2:
            if img_array.max() <= 1.0:
                img_array = np.stack([img_array, img_array, img_array], axis=2)
            else:
                img_array = np.stack([img_array, img_array, img_array], axis=2)
                
        # Define default methods that are more robust
        if methods is None:
            # Default to more reliable methods 
            methods = ['bilateral', 'nlm', 'combined']
        
        if weights is None:
            weights = [0.4, 0.3, 0.3]  # Default weights
        
        if len(methods) != len(weights):
            raise ValueError("Number of methods and weights must match")
        
        # Collect successful denoising results
        results = []
        successful_weights = []
        
        for i, method in enumerate(methods):
            try:
                denoised = advanced_denoising(img_array, method=method)
                # Check if denoising was successful
                if denoised is not None and not np.isnan(denoised).any() and not np.isinf(denoised).any():
                    results.append(denoised)
                    successful_weights.append(weights[i])
            except Exception as e:
                print(f"Error applying method {method}: {str(e)}")
                # Continue with other methods
                continue
        
        # If we have at least one successful result
        if results:
            # Normalize weights
            if sum(successful_weights) > 0:
                successful_weights = [w / sum(successful_weights) for w in successful_weights]
            else:
                successful_weights = [1.0 / len(results)] * len(results)
            
            # Combine results using weighted average
            result = np.zeros_like(results[0], dtype=np.float32)
            for i, denoised in enumerate(results):
                result += denoised * successful_weights[i]
            
            # Ensure result is in correct range
            if img_array.max() <= 1.0:
                return np.clip(result, 0, 1)
            else:
                return np.clip(result, 0, 255).astype(np.uint8)
        else:
            # If all methods failed, use Gaussian blur as fallback
            print("All ensemble methods failed, falling back to Gaussian blur")
            if img_array.max() <= 1.0:
                img_255 = (img_array * 255).astype(np.uint8)
            else:
                img_255 = img_array.astype(np.uint8)
                
            # Make sure image is contiguous in memory    
            img_255 = np.ascontiguousarray(img_255)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_255, (5, 5), 1.5)
            
            if img_array.max() <= 1.0:
                return blurred.astype(np.float32) / 255.0
            else:
                return blurred
                
    except Exception as e:
        print(f"Fatal error in ensemble denoising: {str(e)}")
        # Create a simple blurred version
        try:
            if img_array.max() <= 1.0:
                img_255 = (img_array * 255).astype(np.uint8)
            else:
                img_255 = img_array.astype(np.uint8)
                
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_255, (5, 5), 1.5)
            
            if img_array.max() <= 1.0:
                return blurred.astype(np.float32) / 255.0
            else:
                return blurred
        except:
            # Return the input image as a last resort
            return img_array

def detail_preserving_denoising(img_array, strength=7):
    """
    Denoise while preserving important image details.
    
    Args:
        img_array: Image as numpy array (0-1 range)
        strength: Denoising strength (1-10)
        
    Returns:
        Denoised image as numpy array
    """
    try:
        # Ensure image is in the right format
        if img_array.max() <= 1.0:
            img_255 = (img_array * 255).astype(np.uint8)
        else:
            img_255 = img_array.astype(np.uint8)
        
        # Ensure image is in RGB format with 3 channels
        if len(img_255.shape) == 2 or img_255.shape[2] == 1:
            # Convert grayscale to RGB if needed
            img_255 = cv2.cvtColor(img_255, cv2.COLOR_GRAY2BGR)
        elif img_255.shape[2] == 4:
            # Convert RGBA to RGB if needed
            img_255 = cv2.cvtColor(img_255, cv2.COLOR_RGBA2BGR)
        
        # Make sure image is contiguous in memory
        img_255 = np.ascontiguousarray(img_255)
        
        # Create a detail mask using edge detection
        try:
            gray = cv2.cvtColor(img_255, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to protect more area around details
            kernel = np.ones((3, 3), np.uint8)
            detail_mask = cv2.dilate(edges, kernel, iterations=2)
            
            # Apply strong denoising to non-detail areas
            denoised_strong = advanced_denoising(img_255, method='combined', strength=strength)
            
            # Apply gentle denoising to detail areas
            denoised_gentle = advanced_denoising(img_255, method='bilateral', strength=strength/2)
            
            # Combine results using the detail mask
            detail_mask_3d = np.stack([detail_mask, detail_mask, detail_mask], axis=2) / 255.0
            
            if img_array.max() <= 1.0:
                denoised_strong = denoised_strong.astype(np.float32)
                denoised_gentle = denoised_gentle.astype(np.float32)
                result = (detail_mask_3d * denoised_gentle) + ((1 - detail_mask_3d) * denoised_strong)
                return np.clip(result, 0, 1)
            else:
                result = (detail_mask_3d * denoised_gentle) + ((1 - detail_mask_3d) * denoised_strong)
                return np.clip(result, 0, 255).astype(np.uint8)
                
        except Exception as e:
            print(f"Error in detail mask creation: {str(e)}")
            # Fallback to simple adaptive denoising
            if strength > 7:
                # For high strength, use bilateral filter (preserves edges)
                return advanced_denoising(img_255, method='bilateral', strength=strength)
            else:
                # For low strength, use NLM (preserves details)
                return advanced_denoising(img_255, method='nlm', strength=strength)
                
    except Exception as e:
        print(f"Error in detail-preserving denoising: {str(e)}")
        # Ultimate fallback - just return Gaussian blur
        if img_array.max() <= 1.0:
            blurred = cv2.GaussianBlur(img_255, (5, 5), 1.5).astype(np.float32) / 255.0
            return np.clip(blurred, 0, 1)
        else:
            blurred = cv2.GaussianBlur(img_255, (5, 5), 1.5)
            return np.clip(blurred, 0, 255).astype(np.uint8)

def enhance_image_details(img_array, sharpness=1.5, saturation=1.2):
    """
    Enhance details and colors in the image after denoising.
    
    Args:
        img_array: Image as numpy array (0-1 range)
        sharpness: Sharpness enhancement factor
        saturation: Saturation enhancement factor
        
    Returns:
        Enhanced image as numpy array
    """
    try:
        # Check for valid input
        if img_array is None or img_array.size == 0:
            raise ValueError("Empty or invalid image input")
            
        # Check for NaN or infinity values
        if np.isnan(img_array).any() or np.isinf(img_array).any():
            img_array = np.nan_to_num(img_array)
        
        # Convert to proper format for PIL
        if img_array.max() <= 1.0:
            img_255 = (img_array * 255).astype(np.uint8)
        else:
            img_255 = img_array.astype(np.uint8)
        
        # Ensure image is in RGB format with 3 channels
        if len(img_255.shape) == 2:
            # Convert grayscale to RGB
            img_255 = np.stack([img_255, img_255, img_255], axis=2)
        elif img_255.shape[2] == 4:
            # Remove alpha channel for PIL processing
            img_255 = img_255[:, :, :3]
        
        # Create PIL image with error handling
        try:
            img_pil = Image.fromarray(img_255)
        except Exception as e:
            print(f"Error converting to PIL image: {str(e)}")
            # Return input if we can't process it
            return img_array
            
        # Enhance sharpness
        from PIL import ImageEnhance
        try:
            enhancer = ImageEnhance.Sharpness(img_pil)
            img_pil = enhancer.enhance(sharpness)
            
            # Enhance saturation
            enhancer = ImageEnhance.Color(img_pil)
            img_pil = enhancer.enhance(saturation)
            
            # Convert back to numpy array
            result = np.array(img_pil)
            
            # Return in the same range as input
            if img_array.max() <= 1.0:
                return result.astype(np.float32) / 255.0
            else:
                return result
        except Exception as e:
            print(f"Error during enhancement: {str(e)}")
            # Return original image if enhancement fails
            return img_array
            
    except Exception as e:
        print(f"Fatal error in image enhancement: {str(e)}")
        # Return original image in case of error
        return img_array 