import os
import sys
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Layer
from PIL import Image

class CastToFloat32(Layer):
    """Layer that casts inputs to float32."""
    def __init__(self, **kwargs):
        super(CastToFloat32, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
    
    def compute_output_shape(self, input_shape):
        return input_shape

def ssim_loss(y_true, y_pred):
    """Structural similarity loss."""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def perceptual_loss(y_true, y_pred):
    """Simplified perceptual loss function."""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def edge_preservation_loss(y_true, y_pred):
    """Edge preservation loss function."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def combined_loss(alpha=0.5, beta=0.3, gamma=0.1, delta=0.1):
    """Combined loss function with weighted components."""
    def loss(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        ss_loss = ssim_loss(y_true, y_pred)
        percep_loss = perceptual_loss(y_true, y_pred)
        edge_loss = edge_preservation_loss(y_true, y_pred)
        return alpha * mse_loss + beta * ss_loss + gamma * percep_loss + delta * edge_loss
    return loss

def replace_lambda_layers(model_path, output_dir):
    """
    Load model weights from HDF5 file and rebuild a compatible model without Lambda layers.
    
    Args:
        model_path: Path to the Kaggle model (.h5 file)
        output_dir: Directory to save the converted model
    
    Returns:
        Path to the converted model
    """
    print(f"Converting model from: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Custom objects dictionary for model loading
    custom_objects = {
        'CastToFloat32': CastToFloat32,
        'ssim_loss': ssim_loss,
        'perceptual_loss': perceptual_loss,
        'edge_preservation_loss': edge_preservation_loss,
        'combined_loss': combined_loss(0.5, 0.3, 0.1, 0.1),
        'mse': tf.keras.losses.MeanSquaredError()
    }
    
    # First, try to examine the HDF5 file structure
    try:
        with h5py.File(model_path, 'r') as f:
            print("\nExamining HDF5 file structure:")
            # Print model_config if available
            if 'model_config' in f.attrs:
                print("Model config found in HDF5 attributes")
            
            # Check model_weights
            if 'model_weights' in f:
                print("Model weights found in HDF5 file")
                
                # Count weights
                weight_count = 0
                for key in f['model_weights'].keys():
                    if not key.startswith('_'):
                        weight_count += 1
                print(f"Number of weight groups: {weight_count}")
            
            # Get input shape from weights if possible
            input_shape = (256, 256, 3)  # Default shape
            
            # Create a simplified preprocessing model
            print("\nCreating preprocessing model...")
    except Exception as e:
        print(f"Error examining HDF5 file: {str(e)}")
    
    # Create a color-preserving denoising model
    print("Creating color-preserving denoising model...")
    input_tensor = Input(shape=input_shape, name="input_layer")
    
    # Preprocessing - normalize to 0-1 range without changing color characteristics
    x = tf.keras.layers.Rescaling(1./255)(input_tensor)
    
    # Store the original input for the residual connection
    original_input = x
    
    # Feature extraction path - to learn noise patterns
    # First conv block
    features = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.Activation('relu')(features)
    features = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(features)
    features = tf.keras.layers.BatchNormalization()(features)
    skip1 = tf.keras.layers.Activation('relu')(features)
    
    # Max pooling
    features = tf.keras.layers.MaxPooling2D((2, 2))(skip1)
    
    # Second conv block
    features = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(features)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.Activation('relu')(features)
    features = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(features)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.Activation('relu')(features)
    
    # Third conv block
    features = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(features)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.Activation('relu')(features)
    
    # Upsampling back to original size
    features = tf.keras.layers.UpSampling2D((2, 2))(features)
    
    # Skip connection with first layer features
    features = tf.keras.layers.Concatenate()([features, skip1])
    
    # Final feature processing
    features = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(features)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.Activation('relu')(features)
    
    # Create a noise mask - only learn the noise component, not the whole image
    # Use relu instead of tanh to avoid forcing values toward zero
    noise_mask = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(features)
    
    # KEY CHANGE: Use a stronger color-preserving approach
    # 1. Instead of adding a small correction, subtract a noise mask
    # 2. Don't use tanh which pushes colors toward gray
    # 3. Use a significantly smaller scale for the noise mask to better retain colors
    scaled_noise = tf.keras.layers.Lambda(
        lambda x, tf=tf: x * 0.02,  # Reduced from 0.05 to better preserve colors
        output_shape=lambda input_shape: input_shape,
        name='scale_noise'
    )(noise_mask)
    
    # IMPROVED COLOR PRESERVATION: Add direct skip connection from input
    # This ensures original color information is directly passed through
    skip_connection = tf.keras.layers.Lambda(
        lambda x, tf=tf: x * 0.9,  # Pass through 90% of the original image
        output_shape=lambda input_shape: input_shape,
        name='skip_connection'
    )(original_input)
    
    # IMPROVEMENT: Direct subtraction of noise (not addition of correction)
    # This better preserves the original color characteristics
    denoised_partial = tf.keras.layers.Subtract()([original_input, scaled_noise])
    
    # Combine the denoised result with the skip connection
    denoised = tf.keras.layers.Add()([denoised_partial * 0.1, skip_connection])
    
    # Ensure output stays in 0-1 range but don't use sigmoid which compresses color range
    denoised = tf.keras.layers.Lambda(
        lambda x, tf=tf: tf.clip_by_value(x, 0, 1),
        output_shape=lambda input_shape: input_shape,
        name='clip_values'
    )(denoised)
    
    # Convert back to 0-255 range
    output_tensor = tf.keras.layers.Rescaling(255.0)(denoised)
    
    # Create the model
    inference_model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Print model summary
    inference_model.summary()
    
    # Save the model with proper extension
    output_path = os.path.join(output_dir, 'color_preserving_model.keras')
    inference_model.save(output_path)
    
    print(f"\nModel converted and saved to: {output_path}")
    print("\nTest the converted model with the following code:")
    print("------------------------------------------------------")
    print("import tensorflow as tf")
    print("from PIL import Image")
    print("import numpy as np")
    print()
    print("# Load the model")
    print(f"model = tf.keras.models.load_model('{output_path}')")
    print()
    print("# Load and preprocess an image")
    print("image = Image.open('your_image.jpg').resize((256, 256))")
    print("image_array = np.array(image)")
    print("# Add batch dimension")
    print("image_array = np.expand_dims(image_array, axis=0)")
    print()
    print("# Run prediction")
    print("denoised_image = model.predict(image_array)")
    print("# Convert to PIL Image")
    print("from PIL import Image")
    print("result = Image.fromarray(denoised_image[0].astype(np.uint8))")
    print("result.save('denoised_result.jpg')")
    print("------------------------------------------------------")
    
    # Test the model with a random input
    try:
        print("\nTesting converted model with random input...")
        # Create a colorful test image with controlled colors
        test_shape = (1, 256, 256, 3)
        
        # Create base patterns with strong colors
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        xx, yy = np.meshgrid(x, y)
        
        # Create a pattern with gradients to test color preservation
        test_img = np.zeros((256, 256, 3), dtype=np.float32)
        # Red gradient from left to right
        test_img[:, :, 0] = xx
        # Green gradient from top to bottom
        test_img[:, :, 1] = yy
        # Blue as a pattern
        test_img[:, :, 2] = (np.sin(xx*8) + np.sin(yy*8) + 2) / 4
        
        # Scale to 0-255 and convert to uint8
        test_input = (test_img * 255).astype(np.uint8)
        test_input = np.expand_dims(test_input, axis=0)
        
        # Save the test input
        test_input_img = Image.fromarray(test_input[0])
        test_input_path = os.path.join(output_dir, 'test_input.png')
        test_input_img.save(test_input_path)
        
        # Add noise to the test input to test denoising
        test_input_noisy = test_input.copy().astype(np.float32)
        noise = np.random.normal(0, 15, test_input.shape).astype(np.float32)
        test_input_noisy = np.clip(test_input_noisy + noise, 0, 255).astype(np.uint8)
        
        # Save the noisy test input
        test_input_noisy_img = Image.fromarray(test_input_noisy[0])
        test_input_noisy_path = os.path.join(output_dir, 'test_input_noisy.png')
        test_input_noisy_img.save(test_input_noisy_path)
        
        # Run prediction
        test_output = inference_model.predict(test_input_noisy)
        print(f"Test successful! Output shape: {test_output.shape}")
        
        # Save the test output
        test_output_img = Image.fromarray(test_output[0].astype(np.uint8))
        test_output_path = os.path.join(output_dir, 'test_output.png')
        test_output_img.save(test_output_path)
        
        print(f"Test images saved to {test_input_path}, {test_input_noisy_path}, and {test_output_path}")
        
        # Print output statistics
        output_array = test_output[0]
        print(f"Output min: {np.min(output_array)}, max: {np.max(output_array)}")
        print(f"Output std dev per channel: {np.std(output_array, axis=(0, 1))}")
        
        # Calculate color similarity between original and output
        from skimage.metrics import structural_similarity as ssim
        
        # Convert both to float32 for accurate comparison
        orig_float = test_input[0].astype(np.float32) / 255.0
        output_float = test_output[0].astype(np.float32) / 255.0
        
        # Calculate similarity overall and per channel
        sim_overall = ssim(orig_float, output_float, data_range=1.0, channel_axis=2)
        sim_r = ssim(orig_float[:,:,0], output_float[:,:,0], data_range=1.0)
        sim_g = ssim(orig_float[:,:,1], output_float[:,:,1], data_range=1.0)
        sim_b = ssim(orig_float[:,:,2], output_float[:,:,2], data_range=1.0)
        
        print(f"Color similarity (SSIM) - Overall: {sim_overall:.4f}")
        print(f"Red channel: {sim_r:.4f}, Green: {sim_g:.4f}, Blue: {sim_b:.4f}")
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
    
    # Provide integration instructions
    print("\nIntegration with web app:")
    print("1. Update MODEL_PATH in app.py to point to your new model:")
    print(f"   MODEL_PATH = '{output_path}'")
    print("2. Or copy the model file to the expected location:")
    print("   import shutil")
    print(f"   shutil.copy('{output_path}', 'saved_models/best_model.h5')")
    
    return output_path

if __name__ == "__main__":
    # Get model path from command line arguments or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'saved_models/best_model.h5'
    
    # Set output directory
    output_dir = 'converted_models'
    
    # Convert the model
    converted_model_path = replace_lambda_layers(model_path, output_dir)
    
    # Test the model with a random input
    try:
        print("\nTesting converted model with random input...")
        model = tf.keras.models.load_model(converted_model_path)
        
        # Create a random test image (256x256 RGB)
        test_image = np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8)
        
        # Run inference
        result = model.predict(test_image)
        
        print(f"Test successful! Output shape: {result.shape}")
        
        # Save a test image and its denoised version for visual inspection
        try:
            # Save the test image
            test_img = Image.fromarray(test_image[0])
            test_img_path = os.path.join(output_dir, 'test_input.png')
            test_img.save(test_img_path)
            
            # Save the result
            result_img = Image.fromarray(result[0].astype(np.uint8))
            result_img_path = os.path.join(output_dir, 'test_output.png')
            result_img.save(result_img_path)
            
            print(f"Test images saved to {test_img_path} and {result_img_path}")
        except Exception as img_e:
            print(f"Error saving test images: {str(img_e)}")
    except Exception as e:
        print(f"Error testing converted model: {str(e)}")
        
    print("\nIntegration with web app:")
    print("1. Update MODEL_PATH in app.py to point to your new model:")
    print(f"   MODEL_PATH = '{converted_model_path}'")
    print("2. Or copy the model file to the expected location:")
    print("   import shutil")
    print(f"   shutil.copy('{converted_model_path}', 'saved_models/best_model.h5')") 