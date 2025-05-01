"""
Custom loss functions for image denoising.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) loss function.
    Computes the negative of SSIM (1-SSIM) to make it a minimizable loss.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        
    Returns:
        SSIM loss value
    """
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def perceptual_loss(y_true, y_pred):
    """
    Enhanced perceptual loss based on VGG16 feature maps.
    Uses more layers for better detail preservation.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        
    Returns:
        Perceptual loss value
    """
    # Load VGG16 model
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    
    # Create model for feature extraction with more layers
    model = Model(inputs=vgg.input, outputs=[
        vgg.get_layer('block1_conv2').output,
        vgg.get_layer('block2_conv2').output,
        vgg.get_layer('block3_conv3').output,
        vgg.get_layer('block4_conv3').output,  # Added deeper layer
        vgg.get_layer('block5_conv3').output,  # Added deeper layer
    ])
    
    # Make model non-trainable
    for layer in model.layers:
        layer.trainable = False
    
    # Get features
    features_true = model(y_true)
    features_pred = model(y_pred)
    
    # Calculate weighted MSE between features (give less weight to deeper layers)
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing weights for deeper layers
    loss = 0
    for i, (f_true, f_pred) in enumerate(zip(features_true, features_pred)):
        loss += weights[i] * K.mean(K.square(f_true - f_pred))
    
    return loss

def edge_preservation_loss(y_true, y_pred):
    """
    Edge preservation loss using gradient differences.
    Helps preserve sharp edges and details in the image.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        
    Returns:
        Edge preservation loss value
    """
    # Calculate gradients
    def _sobel_gradients(img):
        # Sobel filters for edge detection
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Apply filters to each channel
        grad_list = []
        for i in range(img.shape[-1]):
            img_channel = img[..., i:i+1]
            gx = tf.nn.conv2d(img_channel, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
            gy = tf.nn.conv2d(img_channel, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
            grad = tf.sqrt(tf.square(gx) + tf.square(gy))
            grad_list.append(grad)
        
        return tf.concat(grad_list, axis=-1)
    
    # Get gradients
    gradients_true = _sobel_gradients(y_true)
    gradients_pred = _sobel_gradients(y_pred)
    
    # L1 loss on gradients
    return K.mean(K.abs(gradients_true - gradients_pred))

def combined_loss(y_true, y_pred, alpha=0.5, beta=0.3, gamma=0.1, delta=0.1):
    """
    Enhanced combined loss function: MSE + SSIM + Perceptual + Edge Preservation
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        alpha: Weight for MSE loss
        beta: Weight for SSIM loss
        gamma: Weight for perceptual loss
        delta: Weight for edge preservation loss
        
    Returns:
        Combined loss value
    """
    mse_loss = K.mean(K.square(y_true - y_pred))
    ss_loss = ssim_loss(y_true, y_pred)
    percep_loss = perceptual_loss(y_true, y_pred)
    edge_loss = edge_preservation_loss(y_true, y_pred)
    
    return alpha * mse_loss + beta * ss_loss + gamma * percep_loss + delta * edge_loss

# Create custom wrapper for combined loss with fixed weights
def get_combined_loss(alpha=0.5, beta=0.3, gamma=0.1, delta=0.1):
    """
    Create an enhanced combined loss function with fixed weights.
    
    Args:
        alpha: Weight for MSE loss
        beta: Weight for SSIM loss
        gamma: Weight for perceptual loss
        delta: Weight for edge preservation loss
        
    Returns:
        Combined loss function
    """
    def loss(y_true, y_pred):
        return combined_loss(y_true, y_pred, alpha, beta, gamma, delta)
    
    return loss 