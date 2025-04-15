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
    Perceptual loss based on VGG16 feature maps.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        
    Returns:
        Perceptual loss value
    """
    # Load VGG16 model
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    
    # Create model for feature extraction
    model = Model(inputs=vgg.input, outputs=[
        vgg.get_layer('block1_conv2').output,
        vgg.get_layer('block2_conv2').output,
        vgg.get_layer('block3_conv3').output,
    ])
    
    # Make model non-trainable
    for layer in model.layers:
        layer.trainable = False
    
    # Get features
    features_true = model(y_true)
    features_pred = model(y_pred)
    
    # Calculate MSE between features
    loss = 0
    for f_true, f_pred in zip(features_true, features_pred):
        loss += K.mean(K.square(f_true - f_pred))
    
    return loss

def combined_loss(y_true, y_pred, alpha=0.8, beta=0.1, gamma=0.1):
    """
    Combined loss function: MSE + SSIM + Perceptual
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        alpha: Weight for MSE loss
        beta: Weight for SSIM loss
        gamma: Weight for perceptual loss
        
    Returns:
        Combined loss value
    """
    mse_loss = K.mean(K.square(y_true - y_pred))
    ss_loss = ssim_loss(y_true, y_pred)
    percep_loss = perceptual_loss(y_true, y_pred)
    
    return alpha * mse_loss + beta * ss_loss + gamma * percep_loss

# Create custom wrapper for combined loss with fixed weights
def get_combined_loss(alpha=0.8, beta=0.1, gamma=0.1):
    """
    Create a combined loss function with fixed weights.
    
    Args:
        alpha: Weight for MSE loss
        beta: Weight for SSIM loss
        gamma: Weight for perceptual loss
        
    Returns:
        Combined loss function
    """
    def loss(y_true, y_pred):
        return combined_loss(y_true, y_pred, alpha, beta, gamma)
    
    return loss 