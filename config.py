"""
Configuration file for noise reduction system.
Contains parameters for model, training, and preprocessing.
"""

import os

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (128, 128, 3),  # Changed back to match existing model
    'latent_dim': 256,             # Increased for better feature representation
    'filters': [64, 128, 256, 512], # More and wider filters for better feature extraction
    'kernel_size': 3,              # Size of convolutional kernels
    'activation': 'relu',          # Activation function
    'final_activation': 'sigmoid', # Final layer activation
    'use_skip_connections': True,  # Enable skip connections (U-Net style)
    'use_residual_blocks': True,   # Enabled for better gradient flow and feature preservation
    'use_attention': True,         # Added attention mechanism to focus on important features
    'use_residual_learning': True, # Added residual learning (network learns the noise pattern)
    'dropout_rate': 0.1,           # Reduced dropout for less information loss
}

# Training Configuration
TRAINING_CONFIG = {
    'BATCH_SIZE': 4,               # Kept small to avoid memory issues
    'EPOCHS': 100,                 # Increased for better convergence
    'LEARNING_RATE': 0.0001,       # Further reduced for fine-tuning
    'EARLY_STOPPING_PATIENCE': 15, # Increased to avoid early termination
    'INPUT_SHAPE': (128, 128, 3),  # Changed back to match existing model
    'NOISE_TYPES': ['gaussian', 'poisson', 'salt_pepper', 'speckle'], # Added speckle for more robustness
    'NOISE_PARAMS': {
        'gaussian': {'mean': 0, 'sigma': 0.08},  # Reduced noise level slightly
        'poisson': {'lambda_val': 40},          # Increased lambda for less noise
        'salt_pepper': {'amount': 0.03},        # Reduced noise level
        'speckle': {'mean': 0, 'sigma': 0.08}   # Reduced noise level
    },
    'VALIDATION_SPLIT': 0.2,
    'USE_MIXED_PRECISION': False,  # Disabled mixed precision (can cause issues on some hardware)
    'MODEL_TYPE': 'autoencoder',   # 'autoencoder', 'unet', 'dncnn'
    'SCHEDULER': True,             # Added learning rate scheduler
    'SCHEDULER_PATIENCE': 5,       # Increased patience for learning rate reduction
    'SCHEDULER_FACTOR': 0.5,       # Factor by which to reduce learning rate
    'loss': 'combined',            # Use combined loss function for better results
    'loss_weights': {              # Adjusted loss weights for better perceptual quality
        'mse': 0.5,                # Reduced MSE weight
        'ssim': 0.4,               # Increased SSIM weight for better structural similarity
        'perceptual': 0.1,         # Maintained perceptual loss for better visual quality
        'edge': 0.2                # Added edge preservation loss for detail retention
    },
    'DATA_AUGMENTATION': True,     # Enable data augmentation
}

# Data Processing Configuration
DATA_CONFIG = {
    'target_size': (128, 128),     # Changed back to match existing model
    'noise_types': ['gaussian', 'poisson', 'salt_pepper', 'speckle'], # Multiple noise types for robustness
    'default_noise': 'gaussian',   # Default noise type
    'gaussian_params': {
        'mean': 0,
        'std': 0.08,               # Reduced noise level
    },
    'salt_pepper_params': {
        'amount': 0.03,            # Reduced noise level
    },
    'speckle_params': {
        'mean': 0,
        'var': 0.08,               # Reduced noise level
    },
}

# Paths
PATHS = {
    'TRAINING_DATA': os.path.join('data', 'training'),
    'VALIDATION_DATA': os.path.join('data', 'validation'),
    'MODELS': os.path.join('saved_models'),  # Updated to match your existing directory structure
    'RESULTS': os.path.join('results'),
    'LOGS': os.path.join('logs'),
}

# Ensure paths exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)