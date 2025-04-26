"""
Configuration file for noise reduction system.
Contains parameters for model, training, and preprocessing.
"""

import os

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (128, 128, 3),  # Reduced from 256x256 to save memory
    'latent_dim': 128,             # Increased from 64 for better feature representation
    'filters': [32, 64, 128, 256], # Added more filters for better denoising
    'kernel_size': 3,              # Size of convolutional kernels
    'activation': 'relu',          # Activation function
    'final_activation': 'sigmoid', # Final layer activation
    'use_skip_connections': True,  # Enable skip connections (U-Net style)
    'use_residual_blocks': True,   # Enabled for better gradient flow and feature preservation
    'use_attention': True,         # Added attention mechanism to focus on important features
    'dropout_rate': 0.2,           # Added dropout for regularization
}

# Training Configuration
TRAINING_CONFIG = {
    'BATCH_SIZE': 4,               # Kept small to avoid memory issues
    'EPOCHS': 50,                  # Increased for better fine-tuning
    'LEARNING_RATE': 0.0005,       # Reduced for fine-tuning
    'EARLY_STOPPING_PATIENCE': 10, # Increased to avoid early termination
    'INPUT_SHAPE': (128, 128, 3),  # Maintained size
    'NOISE_TYPES': ['gaussian', 'poisson', 'salt_pepper'], # Use multiple noise types for robustness
    'NOISE_PARAMS': {
        'gaussian': {'mean': 0, 'sigma': 0.1},
        'poisson': {'lambda_val': 30},
        'salt_pepper': {'amount': 0.05},
        'speckle': {'mean': 0, 'sigma': 0.1}
    },
    'VALIDATION_SPLIT': 0.2,
    'USE_MIXED_PRECISION': False,  # Disabled mixed precision (can cause issues on some hardware)
    'MODEL_TYPE': 'autoencoder',   # 'autoencoder', 'unet', 'dncnn'
    'SCHEDULER': True,             # Added learning rate scheduler
    'SCHEDULER_PATIENCE': 3,       # Patience for learning rate reduction
    'SCHEDULER_FACTOR': 0.5,       # Factor by which to reduce learning rate
    'loss': 'combined',            # Use combined loss function for better results
    'loss_weights': {              # Adjusted loss weights for better perceptual quality
        'mse': 0.6,
        'ssim': 0.3,               # Increased SSIM weight for structural similarity
        'perceptual': 0.1          # Including perceptual loss for better visual quality
    },
    'DATA_AUGMENTATION': True,     # Enable data augmentation
}

# Data Processing Configuration
DATA_CONFIG = {
    'target_size': (128, 128),     # Maintained size
    'noise_types': ['gaussian', 'poisson', 'salt_pepper'], # Multiple noise types for robustness
    'default_noise': 'gaussian',   # Default noise type
    'gaussian_params': {
        'mean': 0,
        'std': 0.1,
    },
    'salt_pepper_params': {
        'amount': 0.05,
    },
    'speckle_params': {
        'mean': 0,
        'var': 0.1,
    },
}

# Paths
PATHS = {
    'TRAINING_DATA': os.path.join('data', 'training'),
    'VALIDATION_DATA': os.path.join('data', 'validation'),
    'MODELS': os.path.join('models'),
    'RESULTS': os.path.join('results'),
    'LOGS': os.path.join('logs'),
}

# Ensure paths exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)