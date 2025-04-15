"""
Configuration file for noise reduction system.
Contains parameters for model, training, and preprocessing.
"""

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (256, 256, 3),  # Default input shape (height, width, channels)
    'latent_dim': 128,             # Size of the latent space
    'filters': [64, 128, 256, 512], # Increased number of filters in each convolutional layer
    'kernel_size': 3,              # Size of convolutional kernels
    'activation': 'relu',          # Activation function
    'final_activation': 'sigmoid', # Final layer activation
    'use_skip_connections': True,  # Enable skip connections (U-Net style)
    'use_residual_blocks': True,   # Enable residual blocks
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 16,              # Smaller batch size for better convergence
    'epochs': 100,                 # More training epochs
    'learning_rate': 0.0001,       # Lower learning rate for finer convergence
    'validation_split': 0.2,
    'optimizer': 'adam',
    'loss': 'combined',            # Use combined loss (MSE + SSIM + Perceptual)
    'loss_weights': {              # Weights for combined loss
        'mse': 0.7,                # Mean Squared Error weight
        'ssim': 0.2,               # SSIM loss weight
        'perceptual': 0.1,         # Perceptual loss weight
    }
}

# Data Processing Configuration
DATA_CONFIG = {
    'target_size': (256, 256),  # Size to resize images to
    'noise_types': ['gaussian', 'salt_pepper', 'speckle'],  # Available noise types
    'default_noise': 'gaussian',  # Default noise type
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
    'model_save_path': 'saved_models',
    'results_path': 'results',
}