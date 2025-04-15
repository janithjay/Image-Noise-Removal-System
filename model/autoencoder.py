"""
Autoencoder architecture for noise reduction.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Add, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import MODEL_CONFIG, TRAINING_CONFIG
from .losses import get_combined_loss

class Autoencoder:
    """
    Convolutional Autoencoder for noise reduction in images.
    """
    def __init__(self, input_shape=None):
        """
        Initialize the autoencoder model.
        
        Args:
            input_shape: Tuple of (height, width, channels)
        """
        self.input_shape = input_shape or MODEL_CONFIG['input_shape']
        self.filters = MODEL_CONFIG['filters']
        self.kernel_size = MODEL_CONFIG['kernel_size']
        self.activation = MODEL_CONFIG['activation']
        self.final_activation = MODEL_CONFIG['final_activation']
        self.latent_dim = MODEL_CONFIG['latent_dim']
        
        self.learning_rate = TRAINING_CONFIG['learning_rate']
        self.loss = TRAINING_CONFIG['loss']
        self.optimizer_name = TRAINING_CONFIG['optimizer']

        self.model = self._build_model()
    
    def _build_encoder(self, inputs):
        """
        Build the encoder part of the autoencoder.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Encoded representation
        """
        x = inputs
        
        # Encoder - Downsampling
        for filters in self.filters:
            x = Conv2D(filters, self.kernel_size, activation=self.activation, padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
        
        return x
    
    def _build_decoder(self, encoded):
        """
        Build the decoder part of the autoencoder.
        
        Args:
            encoded: Encoded representation
            
        Returns:
            Reconstructed output
        """
        x = encoded
        
        # Decoder - Upsampling
        for filters in reversed(self.filters):
            x = Conv2D(filters, self.kernel_size, activation=self.activation, padding='same')(x)
            x = BatchNormalization()(x)
            x = UpSampling2D((2, 2))(x)
        
        # Output layer
        decoded = Conv2D(self.input_shape[2], self.kernel_size, activation=self.final_activation, padding='same')(x)
        
        return decoded
    
    def _build_model(self):
        """
        Build the complete autoencoder model.
        
        Returns:
            Compiled Keras model
        """
        # Input
        inputs = Input(shape=self.input_shape)
        
        # Encoder
        encoded = self._build_encoder(inputs)
        
        # Decoder
        decoded = self._build_decoder(encoded)
        
        # Create model
        model = Model(inputs, decoded)
        
        # Compile model
        if self.optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        else:
            optimizer = self.optimizer_name
        
        # Use appropriate loss function
        if self.loss == 'mse':
            loss_function = 'mse'
        elif self.loss == 'combined':
            alpha = TRAINING_CONFIG['loss_weights']['mse']
            beta = TRAINING_CONFIG['loss_weights']['ssim']
            gamma = TRAINING_CONFIG['loss_weights']['perceptual']
            loss_function = get_combined_loss(alpha, beta, gamma)
        else:
            loss_function = self.loss
            
        model.compile(optimizer=optimizer, loss=loss_function)
        
        return model
    
    def summary(self):
        """
        Print model summary.
        """
        return self.model.summary()
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    @staticmethod
    def load(filepath):
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        return tf.keras.models.load_model(filepath)


def residual_block(x, filters, kernel_size=3):
    """
    Create a residual block.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size for convolutions
        
    Returns:
        Output tensor
    """
    # Shortcut connection
    shortcut = x
    
    # First convolution
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Second convolution
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add shortcut
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.2)(x)
    
    return x


class DenoisingAutoencoder(Autoencoder):
    """
    Specialized autoencoder for denoising tasks using U-Net architecture with skip connections.
    """
    def __init__(self, input_shape=None):
        """
        Initialize the denoising autoencoder model.
        
        Args:
            input_shape: Tuple of (height, width, channels)
        """
        # Override the build_model method before calling the parent constructor
        self._original_build_model = self._build_model
        self._build_model = self._build_unet_model
        
        super().__init__(input_shape)
        
        # Restore original build_model method
        self._build_model = self._original_build_model
    
    def _build_unet_model(self):
        """
        Build a U-Net model with skip connections and residual blocks.
        
        Returns:
            Compiled Keras model
        """
        # Input
        inputs = Input(shape=self.input_shape)
        
        # Use skip connections if enabled
        use_skip = MODEL_CONFIG.get('use_skip_connections', False)
        use_residual = MODEL_CONFIG.get('use_residual_blocks', False)
        
        # Store encoder outputs for skip connections
        skip_connections = []
        
        # Encoder
        x = inputs
        for i, filters in enumerate(self.filters):
            # First convolution block
            x = Conv2D(filters, self.kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            
            # Apply residual block if enabled
            if use_residual:
                x = residual_block(x, filters, self.kernel_size)
            
            # Second convolution block
            x = Conv2D(filters, self.kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            
            # Store for skip connection
            if use_skip and i < len(self.filters) - 1:
                skip_connections.append(x)
            
            # Apply pooling except for the last encoder layer
            if i < len(self.filters) - 1:
                x = MaxPooling2D((2, 2))(x)
                
            # Add dropout for regularization in deeper layers
            if i >= 1:
                x = Dropout(0.2)(x)
        
        # Bottleneck
        bottleneck = x
        
        # Decoder
        for i, filters in enumerate(reversed(self.filters[:-1])):
            # Upsampling
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filters, self.kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            
            # Skip connection
            if use_skip:
                x = Concatenate()([x, skip_connections[-(i+1)]])
            
            # First convolution block after skip connection
            x = Conv2D(filters, self.kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            
            # Apply residual block if enabled
            if use_residual:
                x = residual_block(x, filters, self.kernel_size)
            
            # Second convolution block
            x = Conv2D(filters, self.kernel_size, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
        
        # Output layer
        outputs = Conv2D(self.input_shape[2], 1, activation=self.final_activation, padding='same')(x)
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        if self.optimizer_name.lower() == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        else:
            optimizer = self.optimizer_name
        
        # Use appropriate loss function
        if self.loss == 'mse':
            loss_function = 'mse'
        elif self.loss == 'combined':
            alpha = TRAINING_CONFIG['loss_weights']['mse']
            beta = TRAINING_CONFIG['loss_weights']['ssim']
            gamma = TRAINING_CONFIG['loss_weights']['perceptual']
            loss_function = get_combined_loss(alpha, beta, gamma)
        else:
            loss_function = self.loss
            
        model.compile(optimizer=optimizer, loss=loss_function)
        
        return model