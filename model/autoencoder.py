"""
Autoencoder architecture for noise reduction.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import MODEL_CONFIG, TRAINING_CONFIG

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
            
        model.compile(optimizer=optimizer, loss=self.loss)
        
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


class DenoisingAutoencoder(Autoencoder):
    """
    Specialized autoencoder for denoising tasks.
    """
    def __init__(self, input_shape=None):
        super().__init__(input_shape)
        
        # Optionally modify the architecture specifically for denoising
        # For example, add more dropout layers, different layer configurations, etc.