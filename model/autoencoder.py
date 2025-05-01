"""
Autoencoder architecture for noise reduction.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Add, Dropout, LeakyReLU, Multiply, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

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
        
        self.learning_rate = TRAINING_CONFIG['LEARNING_RATE']
        self.loss = TRAINING_CONFIG.get('loss', 'mse')
        self.optimizer_name = TRAINING_CONFIG.get('optimizer', 'adam')

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
    def load(filepath, custom_objects=None):
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            custom_objects: Dictionary of custom objects for model loading
            
        Returns:
            Loaded model wrapped in a model wrapper with compile method
        """
        # Import custom loss functions
        from .losses import ssim_loss, perceptual_loss, combined_loss, get_combined_loss
        
        # Define custom objects dictionary to help with model loading
        base_custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'ssim_loss': ssim_loss,
            'perceptual_loss': perceptual_loss,
            'combined_loss': combined_loss,
            # Add other custom losses or metrics if needed
        }
        
        # Merge with additional custom objects if provided
        if custom_objects:
            base_custom_objects.update(custom_objects)
        
        # Load the model with custom objects
        keras_model = tf.keras.models.load_model(filepath, custom_objects=base_custom_objects)
        
        # Create a simple wrapper class that provides the compile method
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
                self.learning_rate = TRAINING_CONFIG['LEARNING_RATE']
                self.loss = TRAINING_CONFIG.get('loss', 'mse')
                self.optimizer_name = TRAINING_CONFIG.get('optimizer', 'adam')
            
            def compile(self, learning_rate=None):
                """
                Compile the model with the specified learning rate.
                
                Args:
                    learning_rate: Learning rate for the optimizer (optional)
                """
                if learning_rate:
                    self.learning_rate = learning_rate
                
                optimizer = Adam(learning_rate=self.learning_rate)
                
                # Use the current loss function from the model
                loss_function = self.model.loss
                
                self.model.compile(optimizer=optimizer, loss=loss_function)
                return self.model
            
            def predict(self, *args, **kwargs):
                return self.model.predict(*args, **kwargs)
            
            def save(self, filepath):
                self.model.save(filepath)
        
        # Return a wrapped model
        return ModelWrapper(keras_model)


def residual_block(x, filters, kernel_size=3, dropout_rate=0.0):
    """
    Create a residual block with improved feature extraction.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size for convolutions
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Output tensor
    """
    # Shortcut connection
    shortcut = x
    
    # First convolution
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Apply dropout if specified
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    
    # Second convolution
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add shortcut
    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.2)(x)
    
    return x


def channel_attention(x, ratio=16):
    """
    Channel attention module - helps model focus on important feature channels.
    
    Args:
        x: Input tensor
        ratio: Reduction ratio for the dense layer
        
    Returns:
        Output tensor with attention applied
    """
    channel = x.shape[-1]
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(x)
    
    # FC layers with shared weights for channel attention
    fc1 = Dense(channel // ratio, activation='relu')(avg_pool)
    fc2 = Dense(channel, activation='sigmoid')(fc1)
    
    # Reshape to proper dimensions
    fc2 = Reshape((1, 1, channel))(fc2)
    
    # Apply attention
    return Multiply()([x, fc2])


def spatial_attention(x):
    """
    Spatial attention module - helps model focus on important spatial regions.
    
    Args:
        x: Input tensor
        
    Returns:
        Output tensor with attention applied
    """
    # Average pooling along channels
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    
    # Max pooling along channels
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    
    # Concatenate pooling results
    concat = Concatenate()([avg_pool, max_pool])
    
    # Apply convolution
    spatial = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    
    # Apply attention
    return Multiply()([x, spatial])


def cbam_block(x, ratio=16):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention.
    
    Args:
        x: Input tensor
        ratio: Reduction ratio for channel attention
        
    Returns:
        Output tensor with attention applied
    """
    # Apply channel attention
    x = channel_attention(x, ratio)
    
    # Apply spatial attention
    x = spatial_attention(x)
    
    return x


class DenoisingAutoencoder(Autoencoder):
    """
    Specialized autoencoder for denoising tasks using an advanced U-Net architecture 
    with skip connections, residual blocks, dense connections, and attention mechanisms.
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
    
    def _dense_block(self, x, filters, kernel_size=3, dropout_rate=0.0):
        """
        Create a dense block with improved feature extraction.
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Kernel size for convolutions
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Output tensor and concatenated features
        """
        concat_feat = x
        
        for i in range(3):  # 3 layers in each dense block
            # Feature computation with concatenated inputs
            x1 = Conv2D(filters, kernel_size, padding='same')(concat_feat)
            x1 = BatchNormalization()(x1)
            x1 = LeakyReLU(alpha=0.2)(x1)
            
            if dropout_rate > 0:
                x1 = Dropout(dropout_rate)(x1)
                
            # Concatenate with previous features for dense connectivity
            concat_feat = Concatenate()([concat_feat, x1])
            
        return concat_feat
    
    def _transition_down(self, x, filters):
        """
        Transition down block for encoder path.
        
        Args:
            x: Input tensor
            filters: Number of filters
            
        Returns:
            Output tensor
        """
        x = Conv2D(filters, 1, padding='same')(x)  # 1x1 conv to reduce channels
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D(2)(x)
        return x
    
    def _transition_up(self, x, filters):
        """
        Transition up block for decoder path.
        
        Args:
            x: Input tensor
            filters: Number of filters
            
        Returns:
            Output tensor
        """
        x = UpSampling2D(2)(x)
        x = Conv2D(filters, 1, padding='same')(x)  # 1x1 conv to reduce channels
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x
    
    def _enhanced_attention_block(self, x, skip_connection):
        """
        Enhanced attention gate that helps focus on relevant features from skip connection.
        
        Args:
            x: Gating signal (from upsampling path)
            skip_connection: Skip connection signal
            
        Returns:
            Attention weighted skip connection
        """
        filters = x.shape[-1]
        
        # Process gating signal
        g = Conv2D(filters, 1, padding='same')(x)
        g = BatchNormalization()(g)
        
        # Process skip connection
        f = Conv2D(filters, 1, padding='same')(skip_connection)
        f = BatchNormalization()(f)
        
        # Attention map
        h = LeakyReLU(alpha=0.2)(g + f)
        h = Conv2D(filters, 1, padding='same')(h)
        h = BatchNormalization()(h)
        h = tf.nn.sigmoid(h)  # Attention coefficients
        
        # Apply attention
        return h * skip_connection
    
    def _build_unet_model(self):
        """
        Build an advanced U-Net model with dense blocks, residual learning,
        enhanced attention mechanisms, and dual-path skip connections.
        
        Returns:
            Compiled Keras model
        """
        # Input
        inputs = Input(shape=self.input_shape)
        
        # Get configuration settings
        use_skip = MODEL_CONFIG.get('use_skip_connections', True)
        use_residual = MODEL_CONFIG.get('use_residual_blocks', True)
        use_attention = MODEL_CONFIG.get('use_attention', True)
        use_residual_learning = MODEL_CONFIG.get('use_residual_learning', True)
        dropout_rate = MODEL_CONFIG.get('dropout_rate', 0.1)
        
        # Store encoder outputs for skip connections
        skip_connections = []
        
        # Initial convolution
        x = Conv2D(self.filters[0], 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Encoder with dense blocks
        for i, filters in enumerate(self.filters):
            # Dense block
            x = self._dense_block(x, filters, dropout_rate=dropout_rate)
            
            # Store for skip connection
            if use_skip and i < len(self.filters) - 1:
                skip_connections.append(x)
            
            # Transition down (except for the last encoder block)
            if i < len(self.filters) - 1:
                x = self._transition_down(x, self.filters[i+1])
        
        # Bottleneck - additional processing
        if use_residual:
            x = residual_block(x, self.filters[-1], dropout_rate=dropout_rate)
        
        if use_attention:
            x = cbam_block(x)
        
        # Decoder - Upsampling with skip connections and attention
        for i, filters in enumerate(reversed(self.filters[:-1])):
            # Transition up
            x = self._transition_up(x, filters)
            
            # Apply attention to skip connection if enabled
            if use_skip:
                skip_index = len(skip_connections) - i - 1
                skip = skip_connections[skip_index]
                
                if use_attention:
                    # Apply attention gates
                    skip = self._enhanced_attention_block(x, skip)
                
                # Concatenate with skip connection
                x = Concatenate()([x, skip])
            
            # Dense block in decoder
            x = self._dense_block(x, filters, dropout_rate=dropout_rate)
            
            # Apply residual block if enabled
            if use_residual:
                x = residual_block(x, filters, dropout_rate=dropout_rate)
        
        # Final convolution layers
        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(32, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Output layer
        x = Conv2D(self.input_shape[2], 3, padding='same', activation=self.final_activation)(x)
        
        # Residual learning - add input to output for easier learning of noise pattern
        if use_residual_learning:
            outputs = Add()([x, inputs])
        else:
            outputs = x
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model with appropriate optimizer and loss
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
            delta = TRAINING_CONFIG['loss_weights'].get('edge', 0.1)  # Default to 0.1 if not defined
            loss_function = get_combined_loss(alpha, beta, gamma, delta)
        else:
            loss_function = self.loss
        
        model.compile(optimizer=optimizer, loss=loss_function)
        
        return model

    def build(self):
        """
        Build and return the model.
        
        Returns:
            The built model
        """
        self.model = self._build_unet_model()
        return self.model
    
    def compile(self, learning_rate=None):
        """
        Compile the model with the specified learning rate.
        
        Args:
            learning_rate: Learning rate for the optimizer (optional)
        """
        if learning_rate:
            self.learning_rate = learning_rate
        
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
        
        self.model.compile(optimizer=optimizer, loss=loss_function)
        
        return self.model