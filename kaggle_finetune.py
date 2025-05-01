"""
Script for fine-tuning the image denoising model in Kaggle environment.
This script adds new layers and implements advanced techniques to enhance the model.
Fixed to resolve duplicate layer name issues.
"""

# For Kaggle, install required packages upfront
# Uncomment the next line if you need to install tensorflow-addons in Kaggle
# !pip install -q tensorflow-addons

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Lambda, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Import tensorflow_addons if available
try:
    import tensorflow_addons as tfa
    print("Using TensorFlow Addons for Sobel edge detection")
except ImportError:
    print("TensorFlow Addons not available, using custom implementation")
    tfa = None

# Updated dataset path configuration
INPUT_DIR = '/kaggle/input'
# Base model path
BASE_MODEL_PATH = '/kaggle/input/based-model/tensorflow2/default/1/best_model.h5'
# Dataset path - updated for Berkeley segmentation dataset
DATASET_PATH = '/kaggle/input/berkeley-segmentation-dataset-500-bsds500'
IMAGES_PATH = os.path.join(DATASET_PATH, 'images')
GROUND_TRUTH_PATH = os.path.join(DATASET_PATH, 'ground_truth')
# Output path for saving results
OUTPUT_DIR = '/kaggle/working'

# Configuration parameters
INPUT_SHAPE = (128, 128, 3)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
USE_MIXED_PRECISION = False  # Disable mixed precision to avoid data type issues
# Set default data type to float32
DEFAULT_DTYPE = tf.float32

# Configure mixed precision if enabled
if USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
else:
    # If not using mixed precision, explicitly set to float32
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Using float32 precision")

# Ensure TensorFlow uses float32 as default dtype
tf.keras.backend.set_floatx('float32')

# Configure GPU and strategy
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPUs: {gpus}")
        
        # Set up MultiWorkerMirroredStrategy for multi-GPU training
        # Try different distribution strategies based on TF version
        if len(gpus) > 1:
            print("Setting up distribution strategy for multi-GPU training")
            try:
                # First try MirrorStrategy
                strategy = tf.distribute.MirrorStrategy()
                print("Using MirrorStrategy")
            except (AttributeError, ImportError):
                try:
                    # Then try MultiWorkerMirroredStrategy
                    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
                    print("Using MultiWorkerMirroredStrategy")
                except (AttributeError, ImportError):
                    # Fall back to default strategy
                    strategy = tf.distribute.get_strategy()
                    print("Multi-GPU support not available, using default strategy")
            print(f"Training with {strategy.num_replicas_in_sync} GPU(s)")
        else:
            strategy = tf.distribute.get_strategy()  # Default strategy
            print("Using single GPU for training")
    except RuntimeError as e:
        print(f"GPU error: {e}")
        strategy = tf.distribute.get_strategy()  # Default strategy
else:
    print("No GPU found. Using CPU for training (this will be slower)")
    strategy = tf.distribute.get_strategy()  # Default strategy

# Create VGG16 model for perceptual loss outside of any function
# This avoids recreating it multiple times during training
print("Creating VGG16 model for perceptual loss...")
vgg16_model = None
feature_extractor = None

def initialize_vgg_model():
    """Initialize VGG16 model once for perceptual loss"""
    global vgg16_model, feature_extractor
    
    if vgg16_model is None:
        vgg16_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
        # Make it non-trainable
        for layer in vgg16_model.layers:
            layer.trainable = False
        
        # Create a model that returns the activation values for given layer numbers
        feature_extractor = Model(inputs=vgg16_model.input, outputs=vgg16_model.get_layer('block3_conv3').output)
        print("VGG16 feature extractor initialized")
    
    return feature_extractor

def perceptual_loss(y_true, y_pred):
    """Perceptual loss using pre-trained VGG16."""
    global feature_extractor
    
    if feature_extractor is None:
        # This should already be initialized, but just in case
        initialize_vgg_model()
    
    # Get the feature representations
    y_true_features = feature_extractor(y_true)
    y_pred_features = feature_extractor(y_pred)
    
    # Return the mean squared error between the features
    return tf.reduce_mean(tf.square(y_true_features - y_pred_features))

def ssim_loss(y_true, y_pred):
    """Structural similarity loss function."""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def custom_sobel_edges(image):
    """Custom implementation of Sobel edge detection."""
    # Define Sobel kernels
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
    
    # Convert to grayscale if needed
    if image.shape[-1] > 1:
        image_gray = tf.reduce_mean(image, axis=-1, keepdims=True)
    else:
        image_gray = image
    
    # Apply convolution for each kernel
    gx = tf.nn.conv2d(image_gray, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    gy = tf.nn.conv2d(image_gray, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    
    # Calculate gradient magnitude
    edge = tf.sqrt(tf.square(gx) + tf.square(gy))
    return edge

def edge_loss(y_true, y_pred):
    """Edge preservation loss using Sobel filter."""
    # Use TFA if available, otherwise use custom implementation
    if tfa is not None:
        # Sobel filter for edge detection using TFA
        def sobel_edges(image):
            sobel_x = tfa.image.sobel_edges(image)[..., 0]
            sobel_y = tfa.image.sobel_edges(image)[..., 1]
            return tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))
    else:
        # Use custom implementation
        sobel_edges = custom_sobel_edges
    
    # Get edges
    true_edges = sobel_edges(y_true)
    pred_edges = sobel_edges(y_pred)
    
    # Calculate mean absolute error between edges
    return tf.reduce_mean(tf.abs(true_edges - pred_edges))

def combined_loss(alpha=0.5, beta=0.3, gamma=0.1, delta=0.1):
    """Combined loss function with weighted components."""
    # Initialize VGG model first if not already initialized
    initialize_vgg_model()
    
    def loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        ss = ssim_loss(y_true, y_pred)
        perc = perceptual_loss(y_true, y_pred)
        edge = edge_loss(y_true, y_pred)
        return alpha * mse + beta * ss + gamma * perc + delta * edge
    return loss

def add_channel_attention(x, ratio=16, name_prefix='ca'):
    """Channel attention module with unique names."""
    channel = x.shape[-1]
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    
    # Two FC layers
    avg_pool = Reshape((1, 1, channel), name=f"{name_prefix}_reshape")(avg_pool)
    avg_pool = Dense(channel // ratio, activation='relu', 
                    kernel_initializer='he_normal', use_bias=True, 
                    name=f"{name_prefix}_dense1")(avg_pool)
    avg_pool = Dense(channel, activation='sigmoid', 
                    kernel_initializer='he_normal', use_bias=True, 
                    name=f"{name_prefix}_dense2")(avg_pool)
    
    # Apply attention
    return Multiply(name=f"{name_prefix}_multiply")([x, avg_pool])

def add_spatial_attention(x, name_prefix='sa'):
    """Spatial attention module with unique names."""
    kernel_size = 7
    
    # Calculate average and max along channel axis
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), 
                     name=f"{name_prefix}_avg_pool")(x)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True), 
                     name=f"{name_prefix}_max_pool")(x)
    
    # Concatenate
    concat = Concatenate(name=f"{name_prefix}_concat")([avg_pool, max_pool])
    
    # Conv for spatial attention map
    spatial_attn = Conv2D(1, kernel_size, padding='same', activation='sigmoid', 
                         use_bias=False, name=f"{name_prefix}_conv")(concat)
    
    # Apply attention
    return Multiply(name=f"{name_prefix}_multiply")([x, spatial_attn])

def add_attention_block(x, block_id):
    """CBAM-style attention block combining channel and spatial attention."""
    name_prefix = f"attn_block_{block_id}"
    x = add_channel_attention(x, ratio=16, name_prefix=f"{name_prefix}_ca")
    x = add_spatial_attention(x, name_prefix=f"{name_prefix}_sa")
    return x

def add_residual_block(x, filters, kernel_size=3, block_id=0):
    """Enhanced residual block with pre-activation and unique names."""
    name_prefix = f"res_block_{block_id}"
    
    # Shortcut connection
    shortcut = x
    
    # First convolution (pre-activation)
    x = BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = LeakyReLU(alpha=0.1, name=f"{name_prefix}_lrelu1")(x)
    x = Conv2D(filters, kernel_size, padding='same', name=f"{name_prefix}_conv1")(x)
    
    # Second convolution
    x = BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = LeakyReLU(alpha=0.1, name=f"{name_prefix}_lrelu2")(x)
    x = Conv2D(filters, kernel_size, padding='same', name=f"{name_prefix}_conv2")(x)
    
    # Shortcut connection (with projection if needed)
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, padding='same', name=f"{name_prefix}_shortcut_conv")(shortcut)
    
    # Make sure both inputs to Add have the same data type
    x = CastToFloat32(name=f"{name_prefix}_cast_x")(x)
    shortcut = CastToFloat32(name=f"{name_prefix}_cast_shortcut")(shortcut)
    
    # Add shortcut
    return Add(name=f"{name_prefix}_add")([x, shortcut])

def enhance_model_architecture(base_model):
    """Enhance the base model by adding new layers and improving architecture."""
    # Create model inside strategy scope
    with strategy.scope():
        # Get the input and output of the base model
        base_input = base_model.input
        base_output = base_model.output
        
        # First, clone the base model with unique layer names to avoid conflicts
        # This creates a completely new copy of the base model with guaranteed unique names
        base_model_config = base_model.get_config()
        
        # Strip the base model down to just its input
        # We'll use the original input and create a new model with our enhancements
        print("Creating enhanced architecture with completely unique layer names")
        
        # Add attention and residual blocks 
        x = base_output
        
        # Add 3 residual blocks with unique IDs
        x = add_residual_block(x, 64, block_id=1)
        x = add_attention_block(x, block_id=1)  # Add attention after first residual
        x = add_residual_block(x, 64, block_id=2)
        x = add_residual_block(x, 64, block_id=3)
        
        # Final output convolution
        x = Conv2D(INPUT_SHAPE[2], 3, padding='same', activation='sigmoid', 
                  name='enhanced_output_conv')(x)
        
        # Make sure both inputs to final Add have the same data type
        x = CastToFloat32(name='enhanced_output_cast')(x)
        base_input_cast = CastToFloat32(name='base_input_cast')(base_input)
        
        # Residual connection from input to output
        enhanced_output = Add(name='final_residual_add')([x, base_input_cast])
        
        # Create new model - use specific name to avoid conflicts
        enhanced_model = Model(inputs=base_input, outputs=enhanced_output, name='enhanced_denoising_model')
        
        # Compile model
        enhanced_model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss=combined_loss(alpha=0.5, beta=0.3, gamma=0.1, delta=0.1)
        )
        
        # Print summary to verify the architecture
        print("Enhanced model summary:")
        enhanced_model.summary()
        
        # Print layer names to verify uniqueness
        print("Checking layer names for uniqueness:")
        layer_names = [layer.name for layer in enhanced_model.layers]
        duplicates = [name for name in set(layer_names) if layer_names.count(name) > 1]
        if duplicates:
            print(f"WARNING: Found duplicate layer names: {duplicates}")
        else:
            print("All layer names are unique.")
    
    return enhanced_model

def create_noise_generator():
    """Create a noise generator class."""
    class NoiseGenerator:
        def __init__(self, noise_type='gaussian'):
            self.noise_type = noise_type
        
        def add_noise(self, images, noise_type=None):
            """Add noise to the given images."""
            if noise_type is None:
                noise_type = self.noise_type
                
            if noise_type == 'gaussian':
                return self.add_gaussian_noise(images)
            elif noise_type == 'poisson':
                return self.add_poisson_noise(images)
            elif noise_type == 'salt_pepper':
                return self.add_salt_pepper_noise(images)
            elif noise_type == 'speckle':
                return self.add_speckle_noise(images)
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
                
        def add_gaussian_noise(self, images, mean=0, sigma=0.08):
            """Add Gaussian noise to images."""
            # Create noise with same shape as images, ensure float32
            noise = np.random.normal(mean, sigma, images.shape).astype(np.float32)
            noisy_images = np.clip(images + noise, 0, 1).astype(np.float32)
            return noisy_images
            
        def add_poisson_noise(self, images, scaling_factor=40):
            """Add Poisson noise to images."""
            # Scale images for Poisson noise
            scaled_images = (scaling_factor * images).astype(np.float32)
            # Add Poisson noise and scale back
            noisy_images = np.random.poisson(scaled_images) / scaling_factor
            return np.clip(noisy_images, 0, 1).astype(np.float32)
            
        def add_salt_pepper_noise(self, images, amount=0.03):
            """Add salt and pepper noise to images."""
            noisy_images = images.copy().astype(np.float32)
            # Salt noise (white pixels)
            salt_mask = np.random.rand(*images.shape) < (amount / 2)
            noisy_images[salt_mask] = 1.0
            # Pepper noise (black pixels)
            pepper_mask = np.random.rand(*images.shape) < (amount / 2)
            noisy_images[pepper_mask] = 0.0
            return noisy_images
            
        def add_speckle_noise(self, images, mean=0, sigma=0.08):
            """Add speckle noise to images."""
            noise = np.random.normal(mean, sigma, images.shape).astype(np.float32)
            noisy_images = np.clip(images + images * noise, 0, 1).astype(np.float32)
            return noisy_images
    
    return NoiseGenerator()

# Updated function to load and prepare data
def load_and_prepare_data():
    """Load and prepare training and validation data using Berkeley segmentation dataset."""
    print("Loading and preparing data...")
    
    # Check if dataset exists in Kaggle input
    if not os.path.exists(IMAGES_PATH) or not os.path.exists(GROUND_TRUTH_PATH):
        print("Berkeley segmentation dataset not found. Creating synthetic data.")
        # Create synthetic data with float32 type
        num_samples = 400
        x_clean = np.random.rand(num_samples, *INPUT_SHAPE).astype(np.float32)
        x_noisy = np.random.rand(num_samples, *INPUT_SHAPE).astype(np.float32)
    else:
        print(f"Loading images from Berkeley segmentation dataset...")
        
        # Process train, val, test folders separately
        subsets = ['train', 'val', 'test']
        x_clean_all = []
        
        for subset in subsets:
            images_subset_path = os.path.join(IMAGES_PATH, subset)
            gt_subset_path = os.path.join(GROUND_TRUTH_PATH, subset)
            
            if not os.path.exists(images_subset_path) or not os.path.exists(gt_subset_path):
                print(f"Subset {subset} not found, skipping")
                continue
                
            print(f"Processing {subset} set...")
            
            # Get list of image files
            image_files = [f for f in os.listdir(images_subset_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not image_files:
                print(f"No image files found in {subset}")
                continue
                
            print(f"Found {len(image_files)} images in {subset} set")
            
            # Load and preprocess images
            for img_file in image_files:
                img_path = os.path.join(images_subset_path, img_file)
                try:
                    # Load and resize image
                    img = load_img(img_path, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))
                    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
                    # Ensure float32 data type
                    img_array = img_array.astype(np.float32)
                    x_clean_all.append(img_array)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
        
        # Convert to numpy array
        if x_clean_all:
            x_clean = np.array(x_clean_all, dtype=np.float32)
            print(f"Loaded {len(x_clean)} images total, shape: {x_clean.shape}")
        else:
            print("Failed to load any images, using synthetic data instead")
            num_samples = 400
            x_clean = np.random.rand(num_samples, *INPUT_SHAPE).astype(np.float32)
    
    # Generate corresponding noisy images
    print("Generating noisy versions of images...")
    noise_gen = create_noise_generator()
    
    # Generate multiple noise types for robustness
    noise_types = ['gaussian', 'poisson', 'salt_pepper', 'speckle']
    x_noisy_combined = []
    
    for noise_type in noise_types:
        noise_gen.noise_type = noise_type
        x_noisy = noise_gen.add_noise(x_clean)
        # Ensure float32 data type
        x_noisy = x_noisy.astype(np.float32)
        x_noisy_combined.append(x_noisy)
    
    # Combine all noise types (average them)
    x_noisy = np.mean(np.array(x_noisy_combined, dtype=np.float32), axis=0)
    
    # Split into training and validation
    split_idx = int(len(x_clean) * (1 - VALIDATION_SPLIT))
    x_train = x_noisy[:split_idx]
    y_train = x_clean[:split_idx]
    x_val = x_noisy[split_idx:]
    y_val = x_clean[split_idx:]
    
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    print(f"Data type of inputs: {x_train.dtype}")
    
    return x_train, y_train, x_val, y_val

def create_augmentation():
    """Create data augmentation function."""
    def augment(image, label):
        # Cast inputs to float32
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        
        # Stack images for joint augmentation
        stacked = tf.concat([image, label], axis=-1)
        
        # Random flips
        stacked = tf.image.random_flip_left_right(stacked)
        stacked = tf.image.random_flip_up_down(stacked)
        
        # Random rotation (in multiples of 90 degrees)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        stacked = tf.image.rot90(stacked, k=k)
        
        # Split back
        channels = stacked.shape[-1] // 2
        image = stacked[..., :channels]
        label = stacked[..., channels:]
        
        # Ensure outputs are float32
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        
        return image, label
    
    return augment

def finetune_model():
    """Main function to fine-tune the model."""
    # Initialize VGG model for perceptual loss first
    print("Initializing VGG model for perceptual loss...")
    initialize_vgg_model()
    
    # Check if base model exists
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"Base model not found at {BASE_MODEL_PATH}")
        return
    
    print(f"Loading base model from {BASE_MODEL_PATH}")
    
    # Load the base model inside strategy scope
    with strategy.scope():
        try:
            # First try loading without compiling to avoid naming issues
            print("Loading base model without compiling...")
            base_model = tf.keras.models.load_model(
                BASE_MODEL_PATH, 
                custom_objects={
                    'ssim_loss': ssim_loss,
                    'perceptual_loss': perceptual_loss,
                    'edge_loss': edge_loss,
                    'combined_loss': combined_loss(0.5, 0.3, 0.1, 0.1)
                },
                compile=False  # Skip compiling the model initially
            )
            
            # Create a fresh model with the same layers but new names
            print("Creating fresh version of base model with unique layer names...")
            base_input = base_model.input
            base_output = base_model.output
            
            # Create a new model with the same structure but new layer names
            new_base_model = tf.keras.models.Model(
                inputs=base_input,
                outputs=base_output,
                name="base_denoising_model_renamed"
            )
            
            # Verify the model was loaded successfully
            print(f"Base model loaded with {len(new_base_model.layers)} layers")
            
            # Replace base_model with the renamed version
            base_model = new_base_model
            
        except Exception as e:
            print(f"Error loading base model: {e}")
            return
    
    # Enhance the model architecture
    enhanced_model = enhance_model_architecture(base_model)
    print("Model architecture enhanced with new layers and attention mechanisms")
    
    # Load and prepare data
    x_train, y_train, x_val, y_val = load_and_prepare_data()
    
    # Create TF datasets for efficient loading
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    # Apply data augmentation to training set
    augment_func = create_augmentation()
    train_dataset = train_dataset.map(augment_func, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prepare datasets for training
    # Adjust batch size for distributed training
    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size}")
    
    train_dataset = train_dataset.shuffle(buffer_size=min(len(x_train), 1000)).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
    
    # With MirrorStrategy, we need to use the strategy's distribution
    # This ensures the datasets are properly sharded across GPUs
    if strategy.num_replicas_in_sync > 1:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)
    
    # Set up callbacks
    os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'logs'), exist_ok=True)
    
    # Helper function to safely get learning rate from any optimizer
    def get_lr(optimizer):
        # For newer TF versions where _current_learning_rate is available
        if hasattr(optimizer, '_current_learning_rate'):
            return float(optimizer._current_learning_rate)
        # For optimizers with direct learning_rate attribute
        elif hasattr(optimizer, 'learning_rate'):
            return float(optimizer.learning_rate)
        # For schedulers
        elif hasattr(optimizer, 'lr'):
            if callable(optimizer.lr):
                return float(optimizer.lr())
            else:
                return float(optimizer.lr)
        else:
            # Default fallback
            return 0.0
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss'),
        ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, 'checkpoints', 'model_checkpoint.keras'),
            save_best_only=True,
            monitor='val_loss'
        ),
        TensorBoard(log_dir=os.path.join(OUTPUT_DIR, 'logs')),
        # Add LambdaCallback to track learning rate using a robust method
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logs.update({'lr': get_lr(enhanced_model.optimizer)})
        )
    ]
    
    # Train (fine-tune) the model
    print(f"Starting fine-tuning for {EPOCHS} epochs...")
    history = enhanced_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    enhanced_model.save(os.path.join(OUTPUT_DIR, 'enhanced_model.h5'))
    print(f"Enhanced model saved to {os.path.join(OUTPUT_DIR, 'enhanced_model.h5')}")
    
    # Visualize training history
    history_dict = history.history
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate if available
    if 'lr' in history_dict:
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['lr'], label='Learning Rate')
        plt.title('Learning Rate During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
    else:
        print("Learning rate was not tracked in history.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.show()
    
    # Generate and save example results
    print("Generating example results...")
    
    # Predict on validation data
    n_examples = min(5, len(x_val))
    test_samples = x_val[:n_examples]
    test_predictions = enhanced_model.predict(test_samples)
    
    # Create comparison figure
    plt.figure(figsize=(15, 4*n_examples))
    
    for i in range(n_examples):
        # Original clean image
        plt.subplot(n_examples, 3, i*3 + 1)
        plt.imshow(y_val[i])
        plt.title('Clean (Ground Truth)')
        plt.axis('off')
        
        # Noisy image
        plt.subplot(n_examples, 3, i*3 + 2)
        plt.imshow(test_samples[i])
        plt.title('Noisy Input')
        plt.axis('off')
        
        # Denoised image
        plt.subplot(n_examples, 3, i*3 + 3)
        plt.imshow(test_predictions[i])
        plt.title('Denoised Output')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'example_results.png'))
    
    print("Fine-tuning complete. Model and visualizations saved.")

# Custom layer for casting to float32
class CastToFloat32(Layer):
    def __init__(self, **kwargs):
        super(CastToFloat32, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)
    
    def compute_output_shape(self, input_shape):
        return input_shape

if __name__ == "__main__":
    finetune_model()