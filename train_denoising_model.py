import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.data.experimental import AUTOTUNE
import random
from PIL import Image
import glob

# Check if TensorFlow can see the GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Download the Berkeley dataset
bsds500_path = kagglehub.dataset_download('balraj98/berkeley-segmentation-dataset-500-bsds500')

# Set up paths for train, test, and val datasets based on the actual dataset structure
train_images_path = os.path.join(bsds500_path, 'images', 'train')
test_images_path = os.path.join(bsds500_path, 'images', 'test')
val_images_path = os.path.join(bsds500_path, 'images', 'val')

# Print the paths to verify
print(f"Train images path: {train_images_path}")
print(f"Test images path: {test_images_path}")
print(f"Validation images path: {val_images_path}")

# Function to list image files
def get_image_files(directory):
    return glob.glob(os.path.join(directory, "*.jpg"))

# Get image file lists
train_files = get_image_files(train_images_path)
val_files = get_image_files(val_images_path)
test_files = get_image_files(test_images_path)

print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")
print(f"Test images: {len(test_files)}")

# If no images found, try to find the correct path structure
if len(train_files) == 0:
    print("No images found. Attempting to find the correct path structure...")
    # Try with the original path structure
    alt_train_path = os.path.join(bsds500_path, 'BSR', 'BSDS500', 'data', 'images', 'train')
    if os.path.exists(alt_train_path):
        print(f"Found alternative path: {alt_train_path}")
        train_images_path = os.path.join(bsds500_path, 'BSR', 'BSDS500', 'data', 'images', 'train')
        test_images_path = os.path.join(bsds500_path, 'BSR', 'BSDS500', 'data', 'images', 'test')
        val_images_path = os.path.join(bsds500_path, 'BSR', 'BSDS500', 'data', 'images', 'val')
        
        # Get image file lists again
        train_files = get_image_files(train_images_path)
        val_files = get_image_files(val_images_path)
        test_files = get_image_files(test_images_path)
        
        print(f"Updated training images: {len(train_files)}")
        print(f"Updated validation images: {len(val_files)}")
        print(f"Updated test images: {len(test_files)}")
    else:
        # List the top-level directories to help debugging
        print(f"Contents of dataset directory: {os.listdir(bsds500_path)}")
        if os.path.exists(os.path.join(bsds500_path, 'BSR')):
            print(f"Contents of BSR directory: {os.listdir(os.path.join(bsds500_path, 'BSR'))}")

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 100
NOISE_FACTOR = 0.1  # Amount of noise to add

# Function to add noise to images
def add_noise(img, noise_factor=NOISE_FACTOR):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0., 1.)

# Load and preprocess an image
def load_and_preprocess_image(image_path):
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Normalize and resize
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    
    return img

# Create noisy and clean pairs
def create_dataset(file_list, batch_size=BATCH_SIZE, is_training=True):
    def process_path(file_path):
        img = load_and_preprocess_image(file_path)
        
        # Data augmentation for training
        if is_training:
            if random.random() > 0.5:
                img = tf.image.flip_left_right(img)
            if random.random() > 0.5:
                img = tf.image.flip_up_down(img)
        
        # Add noise (TensorFlow operation)
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=NOISE_FACTOR, dtype=tf.float32)
        noisy_img = img + noise
        noisy_img = tf.clip_by_value(noisy_img, 0.0, 1.0)
        
        return noisy_img, img
    
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000) if is_training else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

# Create datasets
train_dataset = create_dataset(train_files, is_training=True)
val_dataset = create_dataset(val_files, is_training=False)
test_dataset = create_dataset(test_files, is_training=False)

# Define the U-Net model for denoising
def build_unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output
    outputs = Conv2D(3, 1, activation='sigmoid')(conv7)
    
    model = Model(inputs, outputs)
    return model

# Set up MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

with strategy.scope():
    # Build and compile the model
    model = build_unet_model()
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae', 'mse']
    )

# Display model summary
model.summary()

# Callbacks - Updated file extension from .h5 to .keras for TF 2.18+
checkpoint = ModelCheckpoint(
    'denoising_model_best.keras',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Save the final model - Updated file extension from .h5 to .keras for TF 2.18+
model.save('denoising_model_final.keras')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png')

# Save the model in TensorFlow SavedModel format (more compatible) - Updated to use directory path
try:
    # Create directory if it doesn't exist
    os.makedirs('denoising_model_savedmodel', exist_ok=True)
    # Save model using tf.saved_model API instead of keras direct save
    tf.saved_model.save(model, 'denoising_model_savedmodel')
    print("Model saved successfully using tf.saved_model.save()")
except Exception as e:
    print(f"Error using tf.saved_model.save(): {e}")
    # Fallback to standard Keras save with .keras extension
    model.save('denoising_model_savedmodel.keras')
    print("Fallback: Model saved with .keras extension")

# Test the model on a few examples
def plot_denoising_results(model, test_dataset, num_samples=5):
    plt.figure(figsize=(15, 5 * num_samples))
    batch = next(iter(test_dataset))
    noisy_images, clean_images = batch
    
    # Only use the specified number of samples
    noisy_images = noisy_images[:num_samples]
    clean_images = clean_images[:num_samples]
    
    # Get predictions
    predictions = model.predict(noisy_images)
    
    for i in range(num_samples):
        # Original image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(clean_images[i])
        plt.title('Original')
        plt.axis('off')
        
        # Noisy image
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(noisy_images[i])
        plt.title('Noisy')
        plt.axis('off')
        
        # Denoised image
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(predictions[i])
        plt.title('Denoised')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_results.png')

# Get a batch of test data and visualize results
try:
    plot_denoising_results(model, test_dataset)
    print("Denoising results saved to 'denoising_results.png'")
except Exception as e:
    print(f"Error visualizing results: {e}")

# Print final message
print("Training complete!")
print("Models saved as:")
print("  - denoising_model_best.keras (Best weights)")
print("  - denoising_model_final.keras (Final weights)")
print("  - denoising_model_savedmodel (TensorFlow SavedModel format)")
print("\nTo load the model in your local environment with TensorFlow 2.19.0:")
print("import tensorflow as tf")
print("model = tf.keras.models.load_model('denoising_model_savedmodel')") 