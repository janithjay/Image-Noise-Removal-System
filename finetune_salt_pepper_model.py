#this code used to finetune the previously finetuned model created by finetune_denoising_model.py in kaggle environment

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.data.experimental import AUTOTUNE
import random
from PIL import Image
import glob
from tensorflow.keras.layers import Conv2D, Input, Concatenate
from tensorflow.keras.models import Model

# Check if TensorFlow can see the GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Check if running in Kaggle environment
IN_KAGGLE = os.path.exists('/kaggle/input')
print(f"Running in Kaggle environment: {IN_KAGGLE}")

# Set paths based on environment
if IN_KAGGLE:
    # Path to previously finetuned model in Kaggle
    model_path = '/kaggle/input/finetuned1/keras/default/1/denoising_model_best.keras'
    # Path to salt and pepper dataset in Kaggle
    dataset_path = '/kaggle/input/salt-and-pepper-noise-images'
    output_dir = '/kaggle/working'
else:
    model_path = 'saved_models/denoising_model_best.keras'
    dataset_path = 'salt-and-pepper-noise-images'
    output_dir = 'saved_models'

os.makedirs(output_dir, exist_ok=True)
print(f"Model will be loaded from: {model_path}")
print(f"Dataset will be loaded from: {dataset_path}")
print(f"Outputs will be saved to: {output_dir}")

# Function to find the dataset structure
def find_dataset_paths(base_path):
    """
    Find the dataset paths for Ground_truth and Noisy_folder
    
    Args:
        base_path: Base path of the salt-and-pepper dataset
        
    Returns:
        Dictionary with paths for clean and noisy images
    """
    print(f"Examining dataset structure in: {base_path}")
    
    # Initialize paths
    paths = {
        'clean': None,
        'noisy': None
    }
    
    # Check if the expected folders exist
    ground_truth_path = os.path.join(base_path, 'Ground_truth')
    noisy_path = os.path.join(base_path, 'Noisy_folder')
    
    if os.path.isdir(ground_truth_path):
        paths['clean'] = ground_truth_path
        print(f"Found ground truth images at: {ground_truth_path}")
    
    if os.path.isdir(noisy_path):
        paths['noisy'] = noisy_path
        print(f"Found noisy images at: {noisy_path}")
    
    # Check if we found all paths
    missing_paths = [k for k, v in paths.items() if v is None]
    if missing_paths:
        print(f"Warning: Could not find paths for: {', '.join(missing_paths)}")
    
    return paths

# Function to list image files
def get_image_files(directory):
    """Get all image files in a directory with various extensions"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)

# Function to create dataset of noisy and clean images
def create_paired_dataset(clean_dir, noisy_dir, batch_size=16, is_training=True, validation_split=0.2):
    """
    Create a dataset of paired color clean and grayscale noisy images.
    
    Args:
        clean_dir: Directory containing clean color images
        noisy_dir: Directory containing noisy grayscale images
        batch_size: Batch size for the dataset
        is_training: Whether this is for training (enables data augmentation)
        validation_split: Fraction of data to use for validation
        
    Returns:
        TensorFlow dataset of (noisy_image, clean_image) pairs
    """
    print(f"Creating dataset from {clean_dir} and {noisy_dir}")
    
    # Get all image files
    clean_files = get_image_files(clean_dir)
    noisy_files = get_image_files(noisy_dir)
    
    if not clean_files or not noisy_files:
        raise ValueError(f"No image files found in {clean_dir} or {noisy_dir}")
    
    print(f"Found {len(clean_files)} clean images and {len(noisy_files)} noisy images")
    
    # Match clean and noisy files by filename
    paired_files = []
    for clean_file in clean_files:
        clean_filename = os.path.basename(clean_file)
        clean_basename = os.path.splitext(clean_filename)[0]
        
        # Try to find matching noisy file(s)
        found_match = False
        for noisy_file in noisy_files:
            noisy_filename = os.path.basename(noisy_file)
            if clean_basename in noisy_filename:
                paired_files.append((clean_file, noisy_file))
                found_match = True
                break
        
        if not found_match:
            print(f"Warning: No noisy counterpart found for {clean_filename}")
    
    print(f"Found {len(paired_files)} valid image pairs")
    if not paired_files:
        raise ValueError(f"No valid image pairs found between {clean_dir} and {noisy_dir}")
    
    # Split the data for training and validation
    np.random.shuffle(paired_files)
    split_idx = int(len(paired_files) * (1 - validation_split))
    
    if is_training:
        selected_pairs = paired_files[:split_idx]  # Training set
    else:
        selected_pairs = paired_files[split_idx:]  # Validation set
    
    # Create tensors from selected pairs
    clean_files_tensor = tf.constant([p[0] for p in selected_pairs])
    noisy_files_tensor = tf.constant([p[1] for p in selected_pairs])
    
    # Create a dataset from the paired files
    dataset = tf.data.Dataset.from_tensor_slices((noisy_files_tensor, clean_files_tensor))
    
    # Function to load and preprocess a pair of images
    def load_and_preprocess_pair(noisy_path, clean_path):
        # Load and preprocess clean image (in color)
        clean_img = tf.io.read_file(clean_path)
        clean_img = tf.image.decode_image(clean_img, channels=3, expand_animations=False)
        clean_img = tf.image.convert_image_dtype(clean_img, tf.float32)
        clean_img = tf.image.resize(clean_img, [256, 256])
        
        # Load and preprocess noisy image (potentially grayscale)
        noisy_img = tf.io.read_file(noisy_path)
        noisy_img = tf.image.decode_image(noisy_img, channels=3, expand_animations=False)
        noisy_img = tf.image.convert_image_dtype(noisy_img, tf.float32)
        noisy_img = tf.image.resize(noisy_img, [256, 256])
        
        # Check if noisy image is grayscale and convert to 3-channel if needed
        # This ensures consistency even if some images are color and some grayscale
        noisy_shape = tf.shape(noisy_img)
        if noisy_shape[-1] == 1:
            # If grayscale, duplicate to 3 channels
            noisy_img = tf.repeat(noisy_img, 3, axis=-1)
        else:
            # If already color but grayscale content (all channels identical),
            # convert to proper grayscale then back to 3 channels
            is_grayscale = tf.math.reduce_all(tf.math.equal(
                noisy_img[..., 0], noisy_img[..., 1])) and tf.math.reduce_all(tf.math.equal(
                noisy_img[..., 0], noisy_img[..., 2]))
            
            if is_grayscale:
                # Convert to proper grayscale then back to 3 identical channels
                gray = tf.image.rgb_to_grayscale(noisy_img)
                noisy_img = tf.repeat(gray, 3, axis=-1)
        
        return noisy_img, clean_img
    
    # Map the loading function over the dataset
    dataset = dataset.map(load_and_preprocess_pair, num_parallel_calls=AUTOTUNE)
    
    # Add data augmentation if training
    if is_training:
        # Function to apply the same random transformations to both images
        def random_augmentation(noisy, clean):
            # Random horizontal flip
            if tf.random.uniform(()) > 0.5:
                noisy = tf.image.flip_left_right(noisy)
                clean = tf.image.flip_left_right(clean)
            
            # Random vertical flip
            if tf.random.uniform(()) > 0.5:
                noisy = tf.image.flip_up_down(noisy)
                clean = tf.image.flip_up_down(clean)
            
            # Random 90-degree rotations
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            noisy = tf.image.rot90(noisy, k=k)
            clean = tf.image.rot90(clean, k=k)
            
            return noisy, clean
        
        dataset = dataset.map(random_augmentation, num_parallel_calls=AUTOTUNE)
    
    # Finish dataset preparation
    dataset = dataset.shuffle(buffer_size=min(1000, len(selected_pairs))) if is_training else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

# Setup parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 30  # Number of epochs for fine-tuning
VALIDATION_SPLIT = 0.2  # 20% for validation

# Set up MirroredStrategy for multi-GPU training if available
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Everything should be in the strategy scope
with strategy.scope():
    # Load the previously trained model
    try:
        base_model = tf.keras.models.load_model(model_path)
        base_model.summary()
        print("Base model loaded successfully!")
        
        # Create a new model that enhances color restoration
        # Get the input layer from the base model
        input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        
        # Get the output from the base denoising model
        denoised_output = base_model(input_tensor)
        
        # Add a color restoration branch
        color_branch = Conv2D(64, (3, 3), activation='relu', padding='same')(denoised_output)
        color_branch = Conv2D(64, (3, 3), activation='relu', padding='same')(color_branch)
        color_branch = Conv2D(32, (3, 3), activation='relu', padding='same')(color_branch)
        
        # Combine the branches
        combined = Concatenate()([denoised_output, color_branch])
        
        # Final colorization layer
        final_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(combined)
        
        # Create the new combined model
        model = Model(inputs=input_tensor, outputs=final_output)
        
        # Preserve weights from the base model
        print("Created enhanced model with color restoration capability")
        model.summary()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct. If in Kaggle, upload your model as a dataset first.")
        exit(1)
    
    # Find the dataset paths
    dataset_paths = find_dataset_paths(dataset_path)
    
    # Set up paths for clean and noisy images
    clean_path = dataset_paths['clean']
    noisy_path = dataset_paths['noisy']
    
    if clean_path is None or noisy_path is None:
        print("Critical error: Could not find clean or noisy image folders")
        exit(1)
    
    # Create datasets
    try:
        print("Creating train dataset...")
        train_dataset = create_paired_dataset(
            clean_path, 
            noisy_path, 
            batch_size=BATCH_SIZE, 
            is_training=True,
            validation_split=VALIDATION_SPLIT
        )
        
        print("Creating validation dataset...")
        val_dataset = create_paired_dataset(
            clean_path, 
            noisy_path, 
            batch_size=BATCH_SIZE, 
            is_training=False,
            validation_split=VALIDATION_SPLIT
        )
        
        # Check if datasets are valid
        print("Checking datasets...")
        try:
            # Try to get one batch from each dataset to verify they work
            train_batch = next(iter(train_dataset))
            print(f"Train batch shapes: {train_batch[0].shape}, {train_batch[1].shape}")
            
            val_batch = next(iter(val_dataset))
            print(f"Validation batch shapes: {val_batch[0].shape}, {val_batch[1].shape}")
        except Exception as e:
            print(f"Error checking datasets: {e}")
            print("Will attempt to continue training anyway...")
    except Exception as e:
        print(f"Error creating datasets: {e}")
        raise e
    
    # Compile the model with only standard metrics
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss='mse',
        metrics=['mae', 'mse']
    )

# Setup file paths for outputs
best_model_path = os.path.join(output_dir, 'denoising_model_sp_best.keras')
final_model_path = os.path.join(output_dir, 'denoising_model_sp_final.keras')
savedmodel_dir = os.path.join(output_dir, 'denoising_model_sp_savedmodel')
history_plot_path = os.path.join(output_dir, 'sp_finetuning_history.png')
results_plot_path = os.path.join(output_dir, 'sp_finetuning_results.png')

# Setup callbacks
checkpoint = ModelCheckpoint(
    best_model_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Fine-tune the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks
)

# Save the final model
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")

# Safely plot training history with error checking
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig(history_plot_path)
print(f"Training history plot saved to: {history_plot_path}")

# Save the model in TensorFlow SavedModel format
try:
    # Create directory if it doesn't exist
    os.makedirs(savedmodel_dir, exist_ok=True)
    # Save model using tf.saved_model API
    tf.saved_model.save(model, savedmodel_dir)
    print(f"Model saved successfully in SavedModel format to: {savedmodel_dir}")
except Exception as e:
    print(f"Error using tf.saved_model.save(): {e}")
    # Fallback to standard Keras save with .keras extension
    fallback_path = savedmodel_dir + ".keras"
    model.save(fallback_path)
    print(f"Fallback: Model saved with .keras extension to: {fallback_path}")

# Test the model on a few examples
def plot_denoising_results(model, val_dataset, num_samples=5):
    plt.figure(figsize=(15, 5 * num_samples))
    batch = next(iter(val_dataset))
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
        plt.title('Noisy (Salt & Pepper)')
        plt.axis('off')
        
        # Denoised image
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(predictions[i])
        plt.title('Denoised')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(results_plot_path)

# Get a batch of validation data and visualize results
try:
    plot_denoising_results(model, val_dataset)
    print(f"Denoising results saved to: {results_plot_path}")
except Exception as e:
    print(f"Error visualizing results: {e}")

# Function to manually calculate PSNR
def calculate_psnr(y_true, y_pred, max_val=1.0):
    """Calculate PSNR between two images or batches of images"""
    # Convert tensors to numpy if needed
    if tf.is_tensor(y_true):
        y_true = y_true.numpy()
    if tf.is_tensor(y_pred):
        y_pred = y_pred.numpy()
    
    # Calculate MSE
    mse = np.mean(np.square(y_true - y_pred))
    # Avoid division by zero
    if mse == 0:
        return 100.0
    # Calculate and return PSNR
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

# Calculate and print metrics on validation set
print("\nCalculating metrics on validation set...")
evaluation = model.evaluate(val_dataset, verbose=1)
metrics_dict = dict(zip(model.metrics_names, evaluation))
print(f"Validation Loss (MSE): {metrics_dict['loss']:.6f}")
print(f"Validation MAE: {metrics_dict['mae']:.6f}")

# Manually calculate PSNR on a validation batch
print("\nCalculating PSNR on a sample validation batch...")
try:
    val_batch = next(iter(val_dataset))
    noisy_images, clean_images = val_batch
    predictions = model.predict(noisy_images)
    
    # Calculate PSNR for each image and average
    psnr_values = []
    for i in range(len(predictions)):
        psnr = calculate_psnr(clean_images[i], predictions[i])
        psnr_values.append(psnr)
    
    avg_psnr = np.mean(psnr_values)
    print(f"Average PSNR on sample batch: {avg_psnr:.2f} dB")
except Exception as e:
    print(f"Error calculating PSNR: {e}")

# Print final message
print("\nFine-tuning complete!")
print("Models saved as:")
print(f"  - {best_model_path} (Best weights)")
print(f"  - {final_model_path} (Final weights)")
print(f"  - {savedmodel_dir} (TensorFlow SavedModel format)")
print("\nModel Capabilities:")
print("This model has been trained to simultaneously denoise salt-and-pepper noise AND restore color.")
print("It expects grayscale noisy images as input and produces color denoised images as output.")
print("\nTo load the model in your local environment with TensorFlow:")
print("import tensorflow as tf")
print(f"model = tf.keras.models.load_model('{best_model_path}')")

# Save a small metadata file with training information
metadata = {
    "dataset": "Salt and Pepper Noise Images",
    "base_model": model_path,
    "image_size": f"{IMG_HEIGHT}x{IMG_WIDTH}",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "initial_learning_rate": "5e-5",
    "validation_split": VALIDATION_SPLIT,
    "model_type": "Denoising with Color Restoration",
    "color_channels": "RGB (3 channels)"
}

with open(os.path.join(output_dir, "sp_finetuning_metadata.txt"), "w") as f:
    for key, value in metadata.items():
        f.write(f"{key}: {value}\n") 