import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub
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

# Check if running in Kaggle environment
IN_KAGGLE = os.path.exists('/kaggle/input')
print(f"Running in Kaggle environment: {IN_KAGGLE}")

# Set paths based on environment
if IN_KAGGLE:
    # Update this path to your uploaded model in Kaggle
    model_path = '/kaggle/input/base2/keras/default/1/denoising_model_best.keras'
    output_dir = '/kaggle/working'
else:
    model_path = 'saved_models/denoising_model_best.keras'
    output_dir = 'saved_models'

os.makedirs(output_dir, exist_ok=True)
print(f"Model will be loaded from: {model_path}")
print(f"Outputs will be saved to: {output_dir}")

# Function to find correct dataset paths by recursively searching directories
def find_dataset_paths(base_path):
    """
    Find the dataset paths for clean and noisy images by recursively searching.
    
    Args:
        base_path: Base path of the downloaded dataset
        
    Returns:
        Dictionary with paths for train, val, and test sets
    """
    print(f"Searching for dataset structure in: {base_path}")
    
    # Initialize paths
    paths = {
        'train_clean': None,
        'train_noisy': None,
        'val_clean': None,
        'val_noisy': None,
        'test_clean': None,
        'test_noisy': None
    }
    
    # Helper function to recursively find folders
    def find_folders(current_path, depth=0, max_depth=5):
        if depth > max_depth:
            return
            
        try:
            contents = os.listdir(current_path)
            
            # Check for ground truth and noisy images folders
            if 'ground truth' in contents and 'noisy images' in contents:
                # Determine which split this is (train, val, test)
                path_parts = current_path.split(os.sep)
                if 'train' in path_parts:
                    paths['train_clean'] = os.path.join(current_path, 'ground truth')
                    paths['train_noisy'] = os.path.join(current_path, 'noisy images')
                    print(f"Found train paths: {paths['train_clean']}")
                elif 'validate' in path_parts or 'val' in path_parts:
                    paths['val_clean'] = os.path.join(current_path, 'ground truth')
                    paths['val_noisy'] = os.path.join(current_path, 'noisy images') 
                    print(f"Found validation paths: {paths['val_clean']}")
                elif 'test' in path_parts:
                    paths['test_clean'] = os.path.join(current_path, 'ground truth')
                    paths['test_noisy'] = os.path.join(current_path, 'noisy images')
                    print(f"Found test paths: {paths['test_clean']}")
            
            # Recursively search subdirectories
            for item in contents:
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path):
                    find_folders(item_path, depth + 1, max_depth)
        except Exception as e:
            print(f"Error searching directory {current_path}: {e}")
    
    # Start search
    find_folders(base_path)
    
    # Check if we found all paths
    missing_paths = [k for k, v in paths.items() if v is None]
    if missing_paths:
        print(f"Warning: Could not find paths for: {', '.join(missing_paths)}")
        
    return paths

# Function to list image files
def get_image_files(directory):
    return glob.glob(os.path.join(directory, "*.jpg"))

# Function to create pairs of noisy and clean images
def create_paired_dataset(clean_dir, noisy_dir, batch_size=16, is_training=True):
    """
    Create a dataset of paired clean and noisy images.
    
    Args:
        clean_dir: Directory containing clean images
        noisy_dir: Directory containing noisy images
        batch_size: Batch size for the dataset
        is_training: Whether this is for training (enables data augmentation)
        
    Returns:
        TensorFlow dataset of (noisy_image, clean_image) pairs
    """
    print(f"Creating dataset from {clean_dir} and {noisy_dir}")
    
    # Get all clean image files
    clean_files = sorted(get_image_files(clean_dir))
    if not clean_files:
        raise ValueError(f"No image files found in {clean_dir}")
    
    # Create paired filenames first (Python operations, not TensorFlow)
    paired_files = []
    for clean_file in clean_files:
        clean_filename = os.path.basename(clean_file)
        found_noisy = False
        
        # Try to find matching noisy files
        for noise_type in ['gauss_', 'sp_', 'poisson_', 'speckle_']:
            noisy_filename = noise_type + clean_filename
            noisy_file = os.path.join(noisy_dir, noisy_filename)
            if os.path.exists(noisy_file):
                paired_files.append((clean_file, noisy_file))
                found_noisy = True
                break  # Just use the first found noise type
        
        if not found_noisy:
            print(f"Warning: No noisy counterpart found for {clean_filename}")
    
    print(f"Found {len(paired_files)} valid image pairs")
    if not paired_files:
        raise ValueError(f"No valid image pairs found between {clean_dir} and {noisy_dir}")
    
    # Now create a dataset from these pre-paired files
    clean_files_tensor = tf.constant([p[0] for p in paired_files])
    noisy_files_tensor = tf.constant([p[1] for p in paired_files])
    
    # Create a dataset from the paired files
    dataset = tf.data.Dataset.from_tensor_slices((noisy_files_tensor, clean_files_tensor))
    
    # Function to load and preprocess a pair of images
    def load_and_preprocess_pair(noisy_path, clean_path):
        # Load and preprocess clean image
        clean_img = tf.io.read_file(clean_path)
        clean_img = tf.image.decode_jpeg(clean_img, channels=3)
        clean_img = tf.image.convert_image_dtype(clean_img, tf.float32)
        clean_img = tf.image.resize(clean_img, [256, 256])
        
        # Load and preprocess noisy image
        noisy_img = tf.io.read_file(noisy_path)
        noisy_img = tf.image.decode_jpeg(noisy_img, channels=3)
        noisy_img = tf.image.convert_image_dtype(noisy_img, tf.float32)
        noisy_img = tf.image.resize(noisy_img, [256, 256])
        
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
            
            return noisy, clean
        
        dataset = dataset.map(random_augmentation, num_parallel_calls=AUTOTUNE)
    
    # Finish dataset preparation
    dataset = dataset.shuffle(buffer_size=min(1000, len(paired_files))) if is_training else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

# Setup parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 50  # Fewer epochs for fine-tuning

# Set up MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Everything should be in the strategy scope
with strategy.scope():
    # Load the previously trained model
    try:
        model = tf.keras.models.load_model(model_path)
        model.summary()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path is correct. If in Kaggle, upload your model as a dataset first.")
        exit(1)
    
    # Download the NISN dataset
    nisn_path = kagglehub.dataset_download('tarunpathak/natural-images-with-synthetic-noise')
    print(f"Dataset downloaded to: {nisn_path}")
    
    # Find the dataset paths
    dataset_paths = find_dataset_paths(nisn_path)
    
    # Set up paths for train, validation, and test datasets
    train_clean_path = dataset_paths['train_clean'] 
    train_noisy_path = dataset_paths['train_noisy']
    val_clean_path = dataset_paths['val_clean']
    val_noisy_path = dataset_paths['val_noisy']
    test_clean_path = dataset_paths['test_clean']
    test_noisy_path = dataset_paths['test_noisy']
    
    # Fall back to default paths if automatic search failed
    if train_clean_path is None:
        print("Falling back to default paths...")
        train_clean_path = os.path.join(nisn_path, 'train', 'train', 'ground truth')
        train_noisy_path = os.path.join(nisn_path, 'train', 'train', 'noisy images')
        val_clean_path = os.path.join(nisn_path, 'validate', 'validate', 'ground truth')
        val_noisy_path = os.path.join(nisn_path, 'validate', 'validate', 'noisy images')
        test_clean_path = os.path.join(nisn_path, 'test', 'test', 'ground truth')
        test_noisy_path = os.path.join(nisn_path, 'test', 'test', 'noisy images')
    
    # Create datasets
    try:
        print("Creating train dataset...")
        train_dataset = create_paired_dataset(train_clean_path, train_noisy_path, batch_size=BATCH_SIZE, is_training=True)
        print("Creating validation dataset...")
        val_dataset = create_paired_dataset(val_clean_path, val_noisy_path, batch_size=BATCH_SIZE, is_training=False)
        print("Creating test dataset...")
        test_dataset = create_paired_dataset(test_clean_path, test_noisy_path, batch_size=BATCH_SIZE, is_training=False)
        
        # Check if datasets are valid
        print("Checking datasets...")
        try:
            # Try to get one batch from each dataset to verify they work
            train_batch = next(iter(train_dataset))
            print(f"Train batch shapes: {train_batch[0].shape}, {train_batch[1].shape}")
            
            val_batch = next(iter(val_dataset))
            print(f"Validation batch shapes: {val_batch[0].shape}, {val_batch[1].shape}")
            
            test_batch = next(iter(test_dataset))
            print(f"Test batch shapes: {test_batch[0].shape}, {test_batch[1].shape}")
        except Exception as e:
            print(f"Error checking datasets: {e}")
            print("Will attempt to continue training anyway...")
    except Exception as e:
        print(f"Error creating datasets: {e}")
        raise e
    
    # Compile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=5e-5),  # Lower learning rate for fine-tuning
        loss='mse',
        metrics=['mae', 'mse']
    )

# Setup file paths for outputs
best_model_path = os.path.join(output_dir, 'denoising_model_finetuned_best.keras')
final_model_path = os.path.join(output_dir, 'denoising_model_finetuned_final.keras')
savedmodel_dir = os.path.join(output_dir, 'denoising_model_finetuned_savedmodel')
history_plot_path = os.path.join(output_dir, 'finetuning_history.png')
results_plot_path = os.path.join(output_dir, 'finetuning_denoising_results.png')

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
    patience=10,
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
    plt.savefig(results_plot_path)

# Get a batch of test data and visualize results
try:
    plot_denoising_results(model, test_dataset)
    print(f"Denoising results saved to: {results_plot_path}")
except Exception as e:
    print(f"Error visualizing results: {e}")

# Print final message
print("\nFine-tuning complete!")
print("Models saved as:")
print(f"  - {best_model_path} (Best weights)")
print(f"  - {final_model_path} (Final weights)")
print(f"  - {savedmodel_dir} (TensorFlow SavedModel format)")
print("\nTo load the model in your local environment with TensorFlow:")
print("import tensorflow as tf")
print(f"model = tf.keras.models.load_model('{best_model_path}')")

# Save a small metadata file with training information
metadata = {
    "dataset": "Natural Images with Synthetic Noise (NISN)",
    "base_model": model_path,
    "image_size": f"{IMG_HEIGHT}x{IMG_WIDTH}",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "initial_learning_rate": "5e-5",
    "training_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "test_images": len(test_dataset)
}

with open(os.path.join(output_dir, "finetuning_metadata.txt"), "w") as f:
    for key, value in metadata.items():
        f.write(f"{key}: {value}\n") 