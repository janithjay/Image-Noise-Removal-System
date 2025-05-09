# Image Denoising Model Training and Fine-tuning

This document provides instructions for training and fine-tuning the U-Net denoising model used in the Image Noise Removal System.

## Environment Setup

The model is designed to be trained in a Python environment with the following specifications:
- Python 3.10+
- TensorFlow 2.18+
- GPU acceleration recommended

## Initial Training with BSDS500

### Dataset
The base model is trained on the Berkeley Segmentation Dataset (BSDS500) with synthetic noise using the `train_denoising_model.py` script.

### Training Steps:
1. Ensure you have the proper environment set up
2. Prepare the BSDS500 dataset
3. Run the training script:
```python
python train_denoising_model.py
```

### Training Outputs
After training, the following files will be generated:
- `denoising_model_best.keras`: Model with the best validation loss
- `denoising_model_final.keras`: Model after the last training epoch

## Fine-tuning Options

The system provides two fine-tuning options:

### 1. General Denoising Fine-tuning

For improving the model's performance on general noise patterns using the `finetune_denoising_model.py` script.

#### Steps:
1. Ensure you have a pre-trained model (`denoising_model_best.keras`) in the `saved_models` directory
2. Run the fine-tuning script:
```python
python finetune_denoising_model.py
```

### 2. Salt and Pepper Noise Fine-tuning

For specialized salt and pepper noise removal using the `finetune_salt_pepper_model.py` script.

#### Steps:
1. Ensure you have a pre-trained model (`denoising_model_best.keras`) in the `saved_models` directory
2. Run the salt and pepper fine-tuning script:
```python
python finetune_salt_pepper_model.py
```

### Fine-tuning Outputs
After fine-tuning, the following files will be generated:
- `denoising_model_finetuned_best.keras`: Fine-tuned model with best validation loss
- `denoising_model_finetuned_final.keras`: Fine-tuned model after the last epoch
- For salt and pepper noise fine-tuning, outputs will include `saltpepper_model_best.keras`

## Using the Trained Model

To use the trained model in your local environment:

```python
import tensorflow as tf
from utils.resolution_adapter import ResolutionAdapter
from PIL import Image

# Load the model
model = tf.keras.models.load_model('saved_models/denoising_model_best.keras')

# Create a model wrapper
class ModelWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, input_data):
        return self.model.predict(input_data)

# Create resolution adapter for handling images of any size
adapter = ResolutionAdapter(ModelWrapper(model), target_size=None, overlap=0.25)

# Load and process an image
img = Image.open('noisy_image.jpg')
denoised_img = adapter.process_image(img)
denoised_img.save('denoised_image.jpg')
```

## Model Architecture

The model uses a U-Net architecture implemented in `model/autoencoder.py` that consists of:
- **Encoder**: Convolutional layers with max pooling
- **Bridge**: Convolutional layers
- **Decoder**: Upsampling, concatenation, and convolutional layers
- **Output**: Convolutional layer with appropriate activation

## Training Parameters

- **Batch Size**: Configurable in training scripts
- **Learning Rate**: Configurable with reduction on plateau
- **Loss Function**: Defined in `model/losses.py`
- **Early Stopping**: Yes, monitoring validation loss
- **Learning Rate Reduction**: Yes, when validation loss plateaus 