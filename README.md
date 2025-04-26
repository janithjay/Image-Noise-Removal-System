# Image Noise Removal System

A deep learning-based system for removing noise from images using an autoencoder neural network architecture.

## Features

- Remove multiple types of noise from images (Gaussian, salt and pepper, speckle, Poisson)
- Web interface for easy image upload and processing
- Command-line interface for batch processing
- Training functionality to create custom models
- Fine-tuning capabilities to improve model performance on specific types of images or noise

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Image-Noise-Removal-System.git
   cd Image-Noise-Removal-System
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

1. Start the web application:
   ```
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000`

3. Upload a noisy image through the web interface and view the denoised result

### Command Line Interface

#### Training a Model

To train a new model on your dataset:

```
python train_model.py
```

Optional arguments:
- `--epochs`: Number of training epochs (overrides config)

#### Fine-tuning a Model

To fine-tune an existing model for improved performance:

```
python fine_tune.py --model-path models/best_model.h5
```

Optional arguments:
- `--learning-rate`: Learning rate for fine-tuning (default: 0.0001)
- `--epochs`: Number of epochs for fine-tuning (default: 20)
- `--batch-size`: Batch size for training (default: 4)
- `--noise-types`: Comma-separated list of noise types to use (default: gaussian)
  - Options: gaussian, poisson, salt_pepper, speckle, all
- `--augmentation`: Enable data augmentation for training
- `--visualize-layers`: Visualize model layers and feature maps
- `--results-dir`: Directory to save results (default: results/)
- `--image-dir`: Directory containing clean images for training (default: clean_images/)

Example of fine-tuning with multiple noise types and data augmentation:
```
python fine_tune.py --model-path models/best_model.h5 --learning-rate 0.0001 --epochs 30 --noise-types gaussian,salt_pepper --augmentation
```

#### Denoising Images

To denoise a single image:

```
python main.py denoise --image path/to/noisy/image.jpg --model path/to/model.h5 --output path/to/save/denoised.jpg
```

## Model Architecture

The system uses a convolutional autoencoder architecture with advanced features:

1. **Encoder**: Reduces the dimensionality of the image while preserving essential features
2. **Bottleneck**: The compressed representation of the image
3. **Decoder**: Reconstructs the image from the compressed representation, removing noise
4. **Skip Connections**: U-Net style connections to preserve spatial information
5. **Attention Mechanism**: Helps the model focus on important features
6. **Residual Blocks**: Improves gradient flow and feature preservation

## Configuration

System parameters can be modified in the `config.py` file:

- Model architecture parameters
- Training parameters
- Data processing settings
- Noise generation parameters

## Advanced Denoising Techniques

This system implements several advanced techniques to improve denoising quality:

1. **Attention Mechanism**: Uses CBAM (Convolutional Block Attention Module) to focus on important features
2. **Residual Learning**: Residual connections make it easier to learn the noise pattern
3. **Multi-scale Processing**: Different filter sizes capture features at various scales
4. **Combined Loss Function**: Uses a weighted combination of MSE, SSIM, and perceptual loss

## Fine-tuning Process

The fine-tuning process helps adapt a pre-trained model to specific types of images or noise:

1. Start with a pre-trained model
2. Use a lower learning rate (typically 10% of the original)
3. Train on your specific dataset with the types of noise you want to target
4. Optionally use data augmentation to improve generalization
5. Monitor performance using PSNR, SSIM, and other quality metrics
6. Save the fine-tuned model for future use

## Directory Structure

```
Image-Noise-Removal-System/
├── app.py                  # Web application
├── config.py               # Configuration parameters
├── main.py                 # Command-line interface
├── train_model.py          # Model training script
├── fine_tune.py            # Model fine-tuning script
├── model/                  # Model architecture and training
│   ├── __init__.py
│   ├── autoencoder.py      # Autoencoder model implementation
│   ├── trainer.py          # Model training utilities
│   └── losses.py           # Custom loss functions
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessor.py     # Image preprocessing utilities
│   └── visualizer.py       # Visualization utilities
├── static/                 # Static files for web app
├── templates/              # HTML templates for web app
├── requirements.txt        # Required packages
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for their deep learning framework
- The scikit-image project for image processing utilities
- OpenCV contributors for computer vision tools 