# Image Noise Removal System

A deep learning-based system for removing noise from images using an autoencoder neural network architecture.

## Features

- Remove multiple types of noise from images (Gaussian, salt and pepper, speckle)
- Web interface for easy image upload and processing
- Command-line interface for batch processing
- Training functionality to create custom models

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
python main.py train --data_dir path/to/clean/images --model_dir path/to/save/model
```

Optional arguments:
- `--epochs`: Number of training epochs (overrides config)

#### Denoising Images

To denoise a single image:

```
python main.py denoise --image path/to/noisy/image.jpg --model path/to/model.h5 --output path/to/save/denoised.jpg
```

## Model Architecture

The system uses a convolutional autoencoder architecture:

1. **Encoder**: Reduces the dimensionality of the image while preserving essential features
2. **Bottleneck**: The compressed representation of the image
3. **Decoder**: Reconstructs the image from the compressed representation, removing noise in the process

## Configuration

System parameters can be modified in the `config.py` file:

- Model architecture parameters
- Training parameters
- Data processing settings
- Noise generation parameters

## Directory Structure

```
Image-Noise-Removal-System/
├── app.py                  # Web application
├── config.py               # Configuration parameters
├── main.py                 # Command-line interface
├── model/                  # Model architecture and training
│   ├── __init__.py
│   ├── autoencoder.py      # Autoencoder model implementation
│   └── trainer.py          # Model training utilities
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   └── preprocessor.py     # Image preprocessing utilities
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