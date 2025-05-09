# Image Noise Removal System

A machine learning-based system for removing noise from images using a U-Net architecture trained on the Berkeley Segmentation Dataset (BSDS500).

## Features

- **U-Net Denoising Model**: Uses a specialized U-Net architecture to effectively remove noise while preserving image details.
- **Resolution Independence**: Handles images of any size through an intelligent patch-based processing system.
- **Web Interface**: Easy-to-use web application for quick image denoising.
- **Command-line Interface**: Simple CLI for batch processing of images.

## Installation

### Prerequisites
- Python 3.10+
- TensorFlow 2.18+
- Flask

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/Image-Noise-Removal-System.git
cd Image-Noise-Removal-System
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

## Usage

### Web Interface
1. Open a web browser and navigate to `http://127.0.0.1:5000/`
2. Upload an image through the interface
3. View and download the denoised result

### Command-line Interface
```bash
# Denoise a single image
python main.py denoise path/to/your/image.jpg

# Specify output path
python main.py denoise path/to/your/image.jpg --output path/to/save/denoised.jpg
```

## Technical Details

### Model Architecture
The system uses a specialized U-Net denoising model implemented in the `model/autoencoder.py` file with the following components:
- **Encoder**: Convolutional blocks with max pooling for feature extraction
- **Bridge**: Convolutional layers
- **Decoder**: Upsampling blocks with skip connections to preserve details
- **Output Layer**: Convolutional layer with appropriate activation

### Resolution Adapter
- Handles images of any size by dividing large images into overlapping patches
- Processes each patch with the fixed-size model
- Seamlessly combines processed patches with weighted blending to avoid edge artifacts
- Implemented in `utils/resolution_adapter.py`

## Training Details

### Dataset
- **Training Data**: Berkeley Segmentation Dataset (BSDS500)
- **Noise Generation**: Synthetic Gaussian noise applied to clean images
- **Data Augmentation**: Data augmentation techniques to increase training data variety

### Training Strategy
- **Loss Function**: Implemented in `model/losses.py`
- **Training Scripts**: Available in `train_denoising_model.py` and `finetune_denoising_model.py`
- **Fine-tuning Capabilities**: Salt and pepper noise model fine-tuning available in `finetune_salt_pepper_model.py`

## Directory Structure

```
Image-Noise-Removal-System/
├── app.py                      # Web application
├── config.py                   # Configuration parameters
├── main.py                     # Command-line interface
├── train_denoising_model.py    # Model training script
├── finetune_denoising_model.py # Model fine-tuning script
├── finetune_salt_pepper_model.py # Salt and pepper noise fine-tuning
├── utils/                      # Utility functions
│   ├── resolution_adapter.py   # Handles images of varying sizes
│   ├── data_loader.py          # Data loading utilities
│   ├── visualizer.py           # Visualization tools
│   ├── enhanced_denoiser.py    # Enhanced denoising functionality
│   └── preprocessor.py         # Image preprocessing tools
├── model/                      # Model architecture and training
│   ├── autoencoder.py          # U-Net model implementation
│   ├── losses.py               # Loss functions
│   ├── trainer.py              # Training utilities
│   └── __init__.py             # Module initialization
├── saved_models/               # Stored model files
│   └── denoising_model_best.keras  # Main denoising model
├── static/                     # Static files for web app
│   ├── uploads/                # Temporary storage for uploaded images
│   └── results/                # Storage for processed images
├── templates/                  # HTML templates for web app
│   ├── index.html              # Main web interface
│   └── result.html             # Result display page
├── install_dependencies.py     # Helper script for dependencies
├── run.bat                     # Batch file for Windows execution
└── requirements.txt            # Required packages
```

## System Requirements

- **Python**: 3.10+
- **TensorFlow**: 2.18/2.19
- **Memory**: 4GB+ recommended
- **Disk Space**: 500MB for model and dependencies
- **GPU**: Optional but recommended for faster processing

## Future Improvements
- Add batch processing for multiple images in the web interface
- Implement image enhancement options beyond denoising
- Add fine-grained noise level control
- Create specialized models for different noise types
- Add progress indicators for large image processing

## Acknowledgments

- TensorFlow team for their deep learning framework
- Berkeley Vision and Learning Center for the BSDS500 dataset
- Python Imaging Library (PIL) contributors 