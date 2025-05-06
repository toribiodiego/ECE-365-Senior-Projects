# Face Recognition Robustness Analysis Tool

This application allows you to compare the performance of different face recognition models using different images of the same person.

## Features

- Compare reference face images with different images of the same person
- Test multiple face recognition models simultaneously
- Support for EdgeFace variants and ResNet18 models
- Face alignment for better recognition accuracy
- Simple web interface with Gradio

## Models Supported

### EdgeFace Models
- **edgeface_base**: Base EdgeFace model (larger size, higher accuracy)
- **edgeface_s_gamma_05**: Small EdgeFace model with gamma 0.5
- **edgeface_xs_gamma_06**: Extra small EdgeFace model with gamma 0.6
- **edgeface_xs_q**: Quantized extra small EdgeFace model (CPU only)
- **edgeface_xxs**: Extra-extra small EdgeFace model (optimized for edge devices)
- **edgeface_xxs_q**: Quantized extra-extra small EdgeFace model (CPU only, fastest)

### ResNet18 Models
- **resnet18_gs**: ResNet18 trained on grayscale images
- **resnet18_color**: ResNet18 trained on color images
- **resnet18_se_gs**: ResNet18 with Squeeze-Excitation blocks, trained on grayscale images
- **resnet18_se_color**: ResNet18 with Squeeze-Excitation blocks, trained on color images

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- OpenCV
- Gradio

### Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   **Important**: Make sure all dependencies are installed properly before running the application. You can verify this by running:
   ```
   python test_app.py
   ```
   
5. Configure environment variables (copy `config/env.example` to `.env`)

### Running the Application

Use the provided `run.sh` script:

```bash
./run.sh
```

Or run the app directly:

```bash
python app.py [--share] [--port PORT] [--preload]
```

Options:
- `--share`: Create a shareable link
- `--port PORT`: Specify the port to run on (default: 7860)
- `--preload`: Preload all models at startup (reduces initial processing time)

## Usage

1. Upload a reference face image
2. Upload a comparison image of the same person
3. Select which models to test
4. Click "Compare Images" to analyze model performance

## Real-world Testing

For best results, we recommend testing with:
- Different images of the same person in various lighting conditions
- Images with and without sunglasses or other accessories
- Different poses and expressions

## Project Structure

- `app.py`: Main application with Gradio interface
- `find_similar_enhanced.py`: Core face recognition functionality
- `face_align.py`: Face detection and alignment
- `main.py`: Model definitions and loading
- `models/`: Directory for downloaded model files
- `embeddings_cache/`: Directory for cached embeddings
- `aligned_faces/`: Directory for storing aligned face images

## Environment Variables

See `config/env.example` for required environment variables.

## License

[MIT License](LICENSE)

## Team Members
- Diego Toribio
- Nicholas Storniolo

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/face-recognition-project.git
cd face-recognition-project
```

2. Run the setup script to download necessary datasets and models:
```bash
bash setup.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

## Usage

### Running the Application

To start the face recognition application:
```bash
bash run.sh
```
or directly:
```bash
python app.py
```

The web interface will be available at http://localhost:7860.

### Features
- Compare multiple face recognition models, including EdgeFace and ResNet18 variants
- Visualize similarity scores across different models
- Automatic face alignment and preprocessing

### Core Modules
- `app.py` - Main Gradio web interface
- `main.py` - Core evaluation framework
- `face_align.py` - Face detection and alignment utilities
- `find_similar_enhanced.py` - Face similarity and retrieval functionality

## Benchmark

This project evaluates the following models:
- EdgeFace variants (base, xs, xxs, quantized versions)
- ResNet18 variants (with/without Squeeze-Excitation, grayscale/color)

We measure:
- Accuracy and performance across different images
- Processing time on edge hardware
- Memory consumption
- Energy usage

## Folder Structure
```
.
├── aligned_faces/         # Storage for aligned faces (generated)
├── app.py                 # Main GUI application
├── assets/                # Static assets for the application
├── embeddings_cache/      # Cache for face embeddings (generated)
├── face_align.py          # Face alignment utilities
├── face_align_integration.py # Integration helper
├── find_similar_enhanced.py  # Enhanced similarity functions
├── generate_all_embeddings.py # Utility for preprocessing
├── hubconf.py             # Model hub configuration
├── main.py                # Main evaluation script
├── patched_hubconf.py     # Patched hub integration
├── requirements.txt       # Python dependencies
├── run.sh                 # Runner script
├── setup.sh               # Installation script
└── test_*.py              # Various test modules
```

## License
[MIT License](LICENSE)

## Acknowledgments
- ResNetFace implementation based on work from [otroshi](https://github.com/otroshi/edgeface)
- Thanks to Professor Sam Keene for guidance throughout the project

### Objective  
Measure how precision reduction influences (1) face-ID accuracy, (2) latency and memory footprint, and (3) saliency-based explainability. By systematically comparing full- and low-bit-width models—and visualizing which facial regions remain salient after quantization—we aim to publish a validated recipe for fast, privacy-preserving face recognition on resource-constrained devices.


### Exhibition



<br>

### Approach



<br>


### Directory Structure
```
.
├── README.md
├── app.py
├── hubconf.py
├── main.py
├── requirements.txt
├── run.sh
└── setup.sh
```
<br>

### Results



<br>

### On Replication



<br>

### References



<br>

# Face Recognition Application

This project demonstrates face recognition with different models and evaluates their performance with different images of the same person.

## Features

- Multiple face recognition models (ResNet18 variants, EdgeFace)
- Interactive webcam capture with face detection and alignment
- Comparison of multiple models on the same image
- Background masking to focus only on facial features

## Setup

1. Clone this repository
2. Run `setup.sh` to download the aligned dataset and models
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

```bash
python app.py
```

This will start a Gradio web interface where you can:
1. Take a photo with your webcam or upload an image
2. Upload a comparison image of the same person
3. Compare recognition performance across all models

## Face Alignment

All input images (from webcam or file upload) are automatically processed with face alignment:

1. Face detection using OpenCV
2. Face cropping with margin
3. Resizing to the model's required input size

This ensures consistency between live captures and the aligned dataset images.

## Command Line Usage

For batch testing, you can use the command line interface:

```bash
python find_similar_enhanced.py path/to/reference/image.jpg path/to/query/image.jpg --model resnet18_gs
```

Options:
- `--model`: Select the face recognition model
- `--list_models`: Show available models

## Available Models

- `edgeface_xxs`: Lightweight model optimized for edge devices
- `resnet18_gs`: ResNet18 trained on grayscale images
- `resnet18_color`: ResNet18 trained on color images
- `resnet18_se_gs`: ResNet18 with Squeeze-Excitation blocks (grayscale)
- `resnet18_se_color`: ResNet18 with Squeeze-Excitation blocks (color)