# Trash Classification - Complete User Manual

This comprehensive manual covers everything you need to know about using the Trash Classification project.

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Getting Started](#getting-started)
5. [Training the Model](#training-the-model)
6. [Testing and Evaluation](#testing-and-evaluation)
7. [Making Predictions](#making-predictions)
8. [Web Application Guide](#web-application-guide)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)
12. [FAQ](#faq)

---

## Introduction

### What is Trash Classification?

Trash Classification is a deep learning project that automatically classifies waste items into four categories:
- **Plastic** - Plastic bottles, containers, and packaging
- **Glass** - Glass bottles, jars, and containers
- **Metal** - Metal cans, containers, and packaging
- **Others** - Paper, cardboard, organic waste, batteries, clothing, etc.

### Use Cases

- **Smart Waste Sorting Bins**: Automatically sort trash in smart bins
- **Recycling Centers**: Assist workers in categorizing waste
- **Educational Tools**: Teach waste management and recycling
- **Mobile Apps**: Help users identify recyclable materials
- **Research**: Study waste composition and recycling patterns

### Technology Stack

- **Framework**: PyTorch 2.0+
- **Model**: ConvNeXt Tiny (pretrained on ImageNet-1K)
- **Web Framework**: Flask
- **Deployment**: Python, Gunicorn, Docker

---

## System Requirements

### Minimum Requirements

- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 5 GB free space
- **Processor**: Modern CPU with 4+ cores

### Recommended for Training (GPU)

- **GPU**: NVIDIA GPU with 4 GB+ VRAM
  - RTX 3090, 3080 (Excellent)
  - RTX 3060, 3050 (Good)
  - GTX 1660, 1650 (Acceptable)
- **CUDA**: 11.8 or higher
- **VRAM**: 6 GB+ recommended

### GPU Compatibility

Check if your system has CUDA support:

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
```

If `CUDA Available: True`, your GPU can be used for faster training.

---

## Installation

### Option 1: Using uv (Recommended)

```bash
# 1. Install uv (if not already installed)
pip install uv

# 2. Create virtual environment
uv venv

# 3. Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
```

### Option 2: Using pip

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
```

### Option 3: Using conda

```bash
# 1. Create conda environment
conda create -n trash_cls python=3.9

# 2. Activate environment
conda activate trash_cls

# 3. Install PyTorch (CUDA version if available)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 4. Install other dependencies
pip install scikit-learn matplotlib seaborn pillow tqdm flask

# 5. Verify installation
python -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
```

### Verifying Installation

Run the verification script:

```bash
python -c "
import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import flask

print('✓ All packages installed successfully!')
print(f'  PyTorch: {torch.__version__}')
print(f'  Torchvision: {torchvision.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## Getting Started

### Quick Start (5 Minutes)

If you already have a trained model (`checkpoints/best_model.pth`):

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Launch web application
python app.py

# 3. Open browser to http://localhost:5000
# 4. Upload or paste an image
# 5. View classification results
```

### First Time Setup (30-60 Minutes)

If you need to train the model from scratch:

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Download dataset (if not already downloaded)
python download.py

# 3. Train the model
python main.py train

# 4. Test the model
python main.py test

# 5. Launch web application
python app.py
```

---

## Training the Model

### Understanding Training

Training involves:
1. Loading the dataset
2. Splitting into train/validation/test sets
3. Training the model over multiple epochs
4. Validating after each epoch
5. Saving the best model based on validation accuracy
6. Early stopping if validation doesn't improve

### Basic Training

```bash
python main.py train
```

This will:
- Use default hyperparameters from `src/config.py`
- Train for up to 50 epochs
- Stop early if validation doesn't improve for 10 epochs
- Save the best model to `checkpoints/best_model.pth`

### Training Output

During training, you'll see:

```
============================================================
TRAINING MODE
============================================================
Model: ConvNeXt Tiny
Classes: ['plastic', 'glass', 'metal', 'others']
Device: cuda
Batch Size: 32
Learning Rate: 0.0003
Epochs: 50
============================================================

Epoch 1/50
----------
Train: 100%|████████████████████| 120/120 [02:15<00:00]
Train Loss: 0.8234, Train Acc: 0.6875

Val: 100%|████████████████████| 26/26 [00:18<00:00]
Val Loss: 0.4567, Val Acc: 0.8234

✓ New best model saved! (Val Acc: 0.8234)

Epoch 2/50
----------
...
```

### Monitoring Training Progress

#### Using TensorBoard (Optional)

```python
# Add to src/train.py
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/trash_classifier')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
```

```bash
# View training in TensorBoard
tensorboard --logdir=runs
```

Open browser to `http://localhost:6006`

#### Viewing Training Curves

After training completes, view the training curves:

```bash
# Open the generated plot
open results/training_history.png  # Mac
xdg-open results/training_history.png  # Linux
start results/training_history.png  # Windows
```

### Custom Training Parameters

Edit `src/config.py` to customize training:

```python
# Training Configuration
BATCH_SIZE = 32          # Adjust based on GPU memory
LEARNING_RATE = 3e-4     # Lower for fine-tuning, higher for scratch
NUM_EPOCHS = 50          # Increase for better performance
WEIGHT_DECAY = 0.05      # Regularization strength

# Early Stopping
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N epochs

# Data Augmentation
# Can be customized in src/dataset.py
```

### Resuming Training

To resume training from a checkpoint:

```bash
# Modify src/train.py or use:
python src/train.py --resume checkpoints/best_model.pth
```

### Training on Different Datasets

1. Prepare your dataset with this structure:
```
your_dataset/
├── plastic/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── glass/
│   ├── image1.jpg
│   └── ...
├── metal/
│   └── ...
└── others/
    └── ...
```

2. Update `src/config.py`:
```python
DATA_DIR = 'path/to/your_dataset'
CLASS_NAMES = ['plastic', 'glass', 'metal', 'others']
NUM_CLASSES = len(CLASS_NAMES)
```

3. Train:
```bash
python main.py train
```

---

## Testing and Evaluation

### Running Tests

```bash
# Test with default checkpoint
python main.py test

# Test with specific checkpoint
python main.py test --checkpoint checkpoints/best_model.pth
```

### Understanding Test Output

```
============================================================
TESTING MODE
============================================================
Checkpoint: checkpoints/best_model.pth
============================================================

Loading model from checkpoints/best_model.pth...

Test Results:
------------
Accuracy: 0.9456
Precision: 0.9434
Recall: 0.9423
F1-Score: 0.9428

Per-Class Metrics:
---------------
Plastic:  Precision=0.9523, Recall=0.9434, F1=0.9478
Glass:    Precision=0.9234, Recall=0.9456, F1=0.9343
Metal:    Precision=0.9456, Recall=0.9234, F1=0.9343
Others:   Precision=0.9523, Recall=0.9567, F1=0.9545

Confusion matrix saved to results/confusion_matrix.png
Test metrics saved to results/test_metrics.json
```

### Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: How many selected items are relevant (TP / (TP + FP))
- **Recall**: How many relevant items are selected (TP / (TP + FN))
- **F1-Score**: Harmonic mean of precision and recall

### Viewing Results

```bash
# View confusion matrix
open results/confusion_matrix.png

# View detailed metrics (JSON)
cat results/test_metrics.json
```

### Interpreting the Confusion Matrix

The confusion matrix shows:
- Diagonal elements: Correct predictions
- Off-diagonal elements: Misclassifications
- Darker colors = higher values

Example analysis:
- If plastic is often confused with glass → Improve image quality or add more training data
- If "others" has low recall → Consider adding more subcategories

---

## Making Predictions

### Command Line Predictions

#### Single Image

```bash
python main.py predict --image path/to/image.jpg
```

Output:
```
============================================================
PREDICTION MODE
============================================================
Image: path/to/image.jpg

Predicted Class: plastic
Confidence: 95.23%

Top 3 Predictions:
1. plastic: 95.23%
2. glass: 2.41%
3. metal: 1.56%
```

#### Multiple Images

```bash
# Predict on all images in a directory
python main.py predict --image path/to/images/

# Predict on all images in current directory
python main.py predict --image .
```

### Python API Predictions

```python
import torch
from PIL import Image
from src.model import create_model
from src.config import Config
from torchvision import transforms

# Load model
device = torch.device(Config.DEVICE)
model = create_model(pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Prepare transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(Config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('test_image.jpg')
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

class_name = Config.CLASS_NAMES[predicted.item()]
print(f"Predicted: {class_name}")
print(f"Confidence: {confidence.item():.2%}")
```

### Batch Predictions

```python
import os
from pathlib import Path

# Predict on all images in a directory
image_dir = Path('test_images')
for img_path in image_dir.glob('*.jpg'):
    result = predict_single_image(img_path)  # Use function from above
    print(f"{img_path.name}: {result['class']} ({result['confidence']:.2%})")
```

---

## Web Application Guide

### Launching the Web Application

```bash
python app.py
```

Output:
```
Loading model...
Model loaded on cuda
 * Running on http://0.0.0.0:5000
```

Open browser to: `http://localhost:5000`

### Using the Web Interface

#### Method 1: Drag and Drop

1. Drag an image file onto the upload area
2. Release to upload
3. Click "Classify Image"
4. View results

#### Method 2: File Browser

1. Click "Choose Image" button
2. Select image from file dialog
3. Click "Classify Image"
4. View results

#### Method 3: Paste from Clipboard

1. Copy an image (Ctrl+C or Cmd+C)
2. Paste on the page (Ctrl+V or Cmd+V)
3. Click "Classify Image"
4. View results

### Understanding Results

The results page shows:

1. **Main Prediction** (Large, colored box)
   - Predicted class name
   - Confidence percentage

2. **All Predictions** (Detailed list)
   - Bar chart showing probability for each class
   - Sorted from highest to lowest
   - Highlighted winner class

3. **Probability Interpretation**
   - >90%: Very confident
   - 70-90%: Confident
   - 50-70%: Less confident
   - <50%: Uncertain

### Web Application Features

#### Responsive Design

Works on:
- Desktop computers
- Tablets
- Mobile phones

#### Image Preview

Shows the uploaded image before classification for verification.

#### Clear Function

Remove current image and start over without reloading.

#### Error Handling

Clear error messages for:
- Invalid file types
- Corrupted images
- Server errors

### Web Application Configuration

Edit `app.py` to customize:

```python
if __name__ == '__main__':
    load_model()
    
    # Custom configuration
    app.run(
        host='0.0.0.0',    # Listen on all interfaces
        port=5000,         # Change port if 5000 is in use
        debug=False,       # Disable for production
        threaded=True      # Enable concurrent requests
    )
```

### Accessing from Other Devices

To access from other devices on your network:

```bash
# Find your IP address
# Linux/Mac:
ifconfig | grep "inet "
# Windows:
ipconfig

# Run app (already configured for all interfaces)
python app.py

# Access from other device:
# http://YOUR_IP:5000
```

---

## Configuration

### Configuration File

All settings are in `src/config.py`:

```python
# Data Configuration
DATA_DIR = 'data/garbage-classification-v2'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_DIR = 'results'

# Model Configuration
IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ['plastic', 'glass', 'metal', 'others']

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.05

# Early Stopping
EARLY_STOPPING_PATIENCE = 10

# Data Split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# System Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

# Data Loading
NUM_WORKERS = 4
PIN_MEMORY = True
```

### Device Configuration

#### Force CPU Usage

```python
# In src/config.py:
DEVICE = 'cpu'
```

#### Force Specific GPU

```python
# In src/config.py:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
DEVICE = 'cuda'
```

### Performance Tuning

#### For Faster Training

```python
# Increase batch size (if GPU memory allows)
BATCH_SIZE = 64

# Increase workers (if CPU allows)
NUM_WORKERS = 8

# Enable mixed precision (requires PyTorch AMP)
# Add to src/train.py
scaler = torch.cuda.amp.GradScaler()
```

#### For Better Accuracy

```python
# Increase epochs
NUM_EPOCHS = 100

# Lower learning rate for fine-tuning
LEARNING_RATE = 1e-4

# Increase patience
EARLY_STOPPING_PATIENCE = 20

# Add more data augmentation in src/dataset.py
```

#### For Lower Memory Usage

```python
# Reduce batch size
BATCH_SIZE = 16

# Reduce workers
NUM_WORKERS = 2

# Disable pinned memory
PIN_MEMORY = False
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Out of Memory Error

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in `src/config.py`:
   ```python
   BATCH_SIZE = 16  # Try 8, 4, etc.
   ```

2. Reduce image size:
   ```python
   IMAGE_SIZE = 192  # Instead of 224
   ```

3. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. Use CPU instead:
   ```python
   DEVICE = 'cpu'
   ```

#### Issue: CUDA Not Available

**Symptom:**
```
CUDA Available: False
```

**Solutions:**
1. Install CUDA version of PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. Update NVIDIA drivers:
   - Download latest drivers from NVIDIA website
   - Install and restart

3. Check CUDA installation:
   ```bash
   nvcc --version
   nvidia-smi
   ```

#### Issue: Slow Training

**Symptom:** Training takes much longer than expected

**Solutions:**
1. Verify GPU is being used:
   ```python
   print(f"Device: {Config.DEVICE}")
   ```

2. Increase batch size:
   ```python
   BATCH_SIZE = 64  # If memory allows
   ```

3. Check CPU bottleneck:
   ```python
   NUM_WORKERS = 8  # Increase workers
   ```

4. Use mixed precision training (advanced)

#### Issue: Poor Accuracy

**Symptom:** Test accuracy is low (<80%)

**Solutions:**
1. Train for more epochs:
   ```python
   NUM_EPOCHS = 100
   ```

2. Lower learning rate:
   ```python
   LEARNING_RATE = 1e-4
   ```

3. Check data quality:
   - Ensure images are clear
   - Verify labels are correct
   - Balance class distribution

4. Increase data augmentation

#### Issue: Web Application Won't Start

**Symptom:** Port already in use error

**Solutions:**
1. Find process using port 5000:
   ```bash
   lsof -i :5000  # Linux/Mac
   netstat -ano | findstr :5000  # Windows
   ```

2. Kill the process or change port:
   ```python
   # In app.py:
   app.run(host='0.0.0.0', port=5001)
   ```

#### Issue: Model Not Found

**Symptom:**
```
Error: Checkpoint file not found: checkpoints/best_model.pth
```

**Solutions:**
1. Train the model first:
   ```bash
   python main.py train
   ```

2. Check checkpoint path:
   ```bash
   ls -la checkpoints/
   ```

3. Verify path in `src/config.py`:
   ```python
   CHECKPOINT_DIR = 'checkpoints'  # Ensure correct path
   ```

#### Issue: Dataset Not Found

**Symptom:** Dataset directory doesn't exist

**Solutions:**
1. Download dataset:
   ```bash
   python download.py
   ```

2. Manually download from Kaggle:
   - Go to: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
   - Download and extract to `data/garbage-classification-v2/`

3. Update path in `src/config.py`:
   ```python
   DATA_DIR = 'path/to/your/dataset'
   ```

### Getting Help

If issues persist:

1. Check logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Try with a minimal example
4. Consult the project's README.md and API.md
5. Search for similar issues on Stack Overflow or PyTorch forums

---

## Advanced Topics

### Transfer Learning

To adapt the model for your own classification task:

1. Prepare your dataset with proper structure
2. Update configuration:
   ```python
   CLASS_NAMES = ['class1', 'class2', 'class3']  # Your classes
   NUM_CLASSES = len(CLASS_NAMES)
   DATA_DIR = 'path/to/your/dataset'
   ```
3. Train:
   ```bash
   python main.py train
   ```

### Model Export

#### Export to ONNX

```python
import torch
from src.model import create_model
from src.config import Config

model = create_model(pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
torch.onnx.export(model, 
                  dummy_input, 
                  'trash_classifier.onnx',
                  input_names=['image'],
                  output_names=['class_probs'],
                  dynamic_axes={'image': {0: 'batch_size'},
                               'class_probs': {0: 'batch_size'}})
```

#### Export to TorchScript

```python
import torch
from src.model import create_model

model = create_model(pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Trace the model
example = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example)
traced_model.save('trash_classifier.pt')
```

### Model Quantization

Reduce model size and improve inference speed:

```python
import torch

model = create_model(pretrained=False)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Quantize to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'checkpoints/quantized_model.pth')
```

### Custom Data Augmentation

Add custom augmentation in `src/dataset.py`:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # Add rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                          saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),  # Add grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

### Batch Inference Script

Create a script to classify multiple images efficiently:

```python
import torch
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

def batch_inference(image_dir, output_file):
    """Classify all images in directory"""
    # Load model (as shown in Python API section)
    
    results = []
    image_paths = list(Path(image_dir).glob('*.jpg'))
    
    for img_path in tqdm(image_paths, desc="Classifying"):
        image = Image.open(img_path)
        # Process and predict
        result = predict_image(image)  # Your prediction function
        results.append({
            'image': str(img_path),
            'class': result['class'],
            'confidence': result['confidence']
        })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

# Usage
batch_inference('test_images/', 'results.json')
```

### Integration with Other Services

#### As a Microservice

Deploy as a REST API service:

```bash
# Using Docker
docker build -t trash-classifier .
docker run -p 5000:5000 trash-classifier

# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### With Nginx (Production)

Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

---

## FAQ

### General Questions

**Q: What image formats are supported?**
A: JPEG, PNG, BMP, GIF, WEBP, TIFF - most common image formats work.

**Q: Can I use this for real-time video classification?**
A: Yes! Extract frames and classify them. With GPU, you can achieve 50-100 FPS.

**Q: Can I add more classes?**
A: Yes. Prepare your dataset and update `CLASS_NAMES` and `NUM_CLASSES` in config.

**Q: How accurate is the model?**
A: On the test dataset, it achieves 92-96% accuracy. Real-world performance varies.

### Training Questions

**Q: How long does training take?**
A: With GPU: 30-60 minutes. With CPU: 2-4 hours.

**Q: Can I resume interrupted training?**
A: Yes. Modify the training script to load from a checkpoint.

**Q: What if I don't have a GPU?**
A: The model will automatically use CPU. Training will be slower but still works.

**Q: How can I improve accuracy?**
A: 
- Train for more epochs
- Use more data
- Improve data quality
- Try data augmentation
- Fine-tune hyperparameters

### Technical Questions

**Q: What does the confidence score mean?**
A: It's the model's certainty, ranging from 0 to 1 (0-100%). Higher = more confident.

**Q: Can I use this in a commercial application?**
A: Yes, but check the dataset license. The code is open-source.

**Q: How do I deploy this to the cloud?**
A: Use Docker, AWS, Google Cloud, or Azure. See API.md for deployment options.

**Q: Can I integrate this with a mobile app?**
A: Yes. Export to ONNX or TFLite and integrate with mobile frameworks.

### Performance Questions

**Q: How many images can I classify per second?**
A: With GPU: 50-100 images/second. With CPU: 10-20 images/second.

**Q: What's the maximum image size?**
A: No strict limit, but larger images will be resized to 224x224.

**Q: Does it work with low-resolution images?**
A: It works, but accuracy decreases for very low-resolution images (<64x64).

### Troubleshooting Questions

**Q: Why am I getting "CUDA out of memory"?**
A: Reduce `BATCH_SIZE` in config or use CPU mode.

**Q: The web app won't start. What should I do?**
A: Check if port 5000 is in use, or change the port in app.py.

**Q: Predictions are incorrect. What's wrong?**
A: Ensure you trained the model first. If still issues, check image quality and labels.

**Q: How do I check which device is being used?**
A: Check the training logs for "Device: cuda" or "Device: cpu".

---

## Additional Resources

- **README.md**: Main project documentation
- **QUICKSTART.md**: Quick start guide
- **API.md**: Complete API reference
- **GitHub Issues**: Report bugs and request features
- **PyTorch Documentation**: https://pytorch.org/docs/
- **ConvNeXt Paper**: A ConvNet for the 2020s

---

## Support

For additional help:
1. Check this manual thoroughly
2. Review other documentation files
3. Search existing issues
4. Create a new issue with:
   - Detailed error messages
   - System configuration
   - Steps to reproduce
   - Expected vs actual behavior

---

**Last Updated**: 2025
**Version**: 1.0.0
