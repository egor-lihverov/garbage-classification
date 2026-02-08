# Quick Start Guide

This guide will help you get started with the Trash Classification project in minutes.

## Setup Environment

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Verify installation (dependencies should be installed already)
python --version
pip list | grep torch
```

If you haven't installed dependencies yet, run:
```bash
pip install -r requirements.txt
```

## Train the Model

```bash
# Train with default settings
python main.py train
```

This will:
- Load the dataset from the configured Kaggle path
- Create train/val/test splits (70/15/15)
- Train ConvNeXt Tiny for up to 50 epochs
- Save best model to `checkpoints/best_model.pth`
- Generate training plots in `results/`

## Test the Model

```bash
# Evaluate on test set
python main.py test
```

This will:
- Load best checkpoint
- Evaluate on test set
- Display comprehensive metrics (accuracy, precision, recall, F1)
- Save confusion matrix and metrics to `results/`

## Predict on New Images

### Command Line Interface

```bash
# Single image
python main.py predict --image /path/to/image.jpg

# Directory of images
python main.py predict --image /path/to/images/
```

### Web Application (Recommended)

Launch the interactive web interface:

```bash
python app.py
```

Then open your browser to `http://localhost:5000`

**Web Application Features:**
- 🎯 **Drag & Drop**: Simply drag an image onto the page
- 📋 **Paste**: Copy an image and press Ctrl+V to paste
- 📂 **Browse**: Click to select files from your computer
- 📱 **Mobile Friendly**: Works on all devices

**Example Workflow:**
1. Start the web app: `python app.py`
2. Open browser to `http://localhost:5000`
3. Upload or paste an image
4. Click "Classify Image"
5. View results with confidence scores and probability bars

## Expected Results

### Training Time
- With GPU: 30-60 minutes
- With CPU: 2-4 hours

### Expected Performance
- Test accuracy: 92-96%
- Macro F1-Score: 0.91-0.95
- Good performance on plastic, glass, and metal classes

### Inference Speed
- **Web Application**: ~10-20ms per image (GPU)
- **Command Line**: ~10-20ms per image (GPU)
- **CPU**: ~50-100ms per image

### Output Files

After training:
```
checkpoints/
└── best_model.pth          # Best model checkpoint

results/
├── training_history.png    # Loss and accuracy curves
├── test_metrics.json       # All metrics
└── confusion_matrix.png    # Confusion matrix visualization
```

## Configuration

Edit `src/config.py` to customize:

```python
# Adjust batch size based on GPU memory
BATCH_SIZE = 32  # Reduce to 16 if out of memory

# Adjust learning rate
LEARNING_RATE = 3e-4

# Change number of epochs
NUM_EPOCHS = 50

# Modify data splits
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
```

## Troubleshooting

### Out of Memory
```python
# In src/config.py:
BATCH_SIZE = 16  # Reduce batch size
NUM_WORKERS = 0  # Disable multiprocessing
```

### Slow Loading
```python
# In src/config.py:
NUM_WORKERS = 2  # Reduce workers
```

### CUDA Not Available
The model will automatically use CPU if CUDA is not available. Training will be slower.

## Class Mapping

The model maps original dataset classes to 4 target classes:

| Original Class | Target Class |
|---------------|--------------|
| plastic | plastic |
| glass | glass |
| metal | metal |
| paper, cardboard, trash, battery, shoes, clothes, organic | others |

## Next Steps

1. Train the model: `python main.py train`
2. Evaluate performance: `python main.py test`
3. Test on your own images: `python main.py predict --image test.jpg`
4. Check `results/` for detailed metrics and visualizations

## Tips

- Use GPU for faster training (check with `python -c "import torch; print(torch.cuda.is_available())"`)
- Monitor training progress - early stopping will halt if validation doesn't improve
- Best model is saved automatically when validation accuracy improves
- Adjust hyperparameters in `src/config.py` if needed
- For production use, consider disabling debug mode in `app.py`: `app.run(host='0.0.0.0', port=5000, debug=False)`

## Quick Reference

### Commands Summary

```bash
# Train model
python main.py train

# Test model
python main.py test

# Predict via CLI
python main.py predict --image path/to/image.jpg

# Launch web app
python app.py
```

### Key Files

- `src/config.py` - Configuration settings
- `main.py` - CLI entry point
- `app.py` - Web application server
- `checkpoints/best_model.pth` - Trained model
- `results/` - Training outputs and metrics

## Support

See `README.md` for detailed documentation including:
- API endpoints and usage
- Advanced configuration options
- Troubleshooting guide
- Performance benchmarks
- Model export options
