"""
Configuration file for Trash Classification project
"""

import torch

class Config:
    # Data paths
    DATA_PATH = "/home/egorl/.cache/kagglehub/datasets/sumn2u/garbage-classification-v2/versions/11/original"
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    
    # Model
    MODEL_NAME = "convnext_tiny"
    NUM_CLASSES = 4
    CLASS_NAMES = ["plastic", "glass", "metal", "others"]
    
    # Training hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 2
    
    # Data
    IMAGE_SIZE = 224
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Training settings
    NUM_WORKERS = 8
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Early stopping
    PATIENCE = 3
    
    # Random seed
    SEED = 42
    
    # Class mapping from original dataset to our classes
    # Based on garbage-classification-v2 dataset structure
    CLASS_MAPPING = {
        'plastic': 'plastic',
        'glass': 'glass',
        'metal': 'metal',
        'paper': 'others',
        'cardboard': 'others',
        'trash': 'others',
        'battery': 'others',
        'shoes': 'others',
        'clothes': 'others',
        'organic': 'others',
    }
