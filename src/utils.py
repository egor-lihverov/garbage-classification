"""
Utility functions for metrics calculation, visualization, and model management
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch
from tqdm import tqdm

from src.config import Config


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for idx, class_name in enumerate(Config.CLASS_NAMES):
        if idx < len(precision_per_class):
            metrics[f'{class_name}_precision'] = float(precision_per_class[idx])
            metrics[f'{class_name}_recall'] = float(recall_per_class[idx])
            metrics[f'{class_name}_f1'] = float(f1_per_class[idx])
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot and save confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=Config.CLASS_NAMES,
        yticklabels=Config.CLASS_NAMES
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path):
    """
    Plot training and validation loss and accuracy
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (without extension)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics(metrics, save_path):
    """
    Save metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(load_path):
    """
    Load metrics from JSON file
    
    Args:
        load_path: Path to load the JSON file from
    
    Returns:
        Dictionary of metrics
    """
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_acc: Best validation accuracy
        save_path: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_path: Path to load the checkpoint from
        device: Device to load the model on
    
    Returns:
        model, optimizer, scheduler, start_epoch, best_acc
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch} with best accuracy: {best_acc:.4f}")
    
    return model, optimizer, scheduler, start_epoch, best_acc


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories():
    """
    Create necessary directories for checkpoints and results
    """
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)


def print_classification_report(y_true, y_pred):
    """
    Print sklearn classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=Config.CLASS_NAMES))
