"""
Testing and evaluation script for Trash Classification
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from src.config import Config
from src.model import create_model
from src.dataset import create_dataloaders
from src.utils import (
    calculate_metrics,
    plot_confusion_matrix,
    save_metrics,
    load_checkpoint,
    print_classification_report,
    create_directories
)


def test(model, test_loader, criterion, device):
    """
    Test model and return predictions
    
    Args:
        model: PyTorch model
        test_loader: Test dataloader
        criterion: Loss function
        device: Device to test on
    
    Returns:
        test_loss, y_true, y_pred, y_scores
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Store results
            running_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_scores = np.array(all_scores)
    
    return test_loss, y_true, y_pred, y_scores


def evaluate(checkpoint_path):
    """
    Main evaluation function
    
    Args:
        checkpoint_path: Path to model checkpoint
    """
    # Create directories
    create_directories()
    
    # Set device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        Config.DATA_PATH,
        Config.CLASS_MAPPING,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    
    # Create model
    print("Loading model...")
    model = create_model(pretrained=False)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Test model
    print("\nEvaluating model on test set...")
    test_loss, y_true, y_pred, y_scores = test(
        model, test_loader, criterion, device
    )
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    metrics['test_loss'] = test_loss
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    
    print("\nPer-class Metrics:")
    for class_name in Config.CLASS_NAMES:
        print(f"\n{class_name.capitalize()}:")
        print(f"  Precision: {metrics[f'{class_name}_precision']:.4f}")
        print(f"  Recall: {metrics[f'{class_name}_recall']:.4f}")
        print(f"  F1-Score: {metrics[f'{class_name}_f1']:.4f}")
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    # Save metrics
    metrics_path = f"{Config.RESULTS_DIR}/test_metrics.json"
    save_metrics(metrics, metrics_path)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot confusion matrix
    cm_path = f"{Config.RESULTS_DIR}/confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    checkpoint_path = f"{Config.CHECKPOINT_DIR}/best_model.pth"
    evaluate(checkpoint_path)
