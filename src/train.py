"""
Training script for Trash Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time

from src.config import Config
from src.model import create_model, get_model_size
from src.dataset import create_dataloaders
from src.utils import (
    save_checkpoint,
    load_checkpoint,
    set_seed,
    create_directories,
    plot_training_history
)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate model
    
    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Average loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train(checkpoint_path=None):
    """
    Main training function
    
    Args:
        checkpoint_path: Path to checkpoint to resume from (optional)
    """
    # Set random seed for reproducibility
    set_seed(Config.SEED)
    
    # Create directories
    create_directories()
    
    # Set device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        Config.DATA_PATH,
        Config.CLASS_MAPPING,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    
    # Create model
    print("Creating model...")
    model = create_model(pretrained=True)
    model = model.to(device)
    
    # Print model info
    total_params, trainable_params = get_model_size(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # Training variables
    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Load checkpoint if provided
    if checkpoint_path:
        model, optimizer, scheduler, start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        history['train_loss'] = [0] * start_epoch
        history['train_acc'] = [0] * start_epoch
        history['val_loss'] = [0] * start_epoch
        history['val_acc'] = [0] * start_epoch
    
    # Training loop
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    total_time = 0
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        print(f"\nEpoch [{epoch+1}/{Config.NUM_EPOCHS}]")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = f"{Config.CHECKPOINT_DIR}/best_model.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, save_path)
            print(f"  New best model saved! (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience counter: {patience_counter}/{Config.PATIENCE}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plot_training_history(
        history,
        f"{Config.RESULTS_DIR}/training_history"
    )
    print(f"Training history plot saved to {Config.RESULTS_DIR}/training_history.png")
    
    return model, best_val_acc


if __name__ == "__main__":
    train()
