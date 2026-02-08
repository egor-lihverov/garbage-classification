"""
Dataset loader for Trash Classification project
"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import Config


class TrashDataset(Dataset):
    """Custom dataset for trash classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels (0-3)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms():
    """
    Returns training and validation/test transforms
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, val_transform


def load_and_split_data(data_path, class_mapping):
    """
    Load images from dataset directory and apply class mapping
    
    Args:
        data_path: Path to dataset root
        class_mapping: Dictionary mapping original classes to target classes
    
    Returns:
        image_paths: List of image paths
        labels: List of corresponding labels (0-3)
    """
    image_paths = []
    labels = []
    
    data_root = Path(data_path)
    
    # Iterate through original class directories
    for original_class in data_root.iterdir():
        if not original_class.is_dir():
            continue
        
        original_class_name = original_class.name.lower()
        
        # Map to target class
        target_class = class_mapping.get(original_class_name)
        if target_class is None:
            print(f"Warning: Class '{original_class_name}' not found in mapping, skipping")
            continue
        
        # Get label index
        label_idx = Config.CLASS_NAMES.index(target_class)
        
        # Collect all images in this class directory
        for img_path in original_class.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(str(img_path))
                labels.append(label_idx)
    
    return image_paths, labels


def split_data(image_paths, labels, train_split=0.7, val_split=0.15, test_split=0.15, seed=42):
    """
    Split data into train, validation, and test sets
    
    Args:
        image_paths: List of image paths
        labels: List of corresponding labels
        train_split: Proportion of training data
        val_split: Proportion of validation data
        test_split: Proportion of test data
        seed: Random seed for reproducibility
    
    Returns:
        train_paths, val_paths, test_paths
        train_labels, val_labels, test_labels
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Combine and shuffle
    data = list(zip(image_paths, labels))
    random.seed(seed)
    random.shuffle(data)
    
    # Calculate split indices
    total = len(data)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    # Split
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Unzip
    train_paths, train_labels = zip(*train_data) if train_data else ([], [])
    val_paths, val_labels = zip(*val_data) if val_data else ([], [])
    test_paths, test_labels = zip(*test_data) if test_data else ([], [])
    
    return list(train_paths), list(train_labels), list(val_paths), list(val_labels), list(test_paths), list(test_labels)


def create_dataloaders(data_path, class_mapping, batch_size=32, num_workers=4):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_path: Path to dataset root
        class_mapping: Dictionary mapping original classes to target classes
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data
    image_paths, labels = load_and_split_data(data_path, class_mapping)
    
    print(f"Loaded {len(image_paths)} images from {data_path}")
    print(f"Class distribution:")
    for idx, class_name in enumerate(Config.CLASS_NAMES):
        count = labels.count(idx)
        print(f"  {class_name}: {count}")
    
    # Split data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_data(
        image_paths, labels,
        train_split=Config.TRAIN_SPLIT,
        val_split=Config.VAL_SPLIT,
        test_split=Config.TEST_SPLIT,
        seed=Config.SEED
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = TrashDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = TrashDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = TrashDataset(test_paths, test_labels, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader
