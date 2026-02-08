"""
Model definition for Trash Classification using ConvNeXt Tiny
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import Config


class TrashClassifier(nn.Module):
    """Trash classification model based on ConvNeXt Tiny"""
    
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(TrashClassifier, self).__init__()
        
        # Load pretrained ConvNeXt Tiny
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace the classifier head for our number of classes
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def create_model(pretrained=True):
    """
    Create a TrashClassifier model
    
    Args:
        pretrained: Whether to use pretrained weights
    
    Returns:
        TrashClassifier model
    """
    model = TrashClassifier(num_classes=Config.NUM_CLASSES, pretrained=pretrained)
    return model


def get_model_size(model):
    """
    Calculate the number of parameters in the model
    
    Args:
        model: PyTorch model
    
    Returns:
        Total parameters and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
