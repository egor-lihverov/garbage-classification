"""
Inference script for predicting on new images
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

from src.config import Config
from src.model import create_model
from src.utils import create_directories


def load_model_for_inference(checkpoint_path):
    """
    Load trained model for inference
    
    Args:
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Loaded model on device
    """
    # Set device
    device = torch.device(Config.DEVICE)
    
    # Create model
    model = create_model(pretrained=False)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device


def get_inference_transform():
    """
    Get transform for inference
    
    Returns:
        Transform pipeline
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transform


def predict_single_image(model, image_path, device, transform):
    """
    Predict class for a single image
    
    Args:
        model: Trained model
        image_path: Path to image file
        device: Device to run inference on
        transform: Image transform
    
    Returns:
        Predicted class index, class name, and confidence scores
    """
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    class_name = Config.CLASS_NAMES[predicted_class]
    
    # Get all class probabilities
    all_probs = probabilities.cpu().numpy()[0]
    class_probabilities = {
        Config.CLASS_NAMES[i]: float(all_probs[i])
        for i in range(len(Config.CLASS_NAMES))
    }
    
    return predicted_class, class_name, confidence_score, class_probabilities


def main():
    """
    Main inference function
    """
    parser = argparse.ArgumentParser(description='Predict trash class for an image')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file or directory of images'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=f'{Config.CHECKPOINT_DIR}/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Show top k predictions'
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, device = load_model_for_inference(args.checkpoint)
    print(f"Model loaded on {device}")
    
    # Get transform
    transform = get_inference_transform()
    
    # Check if input is a file or directory
    from pathlib import Path
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image prediction
        print(f"\nPredicting for: {args.image}")
        predicted_class, class_name, confidence, all_probs = predict_single_image(
            model, str(image_path), device, transform
        )
        
        print(f"\n{'='*60}")
        print(f"Prediction: {class_name.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"{'='*60}")
        
        print(f"\nTop {args.top_k} Predictions:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for i, (cls, prob) in enumerate(sorted_probs[:args.top_k]):
            print(f"  {i+1}. {cls.capitalize()}: {prob:.4f} ({prob*100:.2f}%)")
        
    elif image_path.is_dir():
        # Directory of images
        print(f"\nPredicting for all images in: {args.image}")
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(image_path.glob(ext))
        
        if not image_files:
            print(f"No images found in {args.image}")
            return
        
        print(f"Found {len(image_files)} images\n")
        
        results = []
        for img_file in image_files:
            predicted_class, class_name, confidence, all_probs = predict_single_image(
                model, str(img_file), device, transform
            )
            results.append({
                'image': str(img_file),
                'predicted_class': class_name,
                'confidence': confidence
            })
            
            print(f"{img_file.name}: {class_name} ({confidence*100:.2f}%)")
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        class_counts = {cls: 0 for cls in Config.CLASS_NAMES}
        for result in results:
            class_counts[result['predicted_class']] += 1
        
        for cls, count in class_counts.items():
            if count > 0:
                percentage = (count / len(results)) * 100
                print(f"{cls.capitalize()}: {count} ({percentage:.1f}%)")
    
    else:
        print(f"Error: {args.image} is not a valid file or directory")


if __name__ == "__main__":
    main()
