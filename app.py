"""
Flask web application for trash classification
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import base64
from flask import Flask, request, jsonify, render_template

from src.config import Config
from src.model import create_model

app = Flask(__name__)

# Global variables for model and device
model = None
device = None
transform = None


def load_model():
    """Load the trained model once at startup"""
    global model, device, transform
    
    print("Loading model...")
    device = torch.device(Config.DEVICE)
    
    # Create model
    model = create_model(pretrained=False)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = f'{Config.CHECKPOINT_DIR}/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    print(f"Model loaded on {device}")


def predict_image(image):
    """
    Predict class for an image
    
    Args:
        image: PIL Image object
    
    Returns:
        Dictionary with prediction results
    """
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform image
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
    
    # Sort probabilities
    sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'predicted_class': class_name,
        'confidence': float(confidence_score),
        'all_probabilities': class_probabilities,
        'top_predictions': [
            {'class': cls, 'probability': float(prob)}
            for cls, prob in sorted_probs
        ]
    }


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image prediction requests
    Accepts either file upload or base64 encoded image
    """
    try:
        # Check if file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read image
            image = Image.open(file.stream)
        
        # Check if base64 data was sent
        elif 'image_data' in request.json:
            image_data = request.json['image_data']
            
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        
        else:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Make prediction
        result = predict_image(image)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
