# Trash Classification API Documentation

Complete API reference for the Trash Classification web application.

## Base URL

```
http://localhost:5000
```

## Overview

The Trash Classification API provides RESTful endpoints for classifying trash images using a ConvNeXt Tiny model. The API supports both file uploads and base64-encoded image data.

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing API keys or OAuth.

---

## Endpoints

### 1. Health Check

Check if the API is running and the model is loaded.

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Example:**

```bash
curl http://localhost:5000/health
```

---

### 2. Predict Image (File Upload)

Upload an image file for classification.

**Endpoint:** `POST /predict`

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type   | Required | Description                    |
|-----------|--------|----------|--------------------------------|
| file      | File   | Yes      | Image file to classify (JPG, PNG, etc.) |

**Request:**

```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost:5000/predict
```

**Response (Success):**

```json
{
  "success": true,
  "result": {
    "predicted_class": "plastic",
    "confidence": 0.9523,
    "all_probabilities": {
      "plastic": 0.9523,
      "glass": 0.0241,
      "metal": 0.0156,
      "others": 0.0080
    },
    "top_predictions": [
      {
        "class": "plastic",
        "probability": 0.9523
      },
      {
        "class": "glass",
        "probability": 0.0241
      },
      {
        "class": "metal",
        "probability": 0.0156
      },
      {
        "class": "others",
        "probability": 0.0080
      }
    ]
  }
}
```

**Response (Error):**

```json
{
  "error": "Error message description"
}
```

**Status Codes:**

| Code | Description           |
|------|-----------------------|
| 200  | Success               |
| 400  | Bad Request           |
| 500  | Internal Server Error |

---

### 3. Predict Image (Base64)

Classify an image using base64-encoded data.

**Endpoint:** `POST /predict`

**Content-Type:** `application/json`

**Parameters:**

| Parameter   | Type   | Required | Description                              |
|-------------|--------|----------|------------------------------------------|
| image_data  | String | Yes      | Base64-encoded image data (with or without data URL prefix) |

**Request:**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image_data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD..."}' \
  http://localhost:5000/predict
```

With data URL prefix:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD..."}' \
  http://localhost:5000/predict
```

**Response:** Same as file upload endpoint

---

### 4. Web Interface

Access the interactive web interface.

**Endpoint:** `GET /`

**Response:** HTML page with interactive classification UI

---

## Response Fields

### Result Object

| Field              | Type   | Description                                  |
|--------------------|--------|----------------------------------------------|
| predicted_class    | String | Predicted class name (lowercase)             |
| confidence         | Float  | Confidence score (0.0 - 1.0)                 |
| all_probabilities  | Object | Probability for each class                   |
| top_predictions    | Array  | Array of predictions sorted by probability   |

### Prediction Item

| Field      | Type   | Description              |
|------------|--------|--------------------------|
| class      | String | Class name (lowercase)   |
| probability| Float  | Probability (0.0 - 1.0)  |

---

## Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- GIF
- WEBP
- TIFF

**Note:** Images are automatically converted to RGB format during preprocessing.

---

## Classes

The API classifies images into 4 categories:

| Class   | Description                                    |
|---------|------------------------------------------------|
| plastic | Plastic bottles and containers                |
| glass   | Glass bottles and containers                  |
| metal   | Metal cans and containers                      |
| others  | Paper, cardboard, organic waste, batteries, etc. |

---

## Image Preprocessing

Images are automatically preprocessed before prediction:

1. **Resize**: Resized to 256x256 pixels
2. **Center Crop**: Cropped to 224x224 pixels
3. **Normalization**: Normalized using ImageNet mean/std
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

---

## Error Handling

### Common Errors

**400 Bad Request**

- No file selected
- No image data provided
- Invalid image format

**500 Internal Server Error**

- Image processing error
- Model prediction error
- Server-side exception

### Error Response Format

```json
{
  "error": "Error message description"
}
```

---

## Rate Limiting

Currently, there is no rate limiting implemented. For production deployments, consider implementing rate limiting to prevent abuse.

---

## Performance

### Inference Speed

| Hardware | Average Response Time |
|----------|----------------------|
| RTX 3090 | 10-15ms              |
| RTX 3060 | 15-20ms              |
| GTX 1660 | 20-30ms              |
| CPU (8-core) | 50-100ms       |

### Throughput

- GPU: ~50-100 requests/second
- CPU: ~10-20 requests/second

---

## Python Client Example

```python
import requests
import base64

# Predict with file upload
def predict_with_file(image_path):
    url = "http://localhost:5000/predict"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    return response.json()

# Predict with base64
def predict_with_base64(image_path):
    url = "http://localhost:5000/predict"
    
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    data = {'image_data': image_data}
    response = requests.post(url, json=data)
    
    return response.json()

# Example usage
result = predict_with_file('test_image.jpg')
if result.get('success'):
    print(f"Class: {result['result']['predicted_class']}")
    print(f"Confidence: {result['result']['confidence']:.2%}")
else:
    print(f"Error: {result.get('error')}")
```

---

## JavaScript Client Example

```javascript
// Predict with file upload
async function predictWithFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Predict with base64
async function predictWithBase64(imageData) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_data: imageData })
    });
    
    return await response.json();
}

// Example usage
document.getElementById('fileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    const result = await predictWithFile(file);
    
    if (result.success) {
        console.log(`Class: ${result.result.predicted_class}`);
        console.log(`Confidence: ${(result.result.confidence * 100).toFixed(2)}%`);
    } else {
        console.error(`Error: ${result.error}`);
    }
});
```

---

## cURL Examples

### File Upload

```bash
# Upload and classify an image
curl -X POST \
  -F "file=@image.jpg" \
  http://localhost:5000/predict

# Save response to file
curl -X POST \
  -F "file=@image.jpg" \
  http://localhost:5000/predict \
  -o response.json

# Pretty print response
curl -X POST \
  -F "file=@image.jpg" \
  http://localhost:5000/predict \
  | jq '.'
```

### Base64 Upload

```bash
# Convert image to base64 and upload
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{\"image_data\": \"$(base64 -w 0 image.jpg)\"}" \
  http://localhost:5000/predict

# From existing base64 file
curl -X POST \
  -H "Content-Type: application/json" \
  -d @image_data.json \
  http://localhost:5000/predict
```

### Health Check

```bash
# Check API status
curl http://localhost:5000/health

# Check with verbose output
curl -v http://localhost:5000/health
```

---

## Deployment

### Production Configuration

Edit `app.py` for production:

```python
if __name__ == '__main__':
    load_model()
    
    # Production configuration
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug mode
        threaded=True  # Enable threading for concurrent requests
    )
```

### Using Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using uWSGI

```bash
# Install uWSGI
pip install uwsgi

# Run with uWSGI
uwsgi --http :5000 --wsgi-file app.py --callable app --processes 4
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t trash-classifier .
docker run -p 5000:5000 trash-classifier
```

---

## Security Considerations

1. **Authentication**: Implement API keys or OAuth for production
2. **Rate Limiting**: Add rate limiting to prevent abuse
3. **File Size Limits**: Implement maximum file size restrictions
4. **Input Validation**: Validate all inputs before processing
5. **HTTPS**: Use HTTPS in production environments
6. **CORS**: Configure CORS appropriately for your use case

---

## Support

For issues or questions:
- Check the main README.md for project documentation
- Review QUICKSTART.md for quick setup guide
- Check logs for detailed error messages
