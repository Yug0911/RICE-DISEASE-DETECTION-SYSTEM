"""
Rice Leaf Disease Detection - Web Demo (Flask)
Upload an image via browser and get prediction
"""
import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, render_template_string, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# Load model
print("Loading model...")
model = tf.keras.models.load_model("models/best_5class.h5")
CLASS_NAMES = ['Bacterialblight', 'Brownspot', 'Healthy', 'Leafsmut', 'Rice Blast']
IMG_SIZE = (300, 300)
print("Model loaded. Starting web server...")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Rice Leaf Disease Detection</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c7a7b; text-align: center; }
        .upload-area { border: 2px dashed #4299e1; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; cursor: pointer; }
        .upload-area:hover { background: #ebf8ff; }
        input[type="file"] { display: none; }
        button { background: #4299e1; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #3182ce; }
        .preview { max-width: 400px; display: block; margin: 20px auto; border-radius: 5px; }
        .result { margin-top: 20px; padding: 20px; background: #f0fff4; border-radius: 5px; border-left: 4px solid #2c7a7b; }
        .prediction { font-size: 24px; font-weight: bold; color: #2c7a7b; }
        .confidence { font-size: 18px; color: #666; margin-top: 5px; }
        .prob-list { margin-top: 15px; }
        .prob-item { display: flex; align-items: center; margin: 8px 0; }
        .prob-bar { height: 20px; background: #e2e8f0; border-radius: 3px; overflow: hidden; margin-left: 10px; }
        .prob-fill { height: 100%; background: linear-gradient(90deg, #4299e1, #2c7a7b); }
        .prob-label { width: 150px; font-weight: bold; }
        .stats { text-align: center; color: #666; font-size: 12px; margin-top: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌾 Rice Leaf Disease Detection</h1>
        <p style="text-align: center; color: #666;">Upload a rice leaf image to identify diseases</p>
        
        <form id="uploadForm">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📁 Click to select image or drag & drop</p>
                <p style="font-size: 12px; color: #999;">Supports: JPG, PNG, BMP, TIFF</p>
            </div>
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
            <div style="text-align: center;">
                <button type="submit">🔍 Predict Disease</button>
            </div>
        </form>
        
        <div id="previewContainer" style="text-align: center; display: none;">
            <img id="previewImg" class="preview" src="" alt="Preview">
        </div>
        
        <div id="resultContainer" class="result" style="display: none;">
            <div class="prediction" id="prediction"></div>
            <div class="confidence" id="confidence"></div>
            <div class="prob-list" id="probList"></div>
        </div>
        
        <div class="stats">
            Model: EfficientNetB3 | Accuracy: 95.23% | 5 Classes (Bacterialblight, Brownspot, Healthy, Leafsmut, Rice Blast)
        </div>
    </div>

    <script>
        function previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImg').src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        }
        
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) {
                alert('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            const btn = document.querySelector('button');
            btn.textContent = '⏳ Processing...';
            btn.disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                document.getElementById('prediction').textContent = `Prediction: ${data.prediction}`;
                document.getElementById('confidence').textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
                
                let probsHTML = '';
                for (let [cls, prob] of Object.entries(data.all_probabilities)) {
                    const pct = (prob * 100).toFixed(1);
                    probsHTML += `
                        <div class="prob-item">
                            <span class="prob-label">${cls}</span>
                            <div class="prob-bar"><div class="prob-fill" style="width: ${pct}%"></div></div>
                            <span>${pct}%</span>
                        </div>`;
                }
                document.getElementById('probList').innerHTML = probsHTML;
                document.getElementById('resultContainer').style.display = 'block';
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.textContent = '🔍 Predict Disease';
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Read and preprocess
    img = Image.open(BytesIO(file.read()))
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    
    result = {
        'prediction': CLASS_NAMES[predicted_idx],
        'confidence': float(predictions[predicted_idx] * 100),
        'all_probabilities': {cls: float(prob) for cls, prob in zip(CLASS_NAMES, predictions)}
    }
    return jsonify(result)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Rice Leaf Disease Detection - Web Demo")
    print("="*50)
    print("Open browser to: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
