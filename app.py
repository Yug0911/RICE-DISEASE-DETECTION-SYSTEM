"""
Rice Leaf Disease Detection - Simple Flask App
For Railway deployment
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

port = int(os.environ.get("PORT", 8080))

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from io import BytesIO

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model("models/best_5class.h5")
CLASS_NAMES = ['Bacterialblight', 'Brownspot', 'Healthy', 'Leafsmut', 'Rice Blast']
IMG_SIZE = (300, 300)
print("Model loaded!")

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Rice Leaf Disease Detection</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; background: #0a0a0a; color: #fff; }
        h1 { color: #3b82f6; text-align: center; }
        .upload { border: 2px dashed #3b82f6; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
        button { background: #3b82f6; color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
        button:hover { background: #2563eb; }
        .result { margin-top: 20px; padding: 20px; background: #1a1a1a; border-radius: 10px; }
        .disease { font-size: 24px; font-weight: bold; color: #3b82f6; }
        .confidence { font-size: 18px; color: #10b981; }
    </style>
</head>
<body>
    <h1>Rice Leaf Disease Detection</h1>
    <form method="post" enctype="multipart/form-data">
        <div class="upload">
            <input type="file" name="image" accept="image/*" required>
        </div>
        <button type="submit">Analyze</button>
    </form>
    {{% if prediction %}}
    <div class="result">
        <div class="disease">Disease: {{prediction}}</div>
        <div class="confidence">Confidence: {{confidence}}%</div>
    </div>
    {{% endif %}}
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML, prediction=None, confidence=None)

@app.route('/', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template_string(HTML, prediction=None, confidence=None)
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)[0]
    idx = np.argmax(predictions)
    
    return render_template_string(HTML, 
        prediction=CLASS_NAMES[idx], 
        confidence=f"{predictions[idx]*100:.1f}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)