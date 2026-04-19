"""
Rice Leaf Disease Detection - Professional Web Application
A clean, professional interface for crop disease detection
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
from io import BytesIO

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model("models/best_5class.h5")
CLASS_NAMES = ['Bacterialblight', 'Brownspot', 'Healthy', 'Leafsmut', 'Rice Blast']
IMG_SIZE = (300, 300)
print("Model loaded. Starting server...")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RiceScan - Disease Detection System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Playfair+Display:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --black: #0a0a0a;
            --dark: #141414;
            --dark-gray: #1f1f1f;
            --gray: #2a2a2a;
            --light-gray: #666666;
            --off-white: #f5f5f5;
            --white: #ffffff;
            --green: #2d5a27;
            --green-light: #4a7c44;
            --cream: #faf9f6;
            --brown: #5c4033;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--cream);
            color: var(--black);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .header {
            background: var(--white);
            border-bottom: 1px solid rgba(0,0,0,0.08);
            padding: 20px 0;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-inner {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .logo-mark {
            width: 36px;
            height: 36px;
            background: var(--green);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 18px;
        }
        
        .logo-text {
            font-family: 'Playfair Display', serif;
            font-size: 22px;
            font-weight: 600;
            color: var(--black);
            letter-spacing: -0.5px;
        }
        
        .nav-links {
            display: flex;
            gap: 32px;
            list-style: none;
        }
        
        .nav-links a {
            text-decoration: none;
            color: var(--light-gray);
            font-size: 14px;
            font-weight: 500;
            transition: color 0.2s;
        }
        
        .nav-links a:hover {
            color: var(--black);
        }
        
        .hero {
            max-width: 1200px;
            margin: 0 auto;
            padding: 60px 30px 40px;
            text-align: center;
        }
        
        .hero h1 {
            font-family: 'Playfair Display', serif;
            font-size: 48px;
            font-weight: 600;
            color: var(--black);
            line-height: 1.2;
            margin-bottom: 16px;
            letter-spacing: -1px;
        }
        
        .hero p {
            font-size: 17px;
            color: var(--light-gray);
            max-width: 500px;
            margin: 0 auto;
        }
        
        .main-card {
            max-width: 700px;
            margin: 0 auto 60px;
            padding: 0 30px;
        }
        
        .upload-card {
            background: var(--white);
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        }
        
        .upload-area {
            border: 2px dashed rgba(0,0,0,0.15);
            border-radius: 12px;
            padding: 50px 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            background: var(--cream);
        }
        
        .upload-area:hover {
            border-color: var(--green);
            background: #f8f7f4;
        }
        
        .upload-icon {
            width: 56px;
            height: 56px;
            margin: 0 auto 16px;
            background: var(--dark);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .upload-icon svg {
            width: 24px;
            height: 24px;
            stroke: var(--white);
        }
        
        .upload-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--black);
            margin-bottom: 4px;
        }
        
        .upload-sub {
            font-size: 14px;
            color: var(--light-gray);
        }
        
        input[type="file"] {
            display: none;
        }
        
        .analyze-btn {
            display: block;
            width: 100%;
            background: var(--green);
            color: white;
            border: none;
            padding: 16px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            font-family: inherit;
            margin-top: 20px;
        }
        
        .analyze-btn:hover:not(:disabled) {
            background: var(--green-light);
            transform: translateY(-1px);
        }
        
        .analyze-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .preview-container {
            display: none;
            margin-top: 24px;
        }
        
        .preview-img {
            width: 100%;
            border-radius: 12px;
        }
        
        .result-card {
            display: none;
            margin-top: 24px;
            background: var(--dark);
            border-radius: 14px;
            padding: 28px;
            color: var(--white);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 24px;
        }
        
        .result-label {
            font-size: 12px;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }
        
        .result-disease {
            font-size: 28px;
            font-weight: 600;
            font-family: 'Playfair Display', serif;
        }
        
        .result-confidence {
            text-align: right;
        }
        
        .confidence-num {
            font-size: 36px;
            font-weight: 700;
            color: var(--green-light);
        }
        
        .confidence-suffix {
            font-size: 18px;
            color: rgba(255,255,255,0.6);
        }
        
        .result-divider {
            height: 1px;
            background: rgba(255,255,255,0.15);
            margin: 20px 0;
        }
        
        .prob-title {
            font-size: 11px;
            color: rgba(255,255,255,0.4);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 14px;
        }
        
        .prob-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .prob-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .prob-name {
            width: 110px;
            font-size: 13px;
            color: rgba(255,255,255,0.8);
        }
        
        .prob-bar-bg {
            flex: 1;
            height: 6px;
            background: rgba(255,255,255,0.15);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .prob-bar-fill {
            height: 100%;
            background: var(--green-light);
            border-radius: 3px;
            transition: width 0.5s ease;
        }
        
        .prob-pct {
            width: 45px;
            text-align: right;
            font-size: 13px;
            font-weight: 600;
            color: rgba(255,255,255,0.9);
        }
        
        .footer {
            background: var(--dark);
            color: var(--white);
            padding: 40px 30px;
            text-align: center;
        }
        
        .footer-inner {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .footer-brand {
            font-family: 'Playfair Display', serif;
            font-size: 24px;
            margin-bottom: 12px;
        }
        
        .footer-text {
            font-size: 14px;
            color: rgba(255,255,255,0.5);
        }
        
        .footer-divider {
            height: 1px;
            background: rgba(255,255,255,0.1);
            margin: 30px 0;
        }
        
        .footer-info {
            display: flex;
            justify-content: center;
            gap: 24px;
            font-size: 13px;
            color: rgba(255,255,255,0.4);
        }
        
        @media (max-width: 600px) {
            .hero h1 {
                font-size: 32px;
            }
            
            .nav-links {
                display: none;
            }
            
            .upload-card {
                padding: 24px;
            }
            
            .result-header {
                flex-direction: column;
                gap: 12px;
            }
            
            .result-confidence {
                text-align: left;
            }
            
            .footer-info {
                flex-direction: column;
                gap: 8px;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-inner">
            <div class="logo">
                <div class="logo-mark">R</div>
                <div class="logo-text">RiceScan</div>
            </div>
            <nav>
                <ul class="nav-links">
                    <li><a href="#">Home</a></li>
                    <li><a href="#">About</a></li>
                    <li><a href="#">Contact</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <section class="hero">
        <h1>Detect Rice Leaf Diseases<br>Before It's Too Late</h1>
        <p>Upload a photo of your rice leaf and our AI system will identify any diseases within seconds.</p>
    </section>
    
    <div class="main-card">
        <div class="upload-card">
            <form id="uploadForm">
                <label class="upload-area" for="fileInput">
                    <div class="upload-icon">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                        </svg>
                    </div>
                    <div class="upload-title">Click to upload a leaf image</div>
                    <div class="upload-sub">JPG, PNG up to 10MB</div>
                </label>
                <input type="file" id="fileInput" accept="image/*">
                
                <button type="submit" class="analyze-btn" id="analyzeBtn" disabled>Analyze Image</button>
            </form>
            
            <div class="preview-container" id="previewCont">
                <img id="previewImg" class="preview-img" src="" alt="Preview">
            </div>
            
            <div class="result-card" id="resultCont">
                <div class="result-header">
                    <div>
                        <div class="result-label">Detected Disease</div>
                        <div class="result-disease" id="diseaseName">-</div>
                    </div>
                    <div class="result-confidence">
                        <div class="result-label">Confidence</div>
                        <div class="confidence-num" id="confidenceNum">-</div>
                        <span class="confidence-suffix">%</span>
                    </div>
                </div>
                
                <div class="result-divider"></div>
                
                <div class="prob-title">All Probabilities</div>
                <div class="prob-list" id="probList"></div>
            </div>
        </div>
    </div>
    
    <footer class="footer">
        <div class="footer-inner">
            <div class="footer-brand">RiceScan</div>
            <p class="footer-text">Helping farmers protect their crops with intelligent disease detection</p>
            
            <div class="footer-divider"></div>
            
            <div class="footer-info">
                <span>Model: EfficientNetB3</span>
                <span>Accuracy: 95.23%</span>
                <span>5 Disease Classes</span>
            </div>
        </div>
    </footer>
    
    <script>
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const previewCont = document.getElementById('previewCont');
        const previewImg = document.getElementById('previewImg');
        const resultCont = document.getElementById('resultCont');
        const diseaseName = document.getElementById('diseaseName');
        const confidenceNum = document.getElementById('confidenceNum');
        const probList = document.getElementById('probList');
        
        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    previewCont.style.display = 'block';
                    analyzeBtn.disabled = false;
                    resultCont.style.display = 'none';
                };
                reader.readAsDataURL(this.files[0]);
            }
        });
        
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            if (!fileInput.files[0]) return;
            
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                diseaseName.textContent = data.prediction;
                confidenceNum.textContent = data.confidence.toFixed(1);
                
                const probs = Object.entries(data.all_probabilities)
                    .sort((a, b) => b[1] - a[1]);
                
                let probHTML = '';
                probs.forEach(([cls, prob]) => {
                    const pct = (prob * 100).toFixed(1);
                    probHTML += `
                        <div class="prob-row">
                            <div class="prob-name">${cls}</div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width: ${pct}%"></div>
                            </div>
                            <div class="prob-pct">${pct}%</div>
                        </div>`;
                });
                probList.innerHTML = probHTML;
                resultCont.style.display = 'block';
                
                resultCont.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze Image';
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
    
    img = Image.open(BytesIO(file.read()))
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    
    result = {
        'prediction': CLASS_NAMES[predicted_idx],
        'confidence': float(predictions[predicted_idx] * 100),
        'all_probabilities': {cls: float(prob) for cls, prob in zip(CLASS_NAMES, predictions)}
    }
    return jsonify(result)

if __name__ == '__main__':
    print("\n" + "="*40)
    print("RiceScan - Disease Detection System")
    print("="*40)
    print("Open http://127.0.0.1:5000")
    print("="*40 + "\n")
    app.run(debug=True, port=5000)