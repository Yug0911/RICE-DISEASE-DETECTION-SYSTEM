"""
Rice Leaf Disease Detection - Professional Web Interface
Modern, clean, production-ready UI
"""
import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, render_template_string, jsonify, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__, static_folder='static')

# Configuration
MODEL_PATH = "models/best_5class.h5"
IMG_SIZE = (300, 300)
CLASS_NAMES = ['Bacterialblight', 'Brownspot', 'Healthy', 'Leafsmut', 'Rice Blast']
CLASS_COLORS = {
    'Bacterialblight': '#dc2626',
    'Brownspot': '#ea580c',
    'Healthy': '#16a34a',
    'Leafsmut': '#2563eb',
    'Rice Blast': '#9333ea'
}

# Load model at startup
print("Initializing Rice Leaf Disease Detection System...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded successfully.")
print(f"Target classes: {CLASS_NAMES}")
print(f"Model accuracy: 95.23%")
print("System ready.\n")

# HTML Template
HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Leaf Disease Detection | AI-Powered Diagnosis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0ea5e9;
            --primary-dark: #0284c7;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #0f172a;
            --dark-light: #1e293b;
            --gray: #64748b;
            --gray-light: #94a3b8;
            --bg: #f8fafc;
            --white: #ffffff;
            --shadow: 0 10px 40px rgba(0,0,0,0.08);
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.05);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
            min-height: 100vh;
            color: var(--dark);
            line-height: 1.6;
        }

        /* Navigation */
        nav {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(226, 232, 240, 0.6);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-dark);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--primary), var(--success));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
        }

        .nav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
        }

        .nav-links a {
            color: var(--gray);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
        }

        .nav-links a:hover {
            color: var(--primary);
        }

        .nav-badge {
            background: var(--primary);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* Main Container */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Hero Section */
        .hero {
            text-align: center;
            padding: 3rem 0 4rem;
        }

        .hero h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 1rem;
            line-height: 1.2;
        }

        .hero h1 span {
            color: var(--primary);
        }

        .hero p {
            font-size: 1.125rem;
            color: var(--gray-light);
            max-width: 600px;
            margin: 0 auto 2rem;
        }

        /* Stats Bar */
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }

        .stat {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--gray-light);
            font-size: 0.95rem;
        }

        .stat-value {
            font-weight: 700;
            color: var(--dark);
        }

        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }

        @media (max-width: 968px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Cards */
        .card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--shadow);
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.12);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--bg);
        }

        .card-icon {
            width: 48px;
            height: 48px;
            background: var(--bg);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--dark);
        }

        /* Upload Zone */
        .upload-zone {
            border: 2px dashed var(--primary);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .upload-zone:hover {
            border-color: var(--primary-dark);
            background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        }

        .upload-zone.dragover {
            border-color: var(--primary-dark);
            background: linear-gradient(135deg, #bae6fd 0%, #7dd3fc 100%);
            transform: scale(1.02);
        }

        .upload-zone input[type="file"] {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }

        .upload-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.6;
        }

        .upload-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--dark);
            margin-bottom: 0.5rem;
        }

        .upload-subtitle {
            font-size: 0.875rem;
            color: var(--gray-light);
        }

        .file-info {
            margin-top: 1rem;
            padding: 0.75rem 1rem;
            background: white;
            border-radius: 8px;
            display: none;
            font-weight: 500;
            color: var(--success);
        }

        /* Preview Image */
        .preview-container {
            margin-top: 1.5rem;
            display: none;
            text-align: center;
        }

        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
        }

        .remove-btn {
            margin-top: 0.75rem;
            padding: 0.5rem 1.5rem;
            background: var(--danger);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
        }

        .remove-btn:hover {
            background: #dc2626;
        }

        /* Predict Button */
        .predict-btn {
            width: 100%;
            padding: 1.25rem;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.125rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(14, 165, 233, 0.4);
        }

        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Results Section */
        .results-container {
            display: none;
        }

        .result-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .prediction-badge {
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 30px;
            font-weight: 700;
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .confidence-score {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--success), #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .confidence-label {
            font-size: 0.875rem;
            color: var(--gray-light);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1.5rem;
        }

        /* Probability Bars */
        .prob-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .prob-item {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem 1rem;
            background: var(--bg);
            border-radius: 10px;
            transition: transform 0.2s, background 0.2s;
        }

        .prob-item:hover {
            transform: translateX(4px);
            background: #e0f2fe;
        }

        .prob-name {
            width: 140px;
            font-weight: 600;
            color: var(--dark);
            font-size: 0.95rem;
        }

        .prob-bar-container {
            flex: 1;
            height: 12px;
            background: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
        }

        .prob-bar {
            height: 100%;
            border-radius: 6px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            background: linear-gradient(90deg, var(--primary), var(--primary-dark));
        }

        .prob-value {
            width: 55px;
            text-align: right;
            font-weight: 700;
            color: var(--dark);
            font-size: 0.95rem;
        }

        /* Info Box */
        .info-box {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 4px solid var(--primary);
            padding: 1.25rem;
            border-radius: 8px;
            margin-top: 2rem;
        }

        .info-box h3 {
            color: var(--dark);
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }

        .info-box ul {
            margin-left: 1.25rem;
            color: var(--gray);
        }

        .info-box li {
            margin: 0.5rem 0;
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--gray);
            font-size: 0.875rem;
        }

        .footer-links {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 0.5rem;
        }

        .footer-links a {
            color: var(--primary);
            text-decoration: none;
        }

        /* Loading Spinner */
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2.5rem;
            }
            .nav-container {
                padding: 1rem;
            }
            .card {
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav>
        <div class="nav-container">
            <div class="logo">
                <div class="logo-icon">🌾</div>
                RiceGuard AI
            </div>
            <div class="nav-links">
                <span class="nav-badge">95.23% Accuracy</span>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="container">
        <!-- Hero -->
        <div class="hero">
            <h1>Detect <span>Rice Leaf Diseases</span> with AI</h1>
            <p>Upload a photograph of a rice leaf and our deep learning model will instantly identify any disease. Built on EfficientNetB3 with 5-class classification.</p>
            <div class="stats-bar">
                <div class="stat">
                    <span>Model:</span>
                    <span class="stat-value">EfficientNetB3</span>
                </div>
                <div class="stat">
                    <span>Classes:</span>
                    <span class="stat-value">5</span>
                </div>
                <div class="stat">
                    <span>Test Images:</span>
                    <span class="stat-value">901</span>
                </div>
                <div class="stat">
                    <span>Accuracy:</span>
                    <span class="stat-value">95.23%</span>
                </div>
            </div>
        </div>

        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Left: Upload -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📤</div>
                    <h2 class="card-title">Upload Leaf Image</h2>
                </div>

                <div class="upload-zone" id="uploadZone">
                    <input type="file" id="fileInput" accept="image/jpeg,image/png,image/jpg,image/bmp,image/tiff">
                    <div class="upload-icon">📷</div>
                    <div class="upload-title">Click to upload or drag & drop</div>
                    <div class="upload-subtitle">Supports: JPG, PNG, BMP, TIFF (recommended: clear, close-up leaf photo)</div>
                    <div class="file-info" id="fileInfo"></div>
                </div>

                <div class="preview-container" id="previewContainer">
                    <img id="previewImg" class="preview-image" src="" alt="Preview">
                    <button class="remove-btn" onclick="removeImage()">Remove Image</button>
                </div>

                <button class="predict-btn" id="predictBtn" onclick="predict()" disabled>
                    <span>🔍</span>
                    <span>Analyze Leaf</span>
                </button>

                <div class="info-box">
                    <h3>About This System</h3>
                    <ul>
                        <li>Trained on 5,983 balanced rice leaf images</li>
                        <li>Detects: Bacterial Blight, Brownspot, Healthy, Leaf Smut, Rice Blast</li>
                        <li>Model: EfficientNetB3 with class-weighted training</li>
                        <li>All processing done locally — images are not stored</li>
                    </ul>
                </div>
            </div>

            <!-- Right: Results -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📊</div>
                    <h2 class="card-title">Prediction Results</h2>
                </div>

                <div id="placeholder" style="text-align: center; padding: 3rem 1rem; color: var(--gray);">
                    <div style="font-size: 4rem; opacity: 0.3; margin-bottom: 1rem;">📋</div>
                    <p>Upload an image to see prediction results</p>
                </div>

                <div class="results-container" id="resultsContainer">
                    <div class="result-header">
                        <div class="prediction-badge" id="predictionBadge">--</div>
                        <div class="confidence-label">Confidence Score</div>
                        <div class="confidence-score" id="confidenceScore">--%</div>
                    </div>

                    <div class="prob-list" id="probList"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer>
        <p>Rice Leaf Disease Detection System | Deep Learning Project</p>
        <div class="footer-links">
            <a href="#" onclick="alert('EfficientNetB3 trained on 5,983 images across 5 classes. Test accuracy: 95.23%')">Model Info</a>
            <a href="#" onclick="alert('Project developed as part of Deep Learning coursework')">About</a>
        </div>
    </footer>

    <script>
        let selectedFile = null;

        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const previewContainer = document.getElementById('previewContainer');
        const previewImg = document.getElementById('previewImg');
        const predictBtn = document.getElementById('predictBtn');
        const placeholder = document.getElementById('placeholder');
        const resultsContainer = document.getElementById('resultsContainer');

        // Drag & drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            handleFile(file);
        });

        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });

        function handleFile(file) {
            if (!file || !file.type.startsWith('image/')) {
                alert('Please select a valid image file');
                return;
            }

            selectedFile = file;
            fileInfo.textContent = `Selected: ${file.name} (${(file.size/1024).toFixed(1)} KB)`;
            fileInfo.style.display = 'block';

            // Preview
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                previewContainer.style.display = 'block';
                predictBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }

        function removeImage() {
            selectedFile = null;
            fileInput.value = '';
            fileInfo.style.display = 'none';
            previewContainer.style.display = 'none';
            predictBtn.disabled = true;
            placeholder.style.display = 'block';
            resultsContainer.style.display = 'none';
        }

        async function predict() {
            if (!selectedFile) return;

            // Show loading state
            predictBtn.disabled = true;
            predictBtn.innerHTML = '<span class="spinner"></span> Analyzing...';

            const formData = new FormData();
            formData.append('image', selectedFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Prediction failed');
                }

                const data = await response.json();
                displayResults(data);

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                predictBtn.disabled = false;
                predictBtn.innerHTML = '<span>🔍</span><span>Analyze Leaf</span>';
            }
        }

        function displayResults(data) {
            placeholder.style.display = 'none';
            resultsContainer.style.display = 'block';

            const prediction = data.prediction;
            const confidence = data.confidence;
            const allProbs = data.all_probabilities;

            // Set prediction badge
            const badge = document.getElementById('predictionBadge');
            badge.textContent = prediction;
            badge.style.background = getClassColor(prediction);

            // Set confidence
            document.getElementById('confidenceScore').textContent = confidence.toFixed(1) + '%';

            // Build probability bars
            const probList = document.getElementById('probList');
            probList.innerHTML = '';

            // Sort by probability descending
            const sorted = Object.entries(allProbs).sort((a, b) => b[1] - a[1]);

            sorted.forEach(([className, prob]) => {
                const percentage = (prob * 100).toFixed(1);
                const item = document.createElement('div');
                item.className = 'prob-item';
                item.innerHTML = `
                    <div class="prob-name">${className}</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar" style="width: ${percentage}%; background: ${getClassColor(className)}"></div>
                    </div>
                    <div class="prob-value">${percentage}%</div>
                `;
                probList.appendChild(item);
            });
        }

        function getClassColor(className) {
            const colors = {
                'Bacterialblight': '#dc2626',
                'Brownspot': '#ea580c',
                'Healthy': '#16a34a',
                'Leafsmut': '#2563eb',
                'Rice Blast': '#9333ea'
            };
            return colors[className] || '#0ea5e9';
        }
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

    try:
        # Read and preprocess image
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

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'EfficientNetB3',
        'accuracy': '95.23%',
        'classes': CLASS_NAMES
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  RICE LEAF DISEASE DETECTION - Web Interface")
    print("="*60)
    print("  Model: EfficientNetB3")
    print("  Accuracy: 95.23%")
    print("  Classes: 5 (Bacterialblight, Brownspot, Healthy,")
    print("                    Leafsmut, Rice Blast)")
    print("="*60)
    print("  Server starting at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")

    app.run(debug=False, port=5000, host='0.0.0.0')
