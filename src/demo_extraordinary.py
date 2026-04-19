"""
Rice Leaf Disease Detection — Extraordinary Web Interface
Award-winning design with glassmorphism, mesh gradients, and fluid animations
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

# Configuration
MODEL_PATH = "models/best_5class.h5"
IMG_SIZE = (300, 300)
CLASS_NAMES = ['Bacterialblight', 'Brownspot', 'Healthy', 'Leafsmut', 'Rice Blast']

# Sophisticated color palette (dark mode aesthetic)
COLORS = {
    'primary': '#3b82f6',
    'primary-dark': '#2563eb',
    'accent': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'dark': '#0a0a0a',
    'dark-surface': '#141414',
    'dark-card': '#1a1a1a',
    'text': '#f3f4f6',
    'text-secondary': '#9ca3af',
    'border': '#2a2a2a',
}

# Class-specific gradient colors
CLASS_GRADIENTS = {
    'Bacterialblight': 'linear-gradient(135deg, #dc2626, #b91c1c)',
    'Brownspot': 'linear-gradient(135deg, #ea580c, #c2410c)',
    'Healthy': 'linear-gradient(135deg, #16a34a, #15803d)',
    'Leafsmut': 'linear-gradient(135deg, #2563eb, #1d4ed8)',
    'Rice Blast': 'linear-gradient(135deg, #9333ea, #7c3aed)',
}

# Load model
print("Initializing Rice Leaf Disease Detection System...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded. Accuracy: 95.23%\n")

# HTML Template with extraordinary design
HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Leaf Disease Detection | AI-Powered Diagnosis</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --accent: #8b5cf6;
            --success: #10b981;
            --dark: #0a0a0a;
            --dark-surface: #141414;
            --dark-card: #1a1a1a;
            --text: #f3f4f6;
            --text-secondary: #9ca3af;
            --border: #2a2a2a;
            --glass: rgba(26, 26, 26, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--dark);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
            line-height: 1.6;
        }

        /* Animated Background */
        .bg-gradient {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(ellipse at 20% 0%, rgba(59, 130, 246, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 40% 50%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
            z-index: -1;
            animation: gradientShift 20s ease infinite alternate;
        }

        @keyframes gradientShift {
            0% { transform: scale(1) rotate(0deg); }
            100% { transform: scale(1.1) rotate(2deg); }
        }

        /* Navigation */
        nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            background: rgba(10, 10, 10, 0.8);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
        }

        .nav-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1.25rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-family: 'Playfair Display', serif;
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .nav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
        }

        .nav-link {
            color: var(--text-secondary);
            text-decoration: none;
            font-weight: 500;
            font-size: 0.95rem;
            transition: color 0.3s;
            cursor: pointer;
        }

        .nav-link:hover {
            color: var(--primary);
        }

        .accuracy-badge {
            background: linear-gradient(135deg, var(--success), #059669);
            color: white;
            padding: 0.5rem 1.25rem;
            border-radius: 30px;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
        }

        /* Main Container */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 8rem 2rem 4rem;
        }

        /* Hero Section */
        .hero {
            text-align: center;
            margin-bottom: 4rem;
            animation: fadeInUp 1s ease-out;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .hero h1 {
            font-family: 'Playfair Display', serif;
            font-size: 4.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text) 0%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.5rem;
            line-height: 1.1;
            word-wrap: break-word;
            overflow-wrap: break-word;
            max-width: 100%;
        }

        .hero h1 span {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: var(--text-secondary);
            max-width: 800px;
            margin: 0 auto 3rem;
            font-weight: 300;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            max-width: 1000px;
            margin: 0 auto 4rem;
        }

        @media (max-width: 968px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .hero h1 {
                font-size: 3rem;
            }
        }

        .stat-card {
            background: var(--glass);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            transition: transform 0.3s, border-color 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary);
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }

        .stat-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
        }

        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 2rem;
            align-items: start;
        }

        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Cards */
        .card {
            background: var(--glass);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2.5rem;
            transition: transform 0.4s, box-shadow 0.4s;
        }

        .card:hover {
            transform: translateY(-8px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }

.card-icon {
            width: 56px;
            height: 56px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 700;
            color: white;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 700;
            color: white;
        }

        .upload-icon {
            font-size: 3rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 1rem;
        }

        .placeholder-icon {
            font-size: 4rem;
            font-weight: 600;
            color: var(--primary);
            opacity: 0.3;
            margin-bottom: 1rem;
        }

        .card-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text);
        }

        /* Upload Zone */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: 20px;
            padding: 4rem 2rem;
            text-align: center;
            background: rgba(59, 130, 246, 0.03);
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            border-color: transparent;
        }

        .upload-zone:hover {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.08);
            transform: translateY(-2px);
        }

        .upload-zone.dragover {
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.15);
            transform: scale(1.01);
            box-shadow: 0 20px 40px rgba(59, 130, 246, 0.2);
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
            font-size: 5rem;
            margin-bottom: 1.5rem;
            opacity: 0.7;
        }

        .upload-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.75rem;
        }

        .upload-subtitle {
            font-size: 1rem;
            color: var(--text-secondary);
            max-width: 400px;
            margin: 0 auto;
            line-height: 1.6;
        }

        .file-info {
            margin-top: 1.5rem;
            padding: 1rem 1.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            display: none;
            color: var(--success);
            font-weight: 500;
        }

        /* Preview */
        .preview-container {
            margin-top: 2rem;
            text-align: center;
            display: none;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .preview-image {
            max-width: 100%;
            max-height: 350px;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .remove-btn {
            margin-top: 1rem;
            padding: 0.75rem 2rem;
            background: transparent;
            color: var(--danger);
            border: 1px solid var(--danger);
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .remove-btn:hover {
            background: var(--danger);
            color: white;
            transform: translateY(-2px);
        }

        /* Predict Button */
        .predict-btn {
            width: 100%;
            padding: 1.5rem;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            border: none;
            border-radius: 14px;
            font-size: 1.125rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            margin-top: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            position: relative;
            overflow: hidden;
        }

        .predict-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.6s;
        }

        .predict-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(59, 130, 246, 0.4);
        }

        .predict-btn:hover::before {
            left: 100%;
        }

        .predict-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            background: var(--border);
        }

        .spinner {
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

        /* Results */
        .results-container {
            display: none;
            animation: slideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(30px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .placeholder {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
        }

        .placeholder-icon {
            font-size: 5rem;
            opacity: 0.2;
            margin-bottom: 1rem;
        }

        .result-header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .prediction-badge {
            display: inline-block;
            padding: 1rem 2.5rem;
            background: var(--glass);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            font-weight: 700;
            font-size: 2rem;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s;
        }

        .confidence-label {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.75rem;
            font-weight: 500;
        }

        .confidence-score {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--success), #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Probability Bars */
        .prob-list {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .prob-item {
            display: grid;
            grid-template-columns: 160px 1fr 70px;
            align-items: center;
            gap: 1.25rem;
            padding: 1.25rem 1.5rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            border: 1px solid transparent;
            transition: all 0.3s;
        }

        .prob-item:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: var(--primary);
            transform: translateX(5px);
        }

        .prob-name {
            font-weight: 600;
            color: var(--text);
            font-size: 0.95rem;
        }

        .prob-bar-container {
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }

        .prob-bar {
            height: 100%;
            border-radius: 5px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        .prob-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
            animation: shimmer 2s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .prob-value {
            text-align: right;
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--text);
        }

        /* Info Section */
        .info-section {
            margin-top: 3rem;
            padding: 2rem;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
        }

        .info-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: var(--text);
        }

        .info-list {
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }

        .info-list li {
            padding: 1rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            border-left: 3px solid var(--primary);
            color: var(--text-secondary);
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 3rem 2rem;
            margin-top: 4rem;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .footer-content {
            max-width: 800px;
            margin: 0 auto;
        }

        .footer-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 6rem 1rem 2rem;
            }
            .hero h1 {
                font-size: 2.5rem;
            }
            .stats-grid {
                gap: 1rem;
            }
            .stat-card {
                padding: 1.5rem;
            }
            .card {
                padding: 1.5rem;
            }
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: var(--dark);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary);
        }
    </style>
</head>
<body>
    <div class="bg-gradient"></div>

    <!-- Navigation -->
    <!-- Simplified Navigation -->

    <!-- Main Content -->
    <div class="container">
        <!-- Hero -->
        <div class="hero">
            <h1>Intelligent <span>Rice Disease</span> Detection</h1>
            <p class="hero-subtitle">
                Advanced deep learning system for accurate identification of rice leaf diseases.
            </p>
        </div>

        <!-- Main Interface -->
        <div class="main-grid">
            <!-- Upload Card -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">Upload</div>
                    <h2 class="card-title">Upload Leaf Image</h2>
                </div>

                <div class="upload-zone" id="uploadZone">
                    <input type="file" id="fileInput" accept="image/jpeg,image/png,image/jpg,image/bmp,image/tiff">
                    <div class="upload-icon">Image</div>
                    <div class="upload-title">Drop leaf image or click to browse</div>
                    <div class="upload-subtitle">
                        Supports JPG, PNG, BMP, TIFF. For best results, use clear close-up photos of rice leaves.
                    </div>
                    <div class="file-info" id="fileInfo"></div>
                </div>

                <div class="preview-container" id="previewContainer">
                    <img id="previewImg" class="preview-image" src="" alt="Leaf preview">
                    <button class="remove-btn" onclick="removeImage()">Remove</button>
                </div>

                <button class="predict-btn" id="predictBtn" onclick="predict()" disabled>
                    <span>Analyze Disease</span>
                </button>

                <div class="info-section">
                    <h3 class="info-title">System Overview</h3>
                    <ul class="info-list">
                        <li>EfficientNetB3 architecture with 10.2M parameters</li>
                        <li>Trained on 5,983 balanced rice leaf images</li>
                        <li>Class-weighted loss for imbalanced data</li>
                        <li>Two-phase training: head-only then fine-tuning</li>
                        <li>All processing local — images not stored</li>
                        <li>Test accuracy: 95.23% | F1-macro: 0.880</li>
                    </ul>
                </div>
            </div>

            <!-- Results Card -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">Results</div>
                    <h2 class="card-title">Prediction Results</h2>
                </div>

                <div id="placeholder" class="placeholder">
                    <div class="placeholder-icon">Results</div>
                    <p>Upload an image to receive diagnosis</p>
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
        <div class="footer-content">
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
                predictBtn.textContent = 'Analyze Disease';
            }
        }

        function displayResults(data) {
            placeholder.style.display = 'none';
            resultsContainer.style.display = 'block';

            const prediction = data.prediction;
            const confidence = data.confidence;
            const allProbs = data.all_probabilities;

            // Prediction badge with gradient
            const badge = document.getElementById('predictionBadge');
            badge.textContent = prediction;
            badge.style.background = getClassGradient(prediction);

            // Confidence
            document.getElementById('confidenceScore').textContent = confidence.toFixed(1) + '%';

            // Probability bars (sorted descending)
            const probList = document.getElementById('probList');
            probList.innerHTML = '';

            const sorted = Object.entries(allProbs).sort((a, b) => b[1] - a[1]);

            sorted.forEach(([className, prob]) => {
                const percentage = (prob * 100).toFixed(1);
                const item = document.createElement('div');
                item.className = 'prob-item';
                item.innerHTML = `
                    <div class="prob-name">${className}</div>
                    <div class="prob-bar-container">
                        <div class="prob-bar" style="width: ${percentage}%; background: ${getClassGradient(className)}"></div>
                    </div>
                    <div class="prob-value">${percentage}%</div>
                `;
                probList.appendChild(item);
            });
        }

        function getClassGradient(className) {
            const colors = {
                'Bacterialblight': 'linear-gradient(90deg, #dc2626, #b91c1c)',
                'Brownspot': 'linear-gradient(90deg, #ea580c, #c2410c)',
                'Healthy': 'linear-gradient(90deg, #16a34a, #15803d)',
                'Leafsmut': 'linear-gradient(90deg, #2563eb, #1d4ed8)',
                'Rice Blast': 'linear-gradient(90deg, #9333ea, #7c3aed)'
            };
            return colors[className] || 'linear-gradient(90deg, #3b82f6, #2563eb)';
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

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  RICE LEAF DISEASE DETECTION — Extraordinary Web Interface")
    print("="*70)
    print("  Model: EfficientNetB3")
    print("  Accuracy: 95.23% | Classes: 5")
    print("  Server: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(debug=False, port=5000, host='0.0.0.0')
