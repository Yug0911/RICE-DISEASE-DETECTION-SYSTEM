# Rice Leaf Disease Detection System

A deep learning-based system for detecting and classifying rice leaf diseases from images using Convolutional Neural Networks (CNNs).

## 🌾 Project Overview

This system helps farmers and agricultural experts identify rice leaf diseases early to prevent crop damage and yield losses. The model can classify rice leaves into 8 categories including healthy leaves and various disease types.

### Diseases Detected
- Healthy
- Insect Damage
- Leaf Scald
- Rice Blast
- Rice Leaffolder
- Rice Stripes
- Rice Tungro
- Rice (Unclassified)

## 📊 Dataset

| Dataset | Images | Classes |
|---------|--------|---------|
| Augmented Training | ~15,000 | 8 |
| Original Test | ~4,000 | 8 |

## 🏗️ Model Architecture

```
Input: 224×224×3 RGB Image
       ↓
EfficientNetB4 (Pre-trained on ImageNet)
       ↓
GlobalAveragePooling2D
       ↓
Dense(1024) → BatchNorm → ReLU → Dropout(0.5)
Dense(512) → BatchNorm → ReLU → Dropout(0.4)
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
       ↓
Output: Softmax (8 classes)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- CUDA (optional for GPU)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd DEEP_LEARNING

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python main.py
# Select option 1
```

### Evaluation

```bash
python main.py
# Select option 2
```

### Inference

```bash
python main.py
# Select option 3
# Enter image path when prompted
```

## 📁 Project Structure

```
DEEP_LEARNING/
├── main.py                    # Main runner
├── requirements.txt           # Dependencies
├── Rice Dataset/
│   ├── Original Dataset/     # Test data
│   └── Augmented Dataset/   # Training data
├── src/
│   ├── train_best.py       # Training script
│   ├── evaluate_final.py    # Evaluation script
│   └── inference_improved.py # Prediction script
├── models/
│   └── best_rice_disease.h5 # Trained model
└── results/
    ├── confusion_matrix.png
    └── training_curves.png
```

## 📈 Usage Examples

### Command Line Inference
```
Image path: Rice Dataset/Original Dataset/Healthy/h1.jpg
```

### Python API
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('models/best_rice_disease.h5')
img = load_img('image.jpg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print(f"Disease: {class_names[np.argmax(prediction)]}")
```

## 🎯 Results

- **Training Accuracy**: ~98%+
- **Test Accuracy**: Depends on data quality
- **Inference Time**: <100ms per image

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| IMG_SIZE | (224, 224) | Input image size |
| BATCH_SIZE | 32 | Training batch size |
| EPOCHS | 60 | Maximum epochs |
| LEARNING_RATE | 0.0001 | Optimizer learning rate |

## 📝 Disease Information

| Disease | Description | Recommendation |
|---------|-------------|----------------|
| Healthy | No symptoms | Continue monitoring |
| Insect | Feeding damage | Use insecticides |
| Leaf Scald | Brown lesions | Apply fungicide |
| Rice Blast | Diamond lesions | Use resistant varieties |
| Rice Leaffolder | Folded leaves | Biological control |
| Rice Stripes | Yellow stripes | Control vectors |
| Rice Tungro | Orange discoloration | Remove infected plants |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License

## 👨‍💻 Author

- YUG BHAVSAR
- SEM 6, DEEP LEARNING

## 🙏 Acknowledgments

- Dataset: Custom rice field images
- Pretrained Model: EfficientNet (ImageNet)
- Framework: TensorFlow/Keras