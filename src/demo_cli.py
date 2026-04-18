"""
Rice Leaf Disease Detection - Simple CLI Demo
Quick prediction on any image file
"""
import os
import sys
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = "models/best_5class.h5"
IMG_SIZE = (300, 300)
CLASS_NAMES = ['Bacterialblight', 'Brownspot', 'Healthy', 'Leafsmut', 'Rice Blast']

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded. Ready for predictions.\n")
print("Classes:", CLASS_NAMES)
print("\nUsage: python demo_cli.py <image_path>")
print("   Or: python demo_cli.py (then enter path interactively)\n")

def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array, verbose=0)[0]
    idx = np.argmax(preds)
    
    print("\n" + "="*50)
    print(f"IMAGE: {os.path.basename(img_path)}")
    print("="*50)
    print(f"\nPrediction : {CLASS_NAMES[idx]}")
    print(f"Confidence : {preds[idx]*100:.1f}%")
    print("\nAll Classes:")
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, preds)):
        bar = '█' * int(p*30) + '░' * (30 - int(p*30))
        print(f"  {cls:<20}: {bar} {p*100:>5.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter image path: ").strip().strip('"')
    
    if os.path.exists(img_path):
        predict(img_path)
    else:
        print(f"❌ File not found: {img_path}")
