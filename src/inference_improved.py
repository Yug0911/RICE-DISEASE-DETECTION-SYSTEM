import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


MODEL_PATH = "models/rice_disease_7class_final.h5"
IMG_SIZE = (300, 300)

# Must match training class order
CLASS_NAMES = ['Healthy', 'Insect', 'Leaf Scald', 'Rice Blast', 
               'Rice Leaffolder', 'Rice Stripes', 'Rice Tungro']

DISEASE_INFO = {
    'Healthy': {
        'description': 'No disease symptoms. Leaf is healthy.',
        'recommendation': 'Continue regular monitoring. Maintain proper fertilization.'
    },
    'Insect': {
        'description': 'Insect damage detected. Shows feeding marks or holes.',
        'recommendation': 'Use appropriate insecticides. Introduce biological controls like ladybugs.'
    },
    'Leaf Scald': {
        'description': 'Leaf scald disease. Brown lesions with yellow halos, typically elongated.',
        'recommendation': 'Apply fungicide (e.g., azoxystrobin). Reduce nitrogen fertilization. Ensure proper water drainage.'
    },
    'Rice Blast': {
        'description': 'Rice blast disease. Diamond-shaped gray lesions with dark borders.',
        'recommendation': 'Apply blast-specific fungicides (tricyclazole, isoprothiolane). Avoid excess nitrogen. Use resistant varieties.'
    },
    'Rice Leaffolder': {
        'description': 'Rice leaffolder damage. Leaves are folded/wrapped by larval feeding.',
        'recommendation': 'Use biological control (Trichogramma). Apply targeted insecticides if infestation > threshold.'
    },
    'Rice Stripes': {
        'description': 'Rice stripe disease. Yellow to white striping along leaves.',
        'recommendation': 'Control leafhopper vectors. Remove infected plants early. Use resistant varieties.'
    },
    'Rice Tungro': {
        'description': 'Rice tungro disease. Yellow-orange discoloration and severe stunting.',
        'recommendation': 'Control brown planthopper vectors. Remove and destroy infected plants. Use tungro-resistant varieties.'
    }
}


def load_trained_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_disease(model, image_path):
    original_img, processed_img = preprocess_image(image_path)
    
    predictions = model.predict(processed_img, verbose=0)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nTop Prediction: {CLASS_NAMES[top_indices[0]]}")
    print(f"Confidence: {predictions[top_indices[0]]*100:.2f}%")
    
    info = DISEASE_INFO.get(CLASS_NAMES[top_indices[0]], DISEASE_INFO['Healthy'])
    print(f"\nDescription: {info['description']}")
    print(f"Recommendation: {info['recommendation']}")
    
    print("\nAll Predictions (Top 3):")
    print("-"*40)
    for i, idx in enumerate(top_indices):
        cls = CLASS_NAMES[idx]
        prob = predictions[idx] * 100
        marker = "▶" if i == 0 else " "
        print(f"  {marker} {cls}: {prob:.2f}%")
    print("="*60)
    
    return CLASS_NAMES[top_indices[0]], predictions[top_indices[0]]


def main():
    print("="*60)
    print("RICE LEAF DISEASE DETECTION - INFERENCE")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    
    model = load_trained_model()
    
    print("\nEnter image path for prediction, or 'quit' to exit.")
    while True:
        image_path = input("\nImage path: ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print(f"Error: File not found - {image_path}")
            continue
        
        try:
            predict_disease(model, image_path)
        except Exception as e:
            print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()