import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


MODEL_PATH = "models/best_rice_disease.h5"
IMG_SIZE = (224, 224)


DISEASE_INFO = {
    'Healthy': {
        'description': 'No disease symptoms. Leaf is healthy.',
        'recommendation': 'Continue regular monitoring. Maintain proper fertilization.'
    },
    'Insect': {
        'description': 'Insect damage detected. Shows feeding marks or holes.',
        'recommendation': 'Use appropriate insecticides. Introduce biological controls.'
    },
    'Leaf scald': {
        'description': 'Leaf scald disease. Brown lesions with yellow halos.',
        'recommendation': 'Apply fungicide. Reduce nitrogen. Ensure proper drainage.'
    },
    'Rice Blast': {
        'description': 'Rice blast disease. Diamond-shaped gray lesions.',
        'recommendation': 'Apply blast-specific fungicide. Avoid excess nitrogen. Use resistant varieties.'
    },
    'Rice Leaffolder': {
        'description': 'Rice leaffolder damage. Leaves are folded/wrapped.',
        'recommendation': 'Use biological control. Apply targeted insecticides if needed.'
    },
    'Rice Stripes': {
        'description': 'Rice stripe disease. Yellow to white striping on leaves.',
        'recommendation': 'Control leafhopper vectors. Remove infected plants.'
    },
    'Rice Tungro': {
        'description': 'Rice tungro disease. Yellow-orange discoloration and stunting.',
        'recommendation': 'Control brown planthopper vectors. Use resistant varieties.'
    },
    'rice': {
        'description': 'Rice leaf condition detected.',
        'recommendation': 'Consult agricultural expert for diagnosis.'
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
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        "Rice Dataset/Original Dataset",
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    class_indices = test_generator.class_indices
    reverse_indices = {v: k for k, v in class_indices.items()}
    
    original_img, processed_img = preprocess_image(image_path)
    
    predictions = model.predict(processed_img, verbose=0)[0]
    
    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx] * 100
    predicted_class = reverse_indices[top_idx]
    
    info = DISEASE_INFO.get(predicted_class, DISEASE_INFO['rice'])
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"\nPredicted Disease: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nDescription: {info['description']}")
    print(f"Recommendation: {info['recommendation']}")
    
    print("\nAll Class Probabilities:")
    print("-"*40)
    for idx in np.argsort(predictions)[::-1]:
        class_name = reverse_indices[idx]
        prob = predictions[idx] * 100
        print(f"  {class_name}: {prob:.2f}%")
    print("="*60)
    
    return predicted_class, confidence


def main():
    print("="*60)
    print("RICE LEAF DISEASE DETECTION - INFERENCE")
    print("="*60)
    
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