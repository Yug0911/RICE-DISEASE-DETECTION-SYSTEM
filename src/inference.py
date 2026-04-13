import os
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


MODEL_PATH = "models/mobilenet_v2_rice_disease.h5"
IMG_SIZE = (224, 224)


DISEASE_INFO = {
    'Healthy': {
        'description': 'No disease symptoms detected. The rice leaf appears healthy.',
        'recommendation': 'Continue regular monitoring. Maintain proper fertilizer application.'
    },
    'Insect': {
        'description': 'Insect damage detected on the leaf.',
        'recommendation': 'Consider using appropriate insecticides. Monitor for pest infestation.'
    },
    'Leaf scald': {
        'description': 'Leaf scald disease detected. Symptoms include brown lesions with yellow halos.',
        'recommendation': 'Apply fungicide. Reduce nitrogen fertilization. Ensure proper water drainage.'
    },
    'Rice Blast': {
        'description': 'Rice blast disease detected. Characterized by diamond-shaped lesions.',
        'recommendation': 'Apply blast-specific fungicide. Avoid excessive nitrogen. Use resistant varieties.'
    },
    'Rice Leaffolder': {
        'description': 'Rice leaffolder damage detected. Leaves show folding and feeding damage.',
        'recommendation': 'Use biological control agents. Apply targeted insecticides if needed.'
    },
    'Rice Stripes': {
        'description': 'Rice stripe disease detected. Shows striping pattern on leaves.',
        'recommendation': 'Control vector insects (leafhoppers). Remove infected plants. Use resistant varieties.'
    },
    'Rice Tungro': {
        'description': 'Rice tungro disease detected. Shows yellow-orange discoloration.',
        'recommendation': 'Control brown planthopper vectors. Remove infected plants. Use resistant varieties.'
    },
    'rice': {
        'description': 'Unidentified rice leaf condition.',
        'recommendation': 'Consult agricultural expert for proper diagnosis and treatment.'
    }
}


def load_trained_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    return model


def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_disease(model, image_path, class_indices):
    reverse_indices = {v: k for k, v in class_indices.items()}
    
    original_img, processed_img = preprocess_image(image_path)
    
    predictions = model.predict(processed_img, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[::-1][:3]
    
    results = []
    for idx in top_indices:
        class_name = reverse_indices[idx]
        confidence = predictions[idx] * 100
        
        info = DISEASE_INFO.get(class_name, DISEASE_INFO['rice'])
        
        results.append({
            'class': class_name,
            'confidence': confidence,
            'description': info['description'],
            'recommendation': info['recommendation']
        })
    
    return results


def print_prediction_results(results):
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    print(f"\nTop Prediction: {results[0]['class']}")
    print(f"Confidence: {results[0]['confidence']:.2f}%")
    print(f"\nDescription: {results[0]['description']}")
    print(f"Recommendation: {results[0]['recommendation']}")
    
    print("\n" + "-"*60)
    print("All Predictions:")
    print("-"*60)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['class']}: {res['confidence']:.2f}%")
    print("="*60)


def predict_from_directory(model, directory, class_indices):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(directory).glob(f"*{ext}"))
        image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {directory}")
        return
    
    print(f"\nFound {len(image_files)} images to predict\n")
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        print("-"*40)
        results = predict_disease(model, str(img_path), class_indices)
        print_prediction_results(results)


def main():
    print("="*60)
    print("RICE LEAF DISEASE DETECTION - INFERENCE")
    print("="*60)
    
    model = load_trained_model()
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        "Rice Dataset/Original Dataset",
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    class_indices = test_generator.class_indices
    
    print("\nClass Indices:", class_indices)
    
    print("\nEnter image path for prediction, or 'quit' to exit.")
    while True:
        image_path = input("\nImage path: ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print(f"Error: File not found - {image_path}")
            continue
        
        try:
            results = predict_disease(model, image_path, class_indices)
            print_prediction_results(results)
        except Exception as e:
            print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()