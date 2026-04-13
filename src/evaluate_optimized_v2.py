import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "models/optimized_rice_disease_v2.h5"
TRAIN_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32


def evaluate_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
    
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    val_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    class_names = list(val_generator.class_indices.keys())
    print(f"\nClasses: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Validation samples: {val_generator.samples}")
    
    print("\nEvaluating model...")
    results = model.evaluate(val_generator, verbose=1)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]*100:.2f}%")
    print("="*50)
    
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    cm = confusion_matrix(true_classes, predicted_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('results/optimized_v2_confusion_matrix_v2.png', dpi=150)
    plt.close()
    print("\nConfusion matrix saved to results/optimized_v2_confusion_matrix_v2.png")


if __name__ == "__main__":
    evaluate_model()