import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


MODEL_PATH = "models/rice_disease_7class_final.h5"
TEST_DIR = "Rice Dataset/Processed_7class/test"
IMG_SIZE = (300, 300)
RESULTS_DIR = "results"


def main():
    print("="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    print("\nLoading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded!")
    
    print("\nLoading test data...")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    class_names = list(test_gen.class_indices.keys())
    print(f"Classes: {class_names}")
    print(f"Test samples: {test_gen.samples}")
    
    print("\nEvaluating...")
    y_true = test_gen.classes
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred_classes)
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")


if __name__ == "__main__":
    main()