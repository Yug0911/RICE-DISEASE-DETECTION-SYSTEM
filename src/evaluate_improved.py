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
from sklearn.metrics import precision_recall_fscore_support


MODEL_PATH = "models/improved_rice_disease.h5"
TEST_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (224, 224)
RESULTS_DIR = "results"


def load_test_data():
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator


def evaluate_model(model, test_generator):
    print("Evaluating model on test set...")
    y_true = test_generator.classes
    y_pred = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f"\n*** TEST ACCURACY: {accuracy*100:.2f}% ***\n")
    
    return y_true, y_pred, y_pred_classes, accuracy


def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title('Confusion Matrix - Rice Leaf Disease Detection', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title('Normalized Confusion Matrix (Per Class Accuracy)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_normalized.png", dpi=150, bbox_inches='tight')
    plt.close()


def print_detailed_metrics(y_true, y_pred_classes, class_names):
    print("\n" + "="*70)
    print("DETAILED PER-CLASS METRICS")
    print("="*70)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None
    )
    
    print(f"\n{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for i, name in enumerate(class_names):
        print(f"{name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    print("-" * 70)
    
    print("\n" + classification_report(y_true, y_pred_classes, target_names=class_names))


def main():
    print("="*60)
    print("RICE LEAF DISEASE - IMPROVED MODEL EVALUATION")
    print("="*60)
    
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    print("\nLoading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    
    print("\nLoading test data...")
    test_generator = load_test_data()
    class_names = list(test_generator.class_indices.keys())
    print(f"Classes: {class_names}")
    print(f"Test samples: {test_generator.samples}")
    
    print("\nEvaluating...")
    y_true, y_pred, y_pred_classes, accuracy = evaluate_model(model, test_generator)
    
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    
    print("\nGenerating metrics...")
    print_detailed_metrics(y_true, y_pred_classes, class_names)
    
    print("\n" + "="*60)
    print(f"FINAL TEST ACCURACY: {accuracy*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()