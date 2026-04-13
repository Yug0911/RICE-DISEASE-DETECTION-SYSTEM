import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize


MODEL_PATH = "models/mobilenet_v2_rice_disease.h5"
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
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    return y_true, y_pred, y_pred_classes, accuracy


def print_detailed_metrics(y_true, y_pred_classes, class_names):
    print("\n" + "="*50)
    print("DETAILED METRICS")
    print("="*50)
    
    report = classification_report(y_true, y_pred_classes, 
                                  target_names=class_names, 
                                  output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None
    )
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        print(f"{name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    print("-" * 60)
    print(f"{'Macro Avg':<20} {report['macro avg']['precision']:<12.4f} {report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f} {report['macro avg']['support']:<10}")
    print(f"{'Weighted Avg':<20} {report['weighted avg']['precision']:<12.4f} {report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f} {report['weighted avg']['support']:<10}")
    print("-" * 60)
    
    return report


def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
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
                annot_kws={"size": 14})
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_normalized.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Normalized confusion matrix saved to {RESULTS_DIR}/confusion_matrix_normalized.png")


def plot_roc_curves(y_true, y_pred, class_names):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {RESULTS_DIR}/roc_curves.png")


def save_metrics_to_file(report, accuracy, class_names):
    with open(f"{RESULTS_DIR}/evaluation_summary.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("RICE LEAF DISEASE DETECTION - EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write("-"*60 + "\n")
        
        for class_name in class_names:
            metrics = report[class_name]
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1-score']:.4f}\n")
            f.write(f"  Support:  {int(metrics['support'])}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("Macro Average:\n")
        f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {report['macro avg']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {report['macro avg']['f1-score']:.4f}\n")
        
        f.write("\nWeighted Average:\n")
        f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {report['weighted avg']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {report['weighted avg']['f1-score']:.4f}\n")
    
    print(f"Evaluation summary saved to {RESULTS_DIR}/evaluation_summary.txt")


def main():
    print("="*60)
    print("RICE LEAF DISEASE DETECTION - MODEL EVALUATION")
    print("="*60)
    
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    print("\nLoading model...")
    model = load_model(MODEL_PATH)
    
    print("\nLoading test data...")
    test_generator = load_test_data()
    class_names = list(test_generator.class_indices.keys())
    print(f"Classes: {class_names}")
    print(f"Test samples: {test_generator.samples}")
    
    print("\nEvaluating model...")
    y_true, y_pred, y_pred_classes, accuracy = evaluate_model(model, test_generator)
    
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    plot_roc_curves(y_true, y_pred, class_names)
    
    print("\nGenerating detailed metrics...")
    report = print_detailed_metrics(y_true, y_pred_classes, class_names)
    save_metrics_to_file(report, accuracy, class_names)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved to: {RESULTS_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()