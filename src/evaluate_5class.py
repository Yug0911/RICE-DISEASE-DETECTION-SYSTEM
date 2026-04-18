"""
Evaluate the trained 5-class model on the test set
"""
import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Config
PROCESSED_DIR = "Rice Dataset/Processed_5class"
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
MODEL_PATH = "models/best_5class.h5"
RESULTS_DIR = "results_5class"

Path(RESULTS_DIR).mkdir(exist_ok=True)

# Load test data
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = test_datagen.flow_from_directory(
    f"{PROCESSED_DIR}/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())
print(f"Classes: {class_names}")

# Load model
print(f"\nLoading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Evaluate
print("\nPredicting on test set...")
y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Accuracy
acc = accuracy_score(y_true, y_pred_classes)
print(f"\n{'='*70}")
print(f"TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*70}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=12)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
print(f"\n[OK] Confusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")
plt.close()

# Per-class accuracy
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Accuracy:")
for i, (name, acc_val) in enumerate(zip(class_names, cm_norm.diagonal())):
    print(f"  {name:<20}: {acc_val*100:>6.2f}%")

# Also compute and print train/val history from training if available
# (the model checkpoint doesn't store history, but we can infer from model performance)
print("\n[OK] Evaluation complete.")
