"""
Quick evaluation of the latest model (rice_disease_7class_final.h5)
"""
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = "Rice Dataset/Processed_7class"
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
MODEL_PATH = "models/rice_disease_7class_final.h5"
RESULTS_DIR = "results"

Path(RESULTS_DIR).mkdir(exist_ok=True)

print("Loading test data...")
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    os.path.join(PROCESSED_DIR, 'test'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)
print(f"Test samples: {test_gen.samples}")
print(f"Classes: {list(test_gen.class_indices.keys())}")

print(f"\nLoading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating...")
y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_true, y_pred_classes)

print(f"\n{'='*70}")
print(f"TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*70}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(test_gen.class_indices.keys())))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_gen.class_indices.keys(), yticklabels=test_gen.class_indices.keys())
plt.title('Confusion Matrix - Latest Model'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/latest_confusion_matrix.png", dpi=150)
plt.close()
print(f"\nConfusion matrix saved: {RESULTS_DIR}/latest_confusion_matrix.png")

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Accuracy:")
for i, (name, acc) in enumerate(zip(test_gen.class_indices.keys(), cm_norm.diagonal())):
    print(f"  {name:<20}: {acc*100:>6.2f}%")
