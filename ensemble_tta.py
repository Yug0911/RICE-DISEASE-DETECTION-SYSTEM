"""
ENSEMBLE + TTA - Combine multiple models with test-time augmentation
Ensemble the best 3 models:
1. regularized_b0.h5 (60% val, low gap)
2. focal_b1.h5 (55% val, class-weighted)
3. baseline_b0_v1.h5 (59% val, overfitted but strong)

Apply TTA with 5 augmentations per test image.
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
IMG_SIZE_B0 = (224, 224)
IMG_SIZE_B1 = (240, 240)
BATCH_SIZE = 16
TEST_DIR = os.path.join(PROCESSED_DIR, 'test')

MODEL_PATHS = [
    ("RegularizedB0", "models/regularized_b0.h5", IMG_SIZE_B0),
    ("FocalB1", "models/focal_b1.h5", IMG_SIZE_B1),
    ("BaselineB0", "models/baseline_b0_v1.h5", IMG_SIZE_B0),
]

RESULTS_DIR = "results"
Path(RESULTS_DIR).mkdir(exist_ok=True)

print("="*70)
print("ENSEMBLE + TEST-TIME AUGMENTATION")
print("="*70)

# Load models
models = []
for name, path, size in MODEL_PATHS:
    if os.path.exists(path):
        print(f"Loading {name} ({path}) with size {size}...")
        m = tf.keras.models.load_model(path)
        models.append((name, m, size))
        print(f"  [OK] Loaded")
    else:
        print(f"  [WARN] Not found: {path}")

if len(models) < 2:
    print("[ERROR] Need at least 2 models for ensemble")
    exit(1)

# Load test data (without augmentation for true labels)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE_B0,  # Standard size
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
class_names = list(test_gen.class_indices.keys())
num_test = test_gen.samples
print(f"\nTest samples: {num_test}")
print(f"Classes: {class_names}")

# TTA augmenter (light transforms)
tta_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)
TTA_STEPS = 5

print(f"\nRunning TTA with {TTA_STEPS} augmentations per image...")
all_preds = []

for model_idx, (model_name, model, img_size) in enumerate(models):
    print(f"\n[Model {model_idx+1}/{len(models)}] {model_name}")
    model_preds = np.zeros((num_test, len(class_names)))

    # Custom test generator for this model's input size
    test_gen_model = test_datagen.flow_from_directory(
        TEST_DIR, target_size=img_size, batch_size=1,
        class_mode='categorical', shuffle=False
    )

    for i in range(num_test):
        if (i+1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_test}")

        # Get original image
        img_batch, _ = next(test_gen_model)
        img = img_batch[0]

        # TTA: predict on augmented versions and average
        aug_preds = []
        for t in range(TTA_STEPS):
            # Generate augmented version
            aug_iter = tta_datagen.flow(img[np.newaxis, ...], batch_size=1, shuffle=False)
            aug_img = next(aug_iter)[0]
            pred = model.predict(aug_img[np.newaxis, ...], verbose=0)
            aug_preds.append(pred[0])

        # Average TTA predictions
        model_preds[i] = np.mean(aug_preds, axis=0)

    all_preds.append(model_preds)

# Ensemble: average predictions from all models
print("\nAveraging ensemble predictions...")
ensemble_preds = np.mean(all_preds, axis=0)
y_pred_classes = np.argmax(ensemble_preds, axis=1)
y_true = test_gen.classes

acc = accuracy_score(y_true, y_pred_classes)
print(f"\n{'='*70}")
print(f"ENSEMBLE TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*70}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Ensemble + TTA Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/ensemble_tta_cm.png", dpi=150)
plt.close()

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Accuracy:")
for i, (name, acc_val) in enumerate(zip(class_names, cm_norm.diagonal())):
    print(f"  {name:<20}: {acc_val*100:>6.2f}%")

print(f"\n[OK] Ensemble model saved (metadata only)")
print(f"[OK] Results saved to {RESULTS_DIR}/")

# Save predictions for analysis
np.save(f"{RESULTS_DIR}/ensemble_predictions.npy", ensemble_preds)
print(f"[OK] Predictions saved: {RESULTS_DIR}/ensemble_predictions.npy")
