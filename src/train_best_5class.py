"""
BEST MODEL - 5-Class Balanced Dataset (with class weights)
Optimized for 5-class with 5983 total images
Strategy:
- EfficientNetB3 (good capacity for 5 classes)
- Moderate augmentation (dataset already large enough)
- Class weights (to further balance minority classes)
- Two-phase training
- Standard 300px resolution
"""
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = "Rice Dataset/Processed_5class"
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 100
PHASE1_EPOCHS = 40
MODEL_PATH = "models/best_5class.h5"
RESULTS_DIR = "results_5class"

Path("models").mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

def load_data():
    print("Loading 5-class balanced dataset...")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, 'train'),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True, seed=42
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, 'val'),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, 'test'),
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    print(f"Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
    return train_gen, val_gen, test_gen, list(train_gen.class_indices.keys())

def build_model(num_classes):
    print("\nBuilding EfficientNetB3...")
    base_model = EfficientNetB3(
        weights='imagenet', include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    model = Sequential([
        Input(shape=(*IMG_SIZE, 3)),
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='swish'),
        Dropout(0.3),
        Dense(256, activation='swish'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def fine_tune(model, unfreeze_layers=120):
    print(f"\nFine-tuning: unfreezing top {unfreeze_layers} layers...")
    # Get the base model (EfficientNetB3) - it's the first non-input layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # EfficientNetB3 is a Model
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Could not find base model in architecture")
    
    base_model.trainable = True
    # Freeze all base layers first, then selectively unfreeze top N
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("="*70)
print("BEST MODEL - 5-Class Balanced Dataset")
print("="*70)

train_gen, val_gen, test_gen, class_names = load_data()
num_classes = len(class_names)

# Compute class weights to handle imbalance
print("\nComputing class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=train_gen.classes
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

print(f"\nPhase 1: Train head only for {PHASE1_EPOCHS} epochs, LR=0.001...")
model = build_model(num_classes)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

h1 = model.fit(train_gen, epochs=PHASE1_EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1, class_weight=class_weight_dict)

print("\nPhase 2: Fine-tune top 120 layers, LR=1e-5...")
model = fine_tune(model, unfreeze_layers=120)
h2 = model.fit(train_gen, epochs=EPOCHS-PHASE1_EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1, class_weight=class_weight_dict)

history = {
    'accuracy': h1.history['accuracy'] + h2.history['accuracy'],
    'val_accuracy': h1.history['val_accuracy'] + h2.history['val_accuracy'],
    'loss': h1.history['loss'] + h2.history['loss'],
    'val_loss': h1.history['val_loss'] + h2.history['val_loss']
}

# Save history for later analysis
import json, pickle
with open(f"{RESULTS_DIR}/training_history.pkl", 'wb') as f:
    pickle.dump(history, f)
print(f"\n[OK] Training history saved to {RESULTS_DIR}/training_history.pkl")

print("\n" + "="*70)
print("FINAL EVALUATION ON TEST SET")
print("="*70)

y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_true, y_pred_classes)

print(f"\n{'='*70}")
print(f"TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*70}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Plot training curves
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(history['accuracy'], label='Train', linewidth=2)
ax[0].plot(history['val_accuracy'], label='Val', linewidth=2)
ax[0].set_title('Training Accuracy', fontsize=12)
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[0].grid(True, alpha=0.3)
ax[1].plot(history['loss'], label='Train', linewidth=2)
ax[1].plot(history['val_loss'], label='Val', linewidth=2)
ax[1].set_title('Training Loss', fontsize=12)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()
ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=12)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Accuracy:")
for i, (name, acc_val) in enumerate(zip(class_names, cm_norm.diagonal())):
    print(f"  {name:<20}: {acc_val*100:>6.2f}%")

# Summary
train_final = history['accuracy'][-1]
val_final = history['val_accuracy'][-1]
gap = train_final - val_final

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Final Train Accuracy : {train_final*100:.2f}%")
print(f"Final Val Accuracy   : {val_final*100:.2f}%")
print(f"Train-Val Gap        : {gap*100:.1f}%")
print(f"Test Accuracy        : {acc*100:.2f}%")
print(f"Best Val Accuracy    : {max(history['val_accuracy'])*100:.2f}% (epoch {np.argmax(history['val_accuracy'])+1})")
print(f"{'='*70}")
print(f"\n[OK] Model saved: {MODEL_PATH}")
print(f"[OK] Results saved to: {RESULTS_DIR}/")
