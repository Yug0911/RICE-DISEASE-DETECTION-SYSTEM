"""
REGULARIZATION FOCUSED - Fix overfitting with strong regularization
Target: Close the train-val gap (currently 76% vs 59%)
Strategy:
- Moderate augmentation (not zero, not aggressive)
- Heavy dropout (0.6→0.4)
- L2 regularization
- Label smoothing
- Freeze more base layers initially
- Then gradual unfreezing
"""
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROCESSED_DIR = "Rice Dataset/Processed_7class"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 150
PHASE1_EPOCHS = 50
MODEL_PATH = "models/regularized_b0.h5"
RESULTS_DIR = "results"
Path("models").mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

def load_data():
    print("Loading data (moderate augmentation)...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
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
    return train_gen, val_gen, test_gen, list(train_gen.class_indices.keys())

def build_model(num_classes):
    print("\nBuilding EfficientNetB0 with strong regularization...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    model = Sequential([
        Input(shape=(*IMG_SIZE, 3)),
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.5),
        BatchNormalization(),
        Dense(128, activation='swish', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def fine_tune_gradual(model, base_model, unfreeze_layers=100):
    print(f"\nFine-tuning: unfreezing top {unfreeze_layers} layers...")
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("="*70)
print("REGULARIZED EfficientNetB0 - Strong Dropout + L2")
print("="*70)

train_gen, val_gen, test_gen, class_names = load_data()
num_classes = len(class_names)

print(f"\nPhase 1: Train head only for {PHASE1_EPOCHS} epochs, LR=0.001...")
model = build_model(num_classes)
base_model = model.layers[1]

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

h1 = model.fit(train_gen, epochs=PHASE1_EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1)

print("\nPhase 2: Fine-tune top 120 layers, LR=1e-5...")
model = fine_tune_gradual(model, base_model, unfreeze_layers=120)
h2 = model.fit(train_gen, epochs=EPOCHS-PHASE1_EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1)

history = {
    'accuracy': h1.history['accuracy'] + h2.history['accuracy'],
    'val_accuracy': h1.history['val_accuracy'] + h2.history['val_accuracy'],
    'loss': h1.history['loss'] + h2.history['loss'],
    'val_loss': h1.history['val_loss'] + h2.history['val_loss']
}

print("\n" + "="*70)
print("EVALUATION")
print("="*70)

y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_true, y_pred_classes)

print(f"\nTEST ACCURACY: {acc*100:.2f}%")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history['accuracy'], label='Train')
ax[0].plot(history['val_accuracy'], label='Val')
ax[0].set_title('Accuracy'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy')
ax[0].legend(); ax[0].grid(True, alpha=0.3)
ax[1].plot(history['loss'], label='Train')
ax[1].plot(history['val_loss'], label='Val')
ax[1].set_title('Loss'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss')
ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/regularized_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/regularized_cm.png", dpi=150)
plt.close()

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Accuracy:")
for i, (name, acc) in enumerate(zip(class_names, cm_norm.diagonal())):
    print(f"  {name:<20}: {acc*100:>6.2f}%")

train_final = history['accuracy'][-1]
val_final = history['val_accuracy'][-1]
gap = train_final - val_final
print(f"\n[OK] Model saved: {MODEL_PATH}")
print(f"[OK] Final Train Acc: {train_final*100:.2f}%, Val Acc: {val_final*100:.2f}%, Gap: {gap*100:.1f}%")
if gap > 0.15:
    print(f"[WARN] Gap > 15% ({gap*100:.1f}%) - still overfitting")
else:
    print(f"[OK] Gap <= 15% - good generalization")
