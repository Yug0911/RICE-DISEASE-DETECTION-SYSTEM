"""
TINY CNN - Minimal model to probe if data has any learnable signal
If this also overfits/underperforms, the dataset itself is the problem
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
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROCESSED_DIR = "Rice Dataset/Processed_7class"
IMG_SIZE = (150, 150)  # Small images
BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = "models/tiny_cnn.h5"
RESULTS_DIR = "results"
Path("models").mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

def load_data():
    print("Loading data with moderate augmentation...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
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
    print("\nBuilding Tiny CNN (< 100k params)...")
    model = Sequential([
        Input(shape=(*IMG_SIZE, 3)),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.2),

        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.3),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.4),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dropout(0.5),

        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

print("="*70)
print("TINY CNN PROBE - Is there any learnable signal?")
print("="*70)

train_gen, val_gen, test_gen, class_names = load_data()
num_classes = len(class_names)

model = build_model(num_classes)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\nTraining...")
h = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1)

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
ax[0].plot(h.history['accuracy'], label='Train')
ax[0].plot(h.history['val_accuracy'], label='Val')
ax[0].set_title('Accuracy'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy')
ax[0].legend(); ax[0].grid(True, alpha=0.3)
ax[1].plot(h.history['loss'], label='Train')
ax[1].plot(h.history['val_loss'], label='Val')
ax[1].set_title('Loss'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss')
ax[1].legend(); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/tiny_cnn_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/tiny_cnn_cm.png", dpi=150)
plt.close()

print(f"\n✓ Model saved: {MODEL_PATH}")
print(f"✓ Final val accuracy: {h.history['val_accuracy'][-1]*100:.2f}%")
print(f"\n{'='*70}")
print("DIAGNOSIS:")
if h.history['val_accuracy'][-1] < 0.55:
    print("⚠️  VALIDATION ACCURACY < 55% → Dataset lacks discriminative features")
    print("   Possible causes: Wrong labels, high intra-class variance,")
    print("   inter-class similarity, images too noisy/blurry")
elif h.history['accuracy'][-1] - h.history['val_accuracy'][-1] > 0.15:
    print("⚠️  >15% TRAIN-VAL GAP → Severe overfitting even with tiny model")
    print("   Dataset is too small or has label noise")
else:
    print("✓ Baseline reasonable. Consider this model or slight tuning.")
