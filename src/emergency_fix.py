"""
EMERGENCY FIX: Fast-learning model to escape 30% accuracy trap
"""
import os
import numpy as np
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

TRAIN_DIR = "Rice Dataset/Original Dataset"
TEST_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (224, 224)  # Smaller for speed
BATCH_SIZE = 32  # Larger batch
EPOCHS = 50
LEARNING_RATE = 0.001  # 20x HIGHER
MODEL_SAVE_PATH = "models/emergency_fix.h5"
RESULTS_DIR = "results"

Path("models").mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

def load_data():
    # Strong but fast augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    
    print(f"Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
    
    return train_gen, val_gen, test_gen, list(train_gen.class_indices.keys())

def build_model(num_classes):
    # EfficientNetB0 - lighter, faster to train
    base_model = EfficientNetB0(
        weights='imagenet', include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = True  # Unfreeze ALL layers
    
    model = Sequential([
        Input(shape=(*IMG_SIZE, 3)),
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  # Simpler head
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # HIGH learning rate to actually learn
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("="*60)
print("EMERGENCY FIX MODEL - Escape 30% accuracy trap")
print("="*60)

train_gen, val_gen, test_gen, class_names = load_data()
model = build_model(len(class_names))
model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\nTraining with LR=0.001 (20x higher than before)...")
history = model.fit(
    train_gen, epochs=EPOCHS, validation_data=val_gen,
    callbacks=callbacks, verbose=1
)

print("\nEvaluating...")
y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_true, y_pred_classes)

print(f"\n{'='*60}")
print(f"TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*60}")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

print(f"\nModel saved: {MODEL_SAVE_PATH}")
print("If accuracy is still <50%, your data has fundamental issues.")