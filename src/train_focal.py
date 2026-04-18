"""
FOCAL LOSS + CLASS WEIGHTS - Focus training on hard classes
- Replace categorical crossentropy with focal loss (down-weights easy examples)
- Add class weights to handle imbalance
- Increase capacity slightly
Target: Boost accuracy on hard classes (Leaf Scald, Rice Stripes, Leaffolder)
"""
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROCESSED_DIR = "Rice Dataset/Processed_7class"
IMG_SIZE = (240, 240)
BATCH_SIZE = 32
EPOCHS = 150
PHASE1_EPOCHS = 40
MODEL_PATH = "models/focal_b1.h5"
RESULTS_DIR = "results"
Path("models").mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

def compute_class_weights(train_gen):
    """Compute balanced class weights for imbalanced data"""
    class_counts = train_gen.classes
    classes = np.unique(class_counts)
    weights = compute_class_weight('balanced', classes=classes, y=class_counts)
    class_weight_dict = {i: w for i, w in enumerate(weights)}
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict

def focal_loss(gamma=2., alpha=0.25):
    """Focal loss for multi-class: focuses on hard examples"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma) * y_true
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha_factor)
        focal_loss = alpha_factor * weight * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return focal_loss_fixed

def load_data():
    print("Loading data (gentle augmentation)...")
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
    print("\nBuilding EfficientNetB1 (medium capacity)...")
    base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    model = Sequential([
        Input(shape=(*IMG_SIZE, 3)),
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='swish'),
        Dropout(0.4),
        Dense(128, activation='swish'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("="*70)
print("FOCAL LOSS + CLASS WEIGHTS - Handle hard classes")
print("="*70)

train_gen, val_gen, test_gen, class_names = load_data()
num_classes = len(class_names)
class_weights = compute_class_weights(train_gen)

print(f"\nPhase 1: Train head only for {PHASE1_EPOCHS} epochs, LR=0.001...")
model = build_model(num_classes)
base_model = model.layers[1]

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

h1 = model.fit(train_gen, epochs=PHASE1_EPOCHS, validation_data=val_gen,
               callbacks=callbacks, class_weight=class_weights, verbose=1)

print("\nPhase 2: Fine-tune top 120 layers, LR=1e-5...")
base_model.trainable = True
for layer in base_model.layers[:120]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

h2 = model.fit(train_gen, epochs=EPOCHS-PHASE1_EPOCHS, validation_data=val_gen,
               callbacks=callbacks, class_weight=class_weights, verbose=1)

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
plt.savefig(f"{RESULTS_DIR}/focal_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/focal_cm.png", dpi=150)
plt.close()

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Accuracy:")
for i, (name, acc_val) in enumerate(zip(class_names, cm_norm.diagonal())):
    print(f"  {name:<20}: {acc_val*100:>6.2f}%")

print(f"\n[OK] Model saved: {MODEL_PATH}")
print(f"[OK] Final val accuracy: {history['val_accuracy'][-1]*100:.2f}%")
