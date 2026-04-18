"""
BEST MODEL: No Overfitting - Train on Original + Strong Augmentation
====================================================================
Target: >85% accuracy with NO overfitting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# CRITICAL: Use ORIGINAL dataset for training (not augmented)
TRAIN_DIR = "Rice Dataset/Original Dataset"
TEST_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.00005
MODEL_SAVE_PATH = "models/non_overfitting_rice.h5"
RESULTS_DIR = "results"


def create_directories():
    Path("models").mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)


def load_datasets():
    """Strong augmentation ON-THE-FLY from original images only"""
    print("\nLoading datasets with strong on-the-fly augmentation...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='reflect',
        validation_split=0.2
    )
    
    # Test gets NO augmentation - real images only
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Classes: {class_names}")
    
    return train_gen, val_gen, test_gen, class_names, num_classes


def build_model(num_classes):
    """
    Build model with STRONG regularization to prevent overfitting
    Using EfficientNetB3 - good balance of capacity and regularization
    """
    print("\nBuilding EfficientNetB3 with strong regularization...")
    
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling=None
    )
    
    # Freeze most base layers - only train top layers
    base_model.trainable = True
    for layer in base_model.layers[:150]:
        layer.trainable = False
    
    # Custom head with STRONG dropout
    model = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        base_model,
        GlobalAveragePooling2D(),
        
        Dense(512, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        tf.keras.layers.Activation('swish'),
        Dropout(0.6),
        
        Dense(256, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        tf.keras.layers.Activation('swish'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Use SGD with momentum for better generalization
    optimizer = SGD(
        learning_rate=LEARNING_RATE,
        momentum=0.9,
        nesterov=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # Patience for real improvement
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], 'b-', label='Train')
    axes[0].plot(history.history['val_accuracy'], 'r-', label='Val')
    axes[0].set_title('Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    axes[1].plot(history.history['loss'], 'b-', label='Train')
    axes[1].plot(history.history['val_loss'], 'r-', label='Val')
    axes[1].set_title('Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved")


def main():
    print("="*70)
    print("NON-OVERFITTING RICE DISEASE MODEL")
    print("="*70)
    print(f"Start: {datetime.now()}")
    print("Strategy: Train on ORIGINAL images only + strong on-the-fly augmentation")
    print("="*70)
    
    create_directories()
    
    print("\n[1/4] Loading datasets...")
    train_gen, val_gen, test_gen, class_names, num_classes = load_datasets()
    
    print("\n[2/4] Building model...")
    model = build_model(num_classes)
    print("Model built with EfficientNetB3 + strong dropout")
    
    print("\n[3/4] Training...")
    callbacks = get_callbacks()
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n[4/4] Evaluating...")
    plot_history(history, f"{RESULTS_DIR}/training_curves.png")
    
    # Evaluate
    y_true = test_gen.classes
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred_classes)
    
    print(f"\n{'='*70}")
    print(f"TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*70}")
    
    print("\nPer-class performance:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Normalized Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    
    # Check for overfitting
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    gap = (train_acc - val_acc) * 100
    
    print("\n" + "="*70)
    print("OVERFITTING CHECK:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Val Accuracy:   {val_acc*100:.2f}%")
    print(f"  Gap:            {gap:.2f}%")
    if gap > 10:
        print("  WARNING: Still overfitting!")
    else:
        print("  ✓ Good generalization")
    print("="*70)
    
    print(f"\nModel saved: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()