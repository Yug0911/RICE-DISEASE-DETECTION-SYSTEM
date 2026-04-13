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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


TRAIN_DIR = "Rice Dataset/Augmented Dataset/Part-1/After Augmentation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.00005
MODEL_SAVE_PATH = "models/optimized_rice_disease.h5"
RESULTS_DIR = "results"


def create_directories():
    Path("models").mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)


def load_datasets():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.4,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        validation_split=0.15
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)

    return train_generator, val_generator, test_generator, class_names, num_classes


def compute_class_weights(train_generator, num_classes):
    from collections import Counter
    counter = Counter(train_generator.classes)
    total = sum(counter.values())
    class_weights = {cls: total / (num_classes * count) for cls, count in counter.items()}
    print("\nClass Weights:")
    for cls, weight in class_weights.items():
        print(f"  {cls}: {weight:.2f}")
    return class_weights


def build_optimized_model(num_classes):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.6)(x)
    
    x = Dense(512, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o', markersize=3)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='o', markersize=3)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train Loss', marker='o', markersize=3)
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='o', markersize=3)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
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


def main():
    print("="*70)
    print("OPTIMIZED RICE LEAF DISEASE DETECTION - HIGH ACCURACY MODEL")
    print("="*70)
    print(f"Start Time: {datetime.now()}")
    print("Using ResNet50 with stronger architecture")
    print("="*70)

    create_directories()

    print("\n[1/5] Loading datasets...")
    train_gen, val_gen, test_gen, class_names, num_classes = load_datasets()
    
    print(f"\nClasses: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    
    print("\nClass distribution in training:")
    for class_name, idx in train_gen.class_indices.items():
        count = sum(1 for label in train_gen.labels if label == idx)
        print(f"  {class_name}: {count}")

    print("\n[2/5] Computing class weights for imbalance...")
    class_weights = compute_class_weights(train_gen, num_classes)

    print("\n[3/5] Building optimized model...")
    model = build_optimized_model(num_classes)
    print("Model built with ResNet50 backbone + custom classifier")

    print("\n[4/5] Training model...")
    callbacks = get_callbacks()
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[5/5] Evaluating on test set...")
    y_true = test_gen.classes
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred_classes)
    
    print("\n" + "="*70)
    print(f"TEST ACCURACY: {accuracy*100:.2f}%")
    print("="*70)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 11})
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    plot_training_history(history, f"{RESULTS_DIR}/training_curves.png")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()