import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler


TRAIN_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "models/optimized_rice_disease_v2.h5"
RESULTS_DIR = "results"


def create_directories():
    Path("models").mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)


def load_datasets():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
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

    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)

    return train_generator, val_generator, class_names, num_classes


def build_model(num_classes):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = True
    for layer in base_model.layers[:50]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='o')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def cosine_decay(epoch):
    import math
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * epoch / EPOCHS))


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
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        LearningRateScheduler(cosine_decay, verbose=1)
    ]


def main():
    print("="*60)
    print("OPTIMIZED RICE LEAF DISEASE DETECTION V2")
    print("Using EfficientNetB0 for higher accuracy")
    print("="*60)
    print(f"Start Time: {datetime.now()}")
    print("="*60)

    create_directories()

    print("\n[1/4] Loading datasets...")
    train_gen, val_gen, class_names, num_classes = load_datasets()
    
    print(f"\nClasses: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    print("\n[2/4] Building optimized model...")
    model = build_model(num_classes)
    model.summary()

    print("\n[3/4] Training model...")
    callbacks = get_callbacks()
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[4/4] Saving training curves...")
    plot_training_history(history, f"{RESULTS_DIR}/optimized_v2_training_curves.png")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()