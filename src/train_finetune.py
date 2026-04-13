import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


TRAIN_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_PATH = "models/best_model.h5"


def load_datasets():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )

    return train_gen, val_gen


def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=predictions)
    
    for layer in base.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base


def main():
    print("="*50)
    print("FINE-TUNING TRAINING")
    print("="*50)
    
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    train_gen, val_gen = load_datasets()
    classes = list(train_gen.class_indices.keys())
    num_classes = len(classes)
    
    print(f"Classes: {classes}")
    print(f"Train: {train_gen.samples}, Val: {val_gen.samples}")
    
    model, base = build_model(num_classes)
    
    print("\nPhase 1: Training classifier head...")
    cb = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=cb, verbose=1)
    
    print("\nPhase 2: Fine-tuning base model...")
    for layer in base.layers:
        layer.trainable = True
    
    model.compile(optimizer=Adam(0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history2 = model.fit(train_gen, epochs=20, validation_data=val_gen, callbacks=cb, verbose=1)
    
    print("\n" + "="*50)
    print(f"Phase 1 Best Val Accuracy: {max(history.history['val_accuracy'])*100:.1f}%")
    print(f"Phase 2 Best Val Accuracy: {max(history2.history['val_accuracy'])*100:.1f}%")
    print("="*50)


if __name__ == "__main__":
    main()