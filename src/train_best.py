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
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


TRAIN_DIR = "Rice Dataset/Augmented Dataset/Part-1/After Augmentation"
TEST_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "models/best_rice_disease.h5"
RESULTS_DIR = "results"


def create_directories():
    Path("models").mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)


def load_datasets():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1
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
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)

    return train_generator, val_generator, test_generator, class_names, num_classes


def build_best_model(num_classes):
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(1024, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.5)(x)
    
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


def predict_with_tta(model, image, img_gen, class_indices):
    """Test Time Augmentation for better predictions"""
    reverse_indices = {v: k for k, v in class_indices.items()}
    
    preds = []
    # Original
    img_array = image / 255.0
    preds.append(model.predict(img_array, verbose=0))
    
    # Horizontal flip
    img_flip = np.fliplr(img_array)
    preds.append(model.predict(img_flip, verbose=0))
    
    # Slight rotation
    for angle in [10, -10]:
        img_rot = tf.keras.preprocessing.image.apply_affine_transform(
            image, theta=angle, row_axis=0, col_axis=1, channel_axis=2
        )
        preds.append(model.predict(img_rot, verbose=0))
    
    # Zoom
    zoomed = tf.keras.preprocessing.image.apply_affine_transform(
        image, zoom_factor=1.1, row_axis=0, col_axis=1, channel_axis=2
    )
    preds.append(model.predict(zoomed, verbose=0))
    
    # Average predictions
    avg_pred = np.mean(preds, axis=0)
    return avg_pred, reverse_indices


def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.title('Confusion Matrix - Best Model', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/best_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"CM saved to {RESULTS_DIR}/best_confusion_matrix.png")


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train', marker='o', markersize=4)
    axes[0].plot(history.history['val_accuracy'], label='Val', marker='o', markersize=4)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train', marker='o', markersize=4)
    axes[1].plot(history.history['val_loss'], label='Val', marker='o', markersize=4)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("="*70)
    print("BEST RICE LEAF DISEASE MODEL - EFFICIENTNET B4 + TTA")
    print("="*70)
    print(f"Start: {datetime.now()}")
    print("="*70)

    create_directories()

    print("\n[1/4] Loading datasets...")
    train_gen, val_gen, test_gen, class_names, num_classes = load_datasets()
    
    print(f"\nTraining: {train_gen.samples} samples")
    print(f"Validation: {val_gen.samples} samples")  
    print(f"Test: {test_gen.samples} samples")
    print(f"Classes: {class_names}")

    print("\n[2/4] Building EfficientNetB4 model...")
    model = build_best_model(num_classes)
    print("EfficientNetB4 ready!")

    print("\n[3/4] Training...")
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[4/4] Evaluating...")
    plot_training_history(history, f"{RESULTS_DIR}/best_training_curves.png")
    
    # Standard evaluation
    y_true = test_gen.classes
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred_classes)
    
    print(f"\n{'='*70}")
    print(f"TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*70}")
    
    test_class_names = list(test_gen.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=test_class_names))
    
    plot_confusion_matrix(y_true, y_pred_classes, test_class_names)
    
    print(f"\nBest model saved: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()