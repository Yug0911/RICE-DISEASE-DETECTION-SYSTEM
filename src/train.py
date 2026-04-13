import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


TRAIN_DIR = "Rice Dataset/Augmented Dataset/Part-1/After Augmentation"
TEST_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "models/mobilenet_v2_rice_disease.h5"
RESULTS_DIR = "results"


def create_directories():
    Path("models").mkdir(exist_ok=True)
    Path(RESULTS_DIR).mkdir(exist_ok=True)


def load_datasets():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
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
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)

    return train_generator, val_generator, test_generator, class_names, num_classes


def build_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    for layer in base_model.layers[:130]:
        layer.trainable = False
    for layer in base_model.layers[130:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model


def get_callbacks():
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
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


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def evaluate_model(model, test_generator, class_names):
    print("\n" + "="*50)
    print("EVALUATING MODEL ON TEST SET")
    print("="*50)

    y_true = test_generator.classes
    y_pred = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred_classes)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    plot_confusion_matrix(y_true, y_pred_classes, class_names, 
                         f"{RESULTS_DIR}/confusion_matrix.png")

    return y_true, y_pred, accuracy


def main():
    print("="*50)
    print("RICE LEAF DISEASE DETECTION - TRAINING PIPELINE")
    print("="*50)
    print(f"Start Time: {datetime.now()}")
    print(f"TensorFlow Version: {tf.__version__}")
    print("="*50)

    create_directories()

    print("\n[1/5] Loading datasets...")
    train_gen, val_gen, test_gen, class_names, num_classes = load_datasets()
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")

    print("\n[2/5] Building model...")
    model = build_model(num_classes)
    model.summary()

    print("\n[3/5] Training model...")
    callbacks = get_callbacks()
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[4/5] Saving training curves...")
    plot_training_history(history, f"{RESULTS_DIR}/training_curves.png")

    print("\n[5/5] Evaluating model...")
    y_true, y_pred, accuracy = evaluate_model(model, test_gen, class_names)

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("="*50)


if __name__ == "__main__":
    main()