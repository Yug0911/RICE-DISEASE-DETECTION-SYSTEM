import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "models/final_optimized_model.h5"
TRAIN_DIR = "Rice Dataset/Original Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def evaluate():
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded: {MODEL_PATH}")
    
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    val_gen = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )
    
    classes = list(val_gen.class_indices.keys())
    print(f"\nClasses: {classes}")
    print(f"Validation samples: {val_gen.samples}")
    
    results = model.evaluate(val_gen, verbose=1)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]*100:.2f}%")
    print("="*50)
    
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = val_gen.classes
    
    print("\nClassification Report:")
    print(classification_report(true_classes, pred_classes, target_names=classes))


if __name__ == "__main__":
    evaluate()