"""
Phase 2 Only: Fine-tune the best Phase 1 model
Resumes from models/best_5class.h5 (already trained head)
"""
import os, numpy as np, tensorflow as tf, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

PROCESSED_DIR = "Rice Dataset/Processed_5class"
IMG_SIZE = (300, 300); BATCH_SIZE = 32; EPOCHS = 100
PHASE1_EPOCHS = 40  # We're starting at epoch 40 essentially
MODEL_PATH = "models/best_5class.h5"
RESULTS_DIR = "results_5class"
Path(RESULTS_DIR).mkdir(exist_ok=True)

def load_data():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, 'train'), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=True, seed=42
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, 'val'), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DIR, 'test'), target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )
    print(f"Train: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")
    return train_gen, val_gen, test_gen, list(train_gen.class_indices.keys())

def fine_tune(model, unfreeze_layers=120):
    print(f"\nFine-tuning: unfreezing top {unfreeze_layers} layers of EfficientNetB3...")
    # Find the base model (EfficientNetB3)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Base model not found")
    
    base_model.trainable = True
    # Freeze all, then unfreeze top N
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Load data
print("="*70)
print("PHASE 2 ONLY: Fine-tuning from Phase 1 checkpoint")
print("="*70)
train_gen, val_gen, test_gen, class_names = load_data()
num_classes = len(class_names)

# Compute class weights (same as Phase 1)
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=train_gen.classes)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# Load Phase 1 best model
print(f"\nLoading Phase 1 model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# Fine-tune
model = fine_tune(model, unfreeze_layers=120)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

print(f"\nFine-tuning for up to {EPOCHS-PHASE1_EPOCHS} epochs, LR=1e-5...")
h2 = model.fit(train_gen, epochs=EPOCHS-PHASE1_EPOCHS, validation_data=val_gen, callbacks=callbacks, verbose=1, class_weight=class_weight_dict)

# Since we don't have Phase 1 history, we'll just evaluate directly
print("\n" + "="*70)
print("FINAL EVALUATION ON TEST SET")
print("="*70)

y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_true, y_pred_classes)

print(f"\n{'='*70}")
print(f"TEST ACCURACY: {acc*100:.2f}%")
print(f"{'='*70}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150); plt.close()
print(f"\n[OK] Confusion matrix saved")

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Accuracy:")
for i, (name, acc_val) in enumerate(zip(class_names, cm_norm.diagonal())):
    print(f"  {name:<20}: {acc_val*100:>6.2f}%")

print(f"\n[OK] Model saved: {MODEL_PATH}")
print(f"[OK] Results saved to: {RESULTS_DIR}/")
