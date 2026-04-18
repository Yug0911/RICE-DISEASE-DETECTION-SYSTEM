"""
Fast evaluation - minimal verbose output
"""
import os, numpy as np, tensorflow as tf, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, seaborn as sns

PROCESSED_DIR = "Rice Dataset/Processed_5class"
IMG_SIZE = (300, 300); BATCH_SIZE = 32; MODEL_PATH = "models/best_5class.h5"; RESULTS_DIR = "results_5class"

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    f"{PROCESSED_DIR}/test", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

class_names = list(test_gen.class_indices.keys())
model = tf.keras.models.load_model(MODEL_PATH)

y_true = test_gen.classes
y_pred = model.predict(test_gen, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_true, y_pred_classes)

print(f"\nTEST ACCURACY: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150); plt.close()
print(f"\nConfusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nPer-Class Accuracy:")
for i, (name, acc_val) in enumerate(zip(class_names, cm_norm.diagonal())):
    print(f"  {name:<20}: {acc_val*100:>6.2f}%")
