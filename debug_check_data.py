import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

TRAIN_DIR = "Rice Dataset/Original Dataset"

# Quick sanity check
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=(300,300), batch_size=4,
    class_mode='categorical', subset='training', shuffle=True, seed=42
)

print(f"Class indices: {train_gen.class_indices}")
print(f"Number of samples: {train_gen.samples}")

# Get one batch
x, y = next(train_gen)
print(f"\nBatch shape: {x.shape} (should be 4x300x300x3)")
print(f"Labels shape: {y.shape} (should be 4x7)")
print(f"Label example: {y[0]}")
print(f"Label argmax: {np.argmax(y[0])} -> {list(train_gen.class_indices.keys())[np.argmax(y[0])]}")

# Check pixel values
print(f"\nPixel range: min={x.min():.3f}, max={x.max():.3f} (should be 0-1 after rescale)")
print(f"Sample pixel values (first image, top-left 3x3):")
print(x[0, :3, :3, :])

# Check if images are all same (corrupted/blank)
print(f"\nImage variance (first batch):")
for i in range(4):
    print(f"  Image {i}: mean={x[i].mean():.3f}, std={x[i].std():.3f}")

# Check visualization quickly
import matplotlib.pyplot as plt
plt.figure(figsize=(12,3))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(x[i])
    plt.title(list(train_gen.class_indices.keys())[np.argmax(y[i])])
    plt.axis('off')
plt.savefig('debug_batch.png', dpi=100, bbox_inches='tight')
print("\nSaved debug_batch.png - check if images show actual rice leaves")