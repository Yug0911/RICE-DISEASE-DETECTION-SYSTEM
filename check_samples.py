"""
Display sample images from each class to assess quality
"""
import os
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("Rice Dataset/Processed_7class")

fig, axes = plt.subplots(7, 5, figsize=(12, 16))
fig.suptitle("Sample Images from Each Class (5 per class)", fontsize=14)

class_names = ['Healthy', 'Insect', 'Leaf Scald', 'Rice Blast', 'Rice Leaffolder', 'Rice Stripes', 'Rice Tungro']

for row, class_name in enumerate(class_names):
    class_dir = PROCESSED_DIR / 'train' / class_name
    if class_dir.exists():
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        sample_imgs = random.sample(images, min(5, len(images)))

        for col in range(5):
            ax = axes[row, col]
            if col < len(sample_imgs):
                img = cv2.imread(str(sample_imgs[col]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(class_name, fontsize=10)
            else:
                ax.axis('off')

plt.tight_layout()
plt.savefig("results/sample_images_quality_check.png", dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Saved: results/sample_images_quality_check.png")
print("\nCheck this file to assess image quality, contrast, and class differences")
