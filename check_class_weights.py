import os
import numpy as np
from collections import Counter

# Check train distribution
train_dir = "Rice Dataset/Processed_7class/train"
counts = {}
for cls in os.listdir(train_dir):
    cls_path = os.path.join(train_dir, cls)
    if os.path.isdir(cls_path):
        n = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        counts[cls] = n

print("Training class distribution:")
total = sum(counts.values())
for cls, count in counts.items():
    pct = count / total * 100
    weight = total / (len(counts) * count)
    print(f"  {cls:<20}: {count:>4} images ({pct:>5.1f}%) -> weight={weight:.2f}x")

print(f"\nTotal: {total} images")
print("\nClass weights applied by compute_class_weight('balanced'):")
from sklearn.utils.class_weight import compute_class_weight
# Simulate getting labels
labels = []
for cls_idx, cls in enumerate(sorted(counts.keys())):
    labels.extend([cls_idx] * counts[cls])
labels = np.array(labels)
classes = np.unique(labels)
weights = compute_class_weight('balanced', classes=classes, y=labels)
for i, (cls, w) in enumerate(zip(sorted(counts.keys()), weights)):
    print(f"  {cls:<20}: {w:.2f}x")