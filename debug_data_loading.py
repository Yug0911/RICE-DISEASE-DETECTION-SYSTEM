import os
from collections import Counter

train_dir = "Rice Dataset/Original Dataset"

# Check class distribution
print("Class distribution in Original Dataset:")
for cls in sorted(os.listdir(train_dir)):
    cls_path = os.path.join(train_dir, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {cls}: {count} images")

print("\n" + "="*60)
print("CHECKING if validation split is working correctly...")
print("="*60)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Replicate what load_datasets does
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"\nTraining samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Total: {train_gen.samples + val_gen.samples}")

# Check class distribution in train and val
train_labels = train_gen.classes
val_labels = val_gen.classes

print("\nTraining class distribution:")
train_dist = Counter(train_labels)
for cls_idx, count in sorted(train_dist.items()):
    print(f"  Class {cls_idx} ({list(train_gen.class_indices.keys())[cls_idx]}): {count}")

print("\nValidation class distribution:")
val_dist = Counter(val_labels)
for cls_idx, count in sorted(val_dist.items()):
    print(f"  Class {cls_idx} ({list(val_gen.class_indices.keys())[cls_idx]}): {count}")

# Check if any class missing from val
print("\n" + "="*60)
missing = set(train_dist.keys()) - set(val_dist.keys())
if missing:
    print(f"⚠️  WARNING: Classes missing from validation: {missing}")
else:
    print("✓ All classes present in both train and val")

# Check class balance
print("\nClass balance check:")
for cls_idx in train_dist:
    cls_name = list(train_gen.class_indices.keys())[cls_idx]
    train_pct = train_dist[cls_idx] / len(train_labels) * 100
    val_pct = val_dist.get(cls_idx, 0) / len(val_labels) * 100
    print(f"  {cls_name}: train={train_pct:.1f}%, val={val_pct:.1f}%")