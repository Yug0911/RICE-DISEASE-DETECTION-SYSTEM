"""
Recreate Processed_5class from Original Dataset with selected classes
Classes: Brownspot, Bacterialblight, Leafsmut, Healthy, Rice Blast
70/15/15 split
"""
import os
import shutil
from pathlib import Path
import random
import json

ORIGINAL_DIR = Path("Rice Dataset/Original Dataset")
PROCESSED_DIR = Path("Rice Dataset/Processed_5class")
SELECTED_CLASSES = ['Brownspot', 'Bacterialblight', 'Leafsmut', 'Healthy', 'Rice Blast']
SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED = 42

random.seed(SEED)

print("="*70)
print("CREATING BALANCED 5-CLASS DATASET")
print("="*70)
print(f"Selected classes: {SELECTED_CLASSES}")
print(f"Split ratio: 70% train, 15% val, 15% test")

# Remove old processed dir if exists
if PROCESSED_DIR.exists():
    shutil.rmtree(PROCESSED_DIR)
    print(f"\n[OK] Removed old dataset: {PROCESSED_DIR}")

# Create new directories
for split in ['train', 'val', 'test']:
    for class_name in SELECTED_CLASSES:
        (PROCESSED_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

# Get all image files for each class
class_images = {}
for class_name in SELECTED_CLASSES:
    class_dir = ORIGINAL_DIR / class_name
    if class_dir.exists():
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
        random.shuffle(images)
        class_images[class_name] = images
        print(f"  {class_name:<25}: {len(images):>4} images found")
    else:
        print(f"  [WARN] {class_name} directory not found!")

# Split each class
print(f"\n{'='*70}")
print("SPLITTING DATA")
print("="*70)

stats = {}
for class_name, images in class_images.items():
    n = len(images)
    n_train = int(n * SPLITS['train'])
    n_val = int(n * SPLITS['val'])
    n_test = n - n_train - n_val  # remainder

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]

    stats[class_name] = {'train': n_train, 'val': n_val, 'test': n_test}

    # Copy files
    for img in train_imgs:
        dest = PROCESSED_DIR / 'train' / class_name / img.name
        shutil.copy(img, dest)
    for img in val_imgs:
        dest = PROCESSED_DIR / 'val' / class_name / img.name
        shutil.copy(img, dest)
    for img in test_imgs:
        dest = PROCESSED_DIR / 'test' / class_name / img.name
        shutil.copy(img, dest)

    print(f"\n{class_name}:")
    print(f"  Train: {n_train:>4} ({n_train/n*100:.1f}%)")
    print(f"  Val:   {n_val:>4} ({n_val/n*100:.1f}%)")
    print(f"  Test:  {n_test:>4} ({n_test/n*100:.1f}%)")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
total_train = sum(s['train'] for s in stats.values())
total_val = sum(s['val'] for s in stats.values())
total_test = sum(s['test'] for s in stats.values())
total_all = total_train + total_val + total_test

print(f"\n{'Class':<20} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
print("-"*50)
for class_name in SELECTED_CLASSES:
    s = stats[class_name]
    print(f"{class_name:<20} {s['train']:>6} {s['val']:>6} {s['test']:>6} {s['train']+s['val']+s['test']:>7}")
print("-"*50)
print(f"{'TOTAL':<20} {total_train:>6} {total_val:>6} {total_test:>6} {total_all:>7}")

counts = [stats[c]['train'] for c in SELECTED_CLASSES]
balance_ratio = max(counts) / min(counts)
print(f"\nTrain set balance ratio (max/min): {balance_ratio:.2f}x")
print(f"Images per class (min): {min(counts)}")
print(f"Images per class (max): {max(counts)}")

# Save stats
with open(PROCESSED_DIR / 'dataset_info.json', 'w') as f:
    json.dump({
        'classes': SELECTED_CLASSES,
        'splits': SPLITS,
        'stats': stats,
        'total': total_all
    }, f, indent=2)

print(f"\n[OK] Dataset created at: {PROCESSED_DIR}")
print(f"[OK] Dataset info saved: {PROCESSED_DIR / 'dataset_info.json'}")
