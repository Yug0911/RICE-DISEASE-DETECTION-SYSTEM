import os
import shutil
from pathlib import Path

# Fix class name mismatch between datasets
aug_dir = "Rice Dataset/Augmented Dataset/Part-1/After Augmentation"
orig_dir = "Rice Dataset/Original Dataset"

# Map augmented class names to match original
class_mapping = {
    'Leaf scald': 'Leaf Scald',
    'rice': 'Rice'
}

print("Fixing class name mismatches in augmented dataset...")

for old_name, new_name in class_mapping:
    old_path = Path(aug_dir) / old_name
    new_path = Path(aug_dir) / new_name
    if old_path.exists():
        if new_path.exists():
            # Merge files from old to new
            for f in old_path.glob('*'):
                shutil.move(str(f), str(new_path))
            old_path.rmdir()
        else:
            old_path.rename(new_path)
        print(f"  Renamed: {old_name} -> {new_name}")

print("\nClass names now aligned!")
print("Augmented classes:", list(Path(aug_dir).iterdir()))
print("Original classes:", list(Path(orig_dir).iterdir()))