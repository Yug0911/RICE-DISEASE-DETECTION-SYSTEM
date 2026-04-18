"""
Analyze dataset distribution and class imbalance
"""
import os
from pathlib import Path

PROCESSED_DIR = Path("Rice Dataset/Processed_7class")

print("="*70)
print("CLASS DISTRIBUTION")
print("="*70)

total = 0
for split in ['train', 'val', 'test']:
    split_path = PROCESSED_DIR / split
    if split_path.exists():
        print(f"\n{split.upper()}:")
        split_total = 0
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*')))
                split_total += count
                print(f"  {class_dir.name:<25}: {count:>4} images")
        print(f"  {'Total':<25}: {split_total:>4} images")
        total += split_total

print(f"\n{'TOTAL DATASET':<25}: {total:>4} images")
print(f"\nClass imbalance ratio (max/min): ", end="")
class_counts = {}
for split in ['train', 'val', 'test']:
    split_path = PROCESSED_DIR / split
    if split_path.exists():
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                class_counts[class_dir.name] = class_counts.get(class_dir.name, 0) + len(list(class_dir.glob('*')))

if class_counts:
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    print(f"{max_count}/{min_count} = {max_count/min_count:.2f}x")
