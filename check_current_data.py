"""
Check current dataset distribution after adding new data
"""
import os
from pathlib import Path
from collections import defaultdict

# Check both original and processed
original_dir = Path("Rice Dataset/Original Dataset")
processed_dir = Path("Rice Dataset/Processed_7class")

print("="*70)
print("CURRENT DATASET STATUS")
print("="*70)

# Check original dataset
if original_dir.exists():
    print("\n[ORIGINAL DATASET]")
    class_counts = {}
    for class_dir in sorted(original_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*')))
            class_counts[class_dir.name] = count
            print(f"  {class_dir.name:<25}: {count:>4} images")

    total = sum(class_counts.values())
    print(f"\n  Total: {total} images")
    if class_counts:
        min_class = min(class_counts, key=class_counts.get)
        max_class = max(class_counts, key=class_counts.get)
        print(f"  Smallest class: {min_class} ({class_counts[min_class]} imgs)")
        print(f"  Largest class: {max_class} ({class_counts[max_class]} imgs)")
        print(f"  Imbalance ratio: {class_counts[max_class]/class_counts[min_class]:.2f}x")

# Check processed splits
if processed_dir.exists():
    print("\n[PROCESSED SPLITS (7-class)]")
    for split in ['train', 'val', 'test']:
        split_path = processed_dir / split
        if split_path.exists():
            print(f"\n  {split.upper()}:")
            split_total = 0
            for class_dir in sorted(split_path.iterdir()):
                if class_dir.is_dir():
                    count = len(list(class_dir.glob('*')))
                    split_total += count
                    print(f"    {class_dir.name:<23}: {count:>4} images")
            print(f"    {'Total':<23}: {split_total:>4} images")

print("\n" + "="*70)
print("NEXT STEPS:")
print("1. If classes are now balanced (>300 per class) → retrain")
print("2. If some classes still <100 → consider removing them")
print("="*70)
