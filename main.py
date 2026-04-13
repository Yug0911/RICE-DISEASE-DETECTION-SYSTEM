"""
Rice Leaf Disease Detection - Main Runner
==========================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("="*60)
    print("RICE LEAF DISEASE DETECTION SYSTEM")
    print("="*60)
    print("\nSelect option:")
    print("1. Train BEST Model (EfficientNetB4 + TTA)")
    print("2. Evaluate Model")
    print("3. Run Inference")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        print("\nStarting best model training...")
        from src import train_best
        train_best.main()
    elif choice == '2':
        print("\nStarting evaluation...")
        from src import evaluate_final
        evaluate_final.main()
    elif choice == '3':
        print("\nStarting inference...")
        from src import inference_improved
        inference_improved.main()
    else:
        print("Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()