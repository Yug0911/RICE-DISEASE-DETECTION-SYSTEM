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
    print("1. Train 7-Class Model (with class weights)")
    print("2. Train Simple Baseline (no class weights)")
    print("3. Evaluate Model")
    print("4. Run Inference")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        print("\nStarting 7-class training (with class weights)...")
        from src import train_7class
        train_7class.main()
    elif choice == '2':
        print("\nStarting simple baseline (no class weights, moderate aug)...")
        from src import train_simple
        train_simple.main()
    elif choice == '3':
        print("\nStarting evaluation...")
        from src import evaluate_final
        evaluate_final.main()
    elif choice == '4':
        print("\nStarting inference...")
        from src import inference_improved
        inference_improved.main()
    else:
        print("Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()