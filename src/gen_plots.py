"""
Regenerate training curves from saved history
"""
import pickle, os, matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = "results_5class"
HISTORY_PATH = "results_5class/training_history.pkl"

if not Path(HISTORY_PATH).exists():
    print("History file not found. Training history must be saved separately.")
    print("The training script needs to be modified to save history.")
    # Check if we can reconstruct from any saved data
    import json
    if Path("results_5class/metrics.json").exists():
        with open("results_5class/metrics.json") as f:
            metrics = json.load(f)
        print("Found metrics.json, but training script doesn't save it.")
else:
    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history['accuracy'], label='Train', linewidth=2)
    ax[0].plot(history['val_accuracy'], label='Val', linewidth=2)
    ax[0].set_title('Training Accuracy', fontsize=12)
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy'); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(history['loss'], label='Train', linewidth=2)
    ax[1].plot(history['val_loss'], label='Val', linewidth=2)
    ax[1].set_title('Training Loss', fontsize=12)
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss'); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved to {RESULTS_DIR}/training_curves.png")
