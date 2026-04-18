"""
Generate final training curves from available logs
Reconstructs history from training_output.log
"""
import re, matplotlib.pyplot as plt

log_path = "training_output.log"
with open(log_path) as f:
    lines = f.readlines()

# Parse epoch data from log
# Pattern: "Epoch 1/40" then lines with "accuracy: 0.2500 - loss: 2.0184" per batch
# And epoch end: "131/131 [==...] - accuracy: 0.8349 - loss: 0.5523 - val_accuracy: 0.8940 - val_loss: 0.2570"
epoch_pattern = re.compile(r'Epoch (\d+)/(\d+)')
batch_pattern = re.compile(r'\d+/\d+\s+\[.*?\]\s+-\s+accuracy: ([\d.]+)\s+-\s+loss: ([\d.]+)')
val_pattern = re.compile(r'(\d+)/131.*?-\s+accuracy: ([\d.]+)\s+-\s+loss: ([\d.]+)\s+-\s+val_accuracy: ([\d.]+)\s+-\s+val_loss: ([\d.]+)')

train_acc = []
train_loss = []
val_acc = []
val_loss = []
current_epoch = 0

for line in lines:
    # Check for epoch start
    ep_match = epoch_pattern.search(line)
    if ep_match:
        current_epoch = int(ep_match.group(1))
        if current_epoch <= 40:
            print(f"Epoch {current_epoch}/40 (Phase 1)")
    
    # Check for epoch end line (contains val metrics)
    if current_epoch > 0 and 'val_accuracy:' in line:
        val_match = re.search(r'accuracy: ([\d.]+).*loss: ([\d.]+).*val_accuracy: ([\d.]+).*val_loss: ([\d.]+)', line)
        if val_match:
            train_acc.append(float(val_match.group(1)))
            train_loss.append(float(val_match.group(2)))
            val_acc.append(float(val_match.group(3)))
            val_loss.append(float(val_match.group(4)))
            print(f"  Epoch {current_epoch}: train_acc={train_acc[-1]:.4f}, val_acc={val_acc[-1]:.4f}")

print(f"\nTotal epochs parsed: {len(train_acc)}")

if len(train_acc) > 0:
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(train_acc, label='Train', linewidth=2)
    ax[0].plot(val_acc, label='Val', linewidth=2)
    ax[0].set_title('Training Accuracy (Phase 1)', fontsize=12)
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy'); ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(train_loss, label='Train', linewidth=2)
    ax[1].plot(val_loss, label='Val', linewidth=2)
    ax[1].set_title('Training Loss (Phase 1)', fontsize=12)
    ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss'); ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results_5class/training_curves_phase1.png", dpi=150)
    plt.close()
    print("\n[OK] Training curves saved to results_5class/training_curves_phase1.png")
    
    # Print summary
    best_epoch = max(range(len(val_acc)), key=lambda i: val_acc[i])
    print(f"\nPhase 1 Summary:")
    print(f"  Best Val Accuracy: {val_acc[best_epoch]*100:.2f}% at epoch {best_epoch+1}")
    print(f"  Final Train Acc: {train_acc[-1]*100:.2f}%")
    print(f"  Final Val Acc: {val_acc[-1]*100:.2f}%")
else:
    print("No training metrics found in log")
