import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

data = load_pickle("hyperparam_search_results_copy.pkl")
data = [record for record in data 
                   if record['hidden_size1'] <= 2048
                   and record['hidden_size2'] <= 2048]
print(len(data))
best_record = max(data, key=lambda x: x['val_acc'])
print(best_record)


filtered_records = [record for record in data 
                   if record['lr'] == 0.05
                   and record['hidden_size1'] == 2048
                   and record['hidden_size2'] == 2048]

filtered_records.sort(key=lambda x: x['reg'])
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(filtered_records)))
for i, record in enumerate(filtered_records):
    h1 = record["reg"]
    epochs = range(1, len(record['train_loss_history']) + 1)
    axes[0].plot(epochs, record['train_loss_history'], 'o-', color=colors[i], label=f'beta_reg={h1}')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(epochs, record['val_loss_history'], 'o-', color=colors[i], label=f'beta_reg={h1}')
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    axes[2].plot(epochs, record['val_acc_history'], 'o-', color=colors[i], label=f'beta_reg={h1}')
    axes[2].set_title('Validation Accuracy')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy')
    axes[2].grid(True, linestyle='--', alpha=0.7)

axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'learning_rate_comparison.png'")
plt.show()