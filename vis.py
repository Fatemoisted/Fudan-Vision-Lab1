import matplotlib.pyplot as plt
import numpy as np
import pickle

model_weight_path = "best_model.pkl"
with open(model_weight_path, "rb") as f:
    model_weight = pickle.load(f)
W1 = model_weight['W1']  # (3072, 2048)

image_height = 32
image_width = 32
image_channels = 3

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
vmin, vmax = W1.min(), W1.max()
for i, ax in enumerate(axes.ravel()):
    if i < min(16, W1.shape[1]): 
        weight_img = W1[:, i].reshape(image_height, image_width, image_channels)
        weight_img_normalized = np.zeros_like(weight_img, dtype=float)
        for c in range(image_channels):
            channel = weight_img[:,:,c]
            min_val, max_val = channel.min(), channel.max()
            if max_val > min_val:
                weight_img_normalized[:,:,c] = (channel - min_val) / (max_val - min_val)
            else:
                weight_img_normalized[:,:,c] = 0.0
        ax.imshow(weight_img_normalized)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Neuron {i}")
plt.tight_layout()
plt.savefig("visualization_color.png", dpi=150)
plt.show()