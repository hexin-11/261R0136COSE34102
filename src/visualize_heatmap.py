"""
Visualization script for attention entropy heatmaps.

This script will be used to visualize layer-wise and head-wise
attention entropy patterns of teacher and student Transformer models.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_entropy_heatmap(entropy_matrix, save_path=None, title="Attention Entropy Heatmap"):
    """
    Plot an attention entropy heatmap.

    Args:
        entropy_matrix: 2D array with shape [num_layers, num_heads].
        save_path: Path to save the figure.
        title: Title of the heatmap.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(entropy_matrix, aspect="auto")
    plt.colorbar(label="Attention Entropy")
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    dummy_entropy = np.random.rand(12, 12)
    plot_entropy_heatmap(dummy_entropy, title="Dummy Attention Entropy Heatmap")
