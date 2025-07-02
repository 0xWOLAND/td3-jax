import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def plot_results(file_path, title=None):
    """Plot training results"""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return

    # Load evaluations
    evaluations = np.load(file_path)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot evaluations
    x = np.arange(len(evaluations)) * 5000  # Assuming eval_freq=5000
    plt.plot(x, evaluations, linewidth=2)

    plt.xlabel("Timesteps", fontsize=14)
    plt.ylabel("Average Return", fontsize=14)

    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title("TD3 Training Progress", fontsize=16)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    save_path = file_path.replace(".npy", ".png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .npy results file")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    args = parser.parse_args()

    plot_results(args.file, args.title)
