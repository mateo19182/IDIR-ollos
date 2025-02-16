import os
import re
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Setup logging
logging.basicConfig(level=logging.INFO)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Generate heatmap(s) for success percentages by learning rate and batch size on a given difficulty"
)
parser.add_argument(
    "--difficulty",
    type=str,
    default=None,
    help="The difficulty folder to process (e.g. '0_150')"
)
parser.add_argument(
    "--filters",
    nargs='+',
    type=str,
    default=[],
    help="Only process experiments whose directory name contains ALL of these substrings"
)
args = parser.parse_args()
# If difficulty is provided, create a list with one element.
# Otherwise, process a default list of difficulties.
difficulties = [args.difficulty] if args.difficulty else ['0_150', '150_300', '300_450', '450_600']
filters = args.filters

# Set the experiments directory (relative to project root)
experiments_dir = os.path.join("out", "zzz", "RFMID")
# Dictionary to store heatmap data per difficulty.
# Structure: {difficulty: {learning_rate: {batch_size: [percent1, percent2, ...]}}}
results = {diff: {} for diff in difficulties}

def parse_model_name(model_name):
    """
    Parse model name into its components using regex.
    Expected pattern examples:
      SIREN-5e-05-1500-100000-random
      MLP-1e-05-1000-50000-random
      MLP-0.0001-1000-1000-random
    """
    pattern = r'([A-Za-z]+)-(\d+e-\d+|\d+\.\d+)-(\d+)-(\d+)-(\w+)'
    match = re.match(pattern, model_name)
    if match:
        network_type, lr, epochs, batch_size, sampling = match.groups()
        return {
            'network_type': network_type,
            'learning_rate': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'sampling': sampling
        }
    return None

# Traverse experiment directories
for model_name in os.listdir(experiments_dir):
    model_info = parse_model_name(model_name)
    if not model_info:
        continue

    # Apply text filters on model_info values if provided
    if filters:
        if not all(f in model_info.values() for f in filters):
            continue

    model_path = os.path.join(experiments_dir, model_name)
    if not os.path.isdir(model_path):
        continue

    for diff in difficulties:
        difficulty_path = os.path.join(model_path, diff)
        if not os.path.isdir(difficulty_path):
            logging.info(f"Difficulty '{diff}' not found for model: {model_name}")
            continue

        # Count successes for the given difficulty
        instance_folders = [f for f in os.listdir(difficulty_path)
                            if f.isdigit() and os.path.isdir(os.path.join(difficulty_path, f))]
        total_instances = len(instance_folders)
        success_count = 0

        for inst in instance_folders:
            metrics_path = os.path.join(difficulty_path, inst, "metrics.txt")
            if not os.path.exists(metrics_path):
                continue
            with open(metrics_path, "r") as mf:
                contents = mf.read()
            threshold_match = re.search(r"Threshold for 90% success rate:\s*([0-9]*\.?[0-9]+)", contents)
            improved_match = re.search(r"^\s*improved\s*$", contents, re.MULTILINE)
            if threshold_match and improved_match:
                success_count += 1

        if total_instances == 0:
            logging.warning(f"No instance folders found in {difficulty_path}")
            continue

        percent = (success_count / total_instances) * 100
        lr = model_info['learning_rate']
        bs = model_info['batch_size']
        # Initialize nested dictionaries as needed.
        if lr not in results[diff]:
            results[diff][lr] = {}
        if bs not in results[diff][lr]:
            results[diff][lr][bs] = []
        results[diff][lr][bs].append(percent)
        logging.info(f"Model: {model_name}, Difficulty: {diff}: {percent:.2f}% (lr: {lr}, batch size: {bs})")

# For each difficulty, produce a heatmap if there is data.
for diff, heatmap_dict in results.items():
    if not heatmap_dict:
        logging.info(f"No data to plot for difficulty {diff}.")
        continue

    # Prepare sorted lists for learning rates and batch sizes
    learning_rates = sorted(heatmap_dict.keys(), key=lambda x: float(x.replace('e-', 'e-')))
    all_bs = set()
    for lr_data in heatmap_dict.values():
        all_bs.update(lr_data.keys())
    batch_sizes = sorted(all_bs, key=lambda x: int(x))

    # Create a heatmap array (rows: learning rates, cols: batch sizes)
    heatmap_array = np.full((len(learning_rates), len(batch_sizes)), np.nan)
    for i, lr in enumerate(learning_rates):
        for j, bs in enumerate(batch_sizes):
            if bs in heatmap_dict[lr]:
                values = heatmap_dict[lr][bs]
                avg_percent = np.mean(values)
                heatmap_array[i, j] = avg_percent

    # Plot heatmap
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(heatmap_array, cmap='viridis', aspect='auto',
                    interpolation='nearest', origin='lower',
                    norm=Normalize(vmin=0, vmax=100))
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes, fontsize=12)
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_yticklabels(learning_rates, fontsize=12)
    ax.set_xlabel("Batch Size", fontsize=14)
    ax.set_ylabel("Learning Rate", fontsize=14)
    ax.set_title(f"Success Percentage Heatmap for Difficulty {diff}", fontsize=16)
    fig.colorbar(cax, ax=ax, label="Success Percentage (%)")

    # Add annotation text (exact percentage) in each cell
    for i in range(len(learning_rates)):
        for j in range(len(batch_sizes)):
            value = heatmap_array[i, j]
            if not np.isnan(value):
                # Choose text color based on cell value
                color = "white" if value < 50 else "black"
                ax.text(j, i, f"{value:.1f}%", ha='center', va='center', color=color, fontsize=12)

    plt.tight_layout()
    plot_filename = f"experiment_heatmap_RFMID_{diff}.png"
    plt.savefig(plot_filename)
    plt.show()
    logging.info(f"Done! Plot saved as {plot_filename}")