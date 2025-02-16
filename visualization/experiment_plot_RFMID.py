import os
import re
import logging
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.lines import Line2D

# Setup logging
logging.basicConfig(level=logging.INFO)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Plot experiment results with multiple optional filters"
)
parser.add_argument(
    "--filters",
    nargs='+',
    type=str,
    default=[],
    help="Only process experiments whose directory name contains ALL of these substrings"
)
args = parser.parse_args()
filters = args.filters

# Set the experiments directory (relative to project root)
experiments_dir = os.path.join("out", "zzz", "RFMID")
model_configs = {}


# Add this after sorting the stats but before creating model_configs entry
def ensure_all_difficulty_ranges(sorted_stats):
    """Ensure all difficulty ranges are present, fill missing ones with 0."""
    expected_ranges = ['0_150', '150_300', '300_450', '450_600', '600_750']
    stats_dict = dict(sorted_stats)
    complete_stats = [(r, stats_dict.get(r, 0.0)) for r in expected_ranges]
    return complete_stats

def parse_model_name(model_name):
    # dir created like: base_out_dir = os.path.join(current_directory, 'out', 'zzz', TARGET, f"{kwargs['network_type']}-{kwargs['lr']}-{kwargs['epochs']}-{kwargs['batch_size']}-{kwargs['sampling']}", dif)
    # example dirs: SIREN-5e-05-1500-100000-random , MLP-1e-05-1000-50000-random , MLP-0.0001-1000-1000-random ...
    """Parse model name into its components using regex."""
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

# Automatically traverse experiment directories
for model_name in os.listdir(experiments_dir):
    model_info = parse_model_name(model_name)
    if not model_info:
        continue

    # Use parsed values for filtering.
    # For each provided filter check whether it exactly matches one of the model_info values.
    # For example, if you want to filter by batch size '1000', you only match on that field.
    if filters:
        if not all(filter_val in model_info.values() for filter_val in filters):
            continue

    model_path = os.path.join(experiments_dir, model_name)
    if not os.path.isdir(model_path):
        continue
    # logging.info(f"Processing model: {model_name}")
    stats = {}
    # Iterate over each difficulty folder (formatted like "digits_digits")
    for difficulty in os.listdir(model_path):
        if not re.match(r'\d+_\d+', difficulty):
            continue
        difficulty_path = os.path.join(model_path, difficulty)
        if not os.path.isdir(difficulty_path):
            continue

        # Count successes by iterating over each instance folder (named as digits)
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
            # Require a number for threshold
            threshold_match = re.search(r"Threshold for 90% success rate:\s*([0-9]*\.?[0-9]+)", contents)
            # Require that there is a line exactly "improved"
            improved_match = re.search(r"^\s*improved\s*$", contents, re.MULTILINE)
            if threshold_match and improved_match:
                success_count += 1

        if total_instances == 0:
            logging.warning(f"No instance folders found in {difficulty_path}")
            continue

        percent = (success_count / total_instances) * 100
        stats[difficulty] = percent
        # logging.info(f"Model {model_name}, Difficulty {difficulty}: {percent:.2f}%")

    if stats:
        # Sort difficulties by the lower bound
        sorted_stats = sorted(stats.items(), key=lambda x: int(x[0].split('_')[0]))
        # Ensure all difficulty ranges are present
        complete_stats = ensure_all_difficulty_ranges(sorted_stats)
        labels, percentages = zip(*complete_stats)
        model_configs[model_name] = {"labels": labels, "percentages": percentages}
        
        logging.info("\nData being added to plot:")
        logging.info(f"Model: {model_name}")
        logging.info(f"Labels: {labels}")
        logging.info(f"Percentages: {percentages}")
        logging.info("-" * 50)
        
        # Parse and format model information
        model_info = parse_model_name(model_name)
        if model_info:
            logging.info(f"\nModel Configuration:\n"
                        f"  Network Type: {model_info['network_type']}\n"
                        f"  Learning Rate: {model_info['learning_rate']}\n"
                        f"  Epochs: {model_info['epochs']}\n"
                        f"  Batch Size: {model_info['batch_size']}\n"
                        f"  Sampling: {model_info['sampling']}\n"
                        f"\nSuccess Rates by Difficulty Range:")
            
            for difficulty, percent in sorted_stats:
                lower, upper = map(int, difficulty.split('_'))
                logging.info(f"  {lower:3d}-{upper:<3d}: {percent:6.2f}%")
            logging.info("-" * 50)

# Add this before the plotting loop

logging.info("\nFinal data to be plotted:")
for model_name, data in model_configs.items():
    logging.info(f"\nModel: {model_name}")
    for label, percentage in zip(data["labels"], data["percentages"]):
        logging.info(f"  {label}: {percentage:.2f}%")
logging.info("-" * 50)

# Use a modern style for plotting
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(16, 8))

# Define markers mapping for different batch sizes
markers = {
    "1000": "o",
    "10000": "s",
    "50000": "^",
    "100000": "D"
}

# Define line styles mapping for different learning rates
line_styles = {
    "5e-05": "-",
    "1e-05": "--",
    "0.0001": "-."
}

# Plot each experiment configuration with marker and line style differentiators
for model_name, data in model_configs.items():
    parsed = parse_model_name(model_name)
    if not parsed:
        continue

    # Use the parsed batch size to select the marker.
    batch_size = parsed['batch_size']
    marker = markers.get(batch_size, 'o')
    
    # Use the parsed learning rate to select a line style.
    learning_rate = parsed['learning_rate']
    ls = line_styles.get(learning_rate, '-')  # fallback line style

    x = np.arange(len(data["labels"]))
    ax.plot(
        x,
        data["percentages"],
        marker=marker,
        markersize=10,
        linestyle=ls,
        linewidth=2,
        label=model_name
    )

# Set axis labels and title
ax.set_xlabel("Difficulty Range")
ax.set_ylabel("Success Rate (%)")
ax.set_title("Success Rates per Difficulty of Transformation")

# Use the first model's difficulty labels for x-ticks if available
if model_configs:
    first_labels = list(next(iter(model_configs.values()))["labels"])
    ax.set_xticks(np.arange(len(first_labels)))
    ax.set_xticklabels(first_labels, fontsize=12)

# Create primary legend for model configurations
model_legend = ax.legend(loc='lower left', title="Models")

# Create custom legend for batch size markers
batch_handles = [
    Line2D([0], [0], marker=marker, color='black', linestyle='', markersize=10, label=bs)
    for bs, marker in markers.items()
]
batch_legend = ax.legend(handles=batch_handles, title="Batch Size", loc='upper right')
ax.add_artist(model_legend)

# Create custom legend for learning rate line styles and position it right under the batch size legend
lr_handles = [
    Line2D([0], [0], color='black', linestyle=ls, linewidth=2, label=lr)
    for lr, ls in line_styles.items()
]
lr_legend = ax.legend(handles=lr_handles, title="Learning Rate", loc='upper right', bbox_to_anchor=(1, 0.55))
ax.add_artist(batch_legend)

# Adjust layout and save/show plot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("experiment_plot_RFMID.png")
plt.show()

print("Done! Plot saved as experiment_plot_RFMID.png")
