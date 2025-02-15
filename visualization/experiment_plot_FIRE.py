import os
import re
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set the experiments directory for FIRE
experiments_dir = os.path.join("out", "zzz", "FIRE")
experiment_results = {}

def parse_experiment_name(exp_name):
    """
    Parse the experiment folder name.
    Expected formats:
      With sampling: "MLP-0.0001-u-1000-131072"
      Without sampling: "SIREN-1e-05-500-20000"
    Note: scientific notation like "1e-05" may be split by '-' so we join if necessary.
    Returns a dict with the following keys:
      network_type, lr, sampling (may be None), epochs, batch_size.
    """
    parts = exp_name.split('-')
    cfg = {}
    cfg["network_type"] = parts[0]
    # Handle potential splitting of lr in scientific notation
    if parts[1].endswith("e"):
        lr_str = parts[1] + "-" + parts[2]
        index = 3
    else:
        lr_str = parts[1]
        index = 2
    try:
        cfg["lr"] = float(lr_str)
    except ValueError:
        cfg["lr"] = 0.0

    remaining = parts[index:]
    if len(remaining) == 3:
        # With sampling: sampling, epochs, batch_size
        cfg["sampling"] = remaining[0]
        cfg["epochs"] = int(remaining[1])
        cfg["batch_size"] = int(remaining[2])
    elif len(remaining) == 2:
        # Without sampling: epochs, batch_size
        cfg["sampling"] = None
        cfg["epochs"] = int(remaining[0])
        cfg["batch_size"] = int(remaining[1])
    else:
        cfg["sampling"] = None
        cfg["epochs"] = 0
        cfg["batch_size"] = 0

    return cfg

def sort_key(exp_name):
    cfg = parse_experiment_name(exp_name)
    # Use tuple ordering: network_type, lr, sampling (None sorts before any string), epochs, batch_size
    return (cfg["network_type"], cfg["lr"], cfg["sampling"] if cfg["sampling"] is not None else '', cfg["epochs"], cfg["batch_size"])

# Iterate over each experiment folder
for exp_name in os.listdir(experiments_dir):
    exp_path = os.path.join(experiments_dir, exp_name)
    if not os.path.isdir(exp_path):
        continue
    logging.info(f"Processing experiment: {exp_name}")
    # Assume instance folders follow the pattern \d+_S (e.g., "97_S")
    instance_folders = [folder for folder in os.listdir(exp_path)
                        if os.path.isdir(os.path.join(exp_path, folder)) and re.match(r'\d+_S$', folder)]
    total_instances = len(instance_folders)
    success_count = 0

    for folder in instance_folders:
        metrics_file = os.path.join(exp_path, folder, "metrics.txt")
        if not os.path.exists(metrics_file):
            continue
        with open(metrics_file, "r") as mf:
            contents = mf.read()
        threshold_match = re.search(r"Threshold for 90% success rate:\s*([0-9]*\.?[0-9]+)", contents)
        improved_match = re.search(r"^\s*improved\s*$", contents, re.MULTILINE)
        if threshold_match and improved_match:
            success_count += 1

    if total_instances == 0:
        logging.warning(f"No valid instance folders found in {exp_path}")
        continue

    percent = (success_count / total_instances) * 100
    experiment_results[exp_name] = percent
    logging.info(f"Experiment {exp_name}: {percent:.2f}%")

if not experiment_results:
    logging.error("No experiment results to plot.")
    exit(1)

# Sort experiments by configuration
sorted_experiments = sorted(experiment_results.keys(), key=sort_key)
sorted_percentages = [experiment_results[exp] for exp in sorted_experiments]

# Color mapping by network type and sampling
colors_map = {
    ("MLP", "r"): "lightblue",
    ("MLP", "u"): "darkblue",
    ("SIREN", "r"): "sandybrown",
    ("SIREN", "u"): "darkorange"
}

bar_colors = []
for exp in sorted_experiments:
    cfg = parse_experiment_name(exp)
    # Default to 'r' if no sampling specified
    sampling = cfg["sampling"] if cfg["sampling"] else "r"
    color = colors_map.get((cfg["network_type"], sampling), "gray")
    bar_colors.append(color)

# Create bar plot for experiments
fig, ax = plt.subplots(figsize=(16, 8))
x_pos = range(len(sorted_experiments))
bars = ax.bar(x_pos, sorted_percentages, color=bar_colors, width=0.8)

# Set labels and title
ax.set_xlabel("Experiment Configuration")
ax.set_ylabel("Success Rate (%)")
ax.set_title("FIRE Experiments Success Rates by Network Type and Sampling Strategy")

# Annotate bars with configuration parameters inside the columns
for i, (exp, height) in enumerate(zip(sorted_experiments, sorted_percentages)):
    cfg = parse_experiment_name(exp)
    # Default to 'r' if no sampling specified
    sampling = cfg["sampling"] if cfg["sampling"] else "r"
    label = f"lr: {cfg['lr']}\nep: {cfg['epochs']}\nbs: {cfg['batch_size']}\nsamp: {sampling}"
    
    # Position text inside the bar if height is sufficient, otherwise above
    if height > 15:  # threshold for putting text inside bar
        y_pos = height/2  # middle of bar
        color = 'white'  # white text for contrast
    else:
        y_pos = height + 1
        color = 'black'
    
    ax.text(x_pos[i], y_pos, label, ha="center", va="center", 
            fontsize=9, color=color, fontweight='bold')

# Create custom legend for network types and sampling strategies
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_map[("MLP", "r")], label='MLP (random)'),
    Patch(facecolor=colors_map[("MLP", "u")], label='MLP (uniform)'),
    Patch(facecolor=colors_map[("SIREN", "r")], label='SIREN (random)'),
    Patch(facecolor=colors_map[("SIREN", "u")], label='SIREN (uniform)')
]
ax.legend(handles=legend_elements, title="Network Type and Sampling", loc="upper right")

plt.tight_layout()
plt.savefig("experiment_plot_FIRE.png")
plt.show()

print("Done! Plot saved as experiment_plot_FIRE.png")