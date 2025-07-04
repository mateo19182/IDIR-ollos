import os
import re
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set the experiments directory for FIRE
experiments_dir = os.path.join("out", "phases", "FIRE")
experiment_results = {}

def parse_experiment_name(exp_name):
    """
    Parse the experiment folder name.
    Expected format: "MLP-0.0001-1500-25000-f2"
    """
    # Skip unfinished experiments
    if exp_name.endswith("_unfinished"):
        return None

    # Handle baseline case
    if exp_name == "S_baseline":
        return {
            "network_type": "baseline",
            "lr": 0.0,
            "epochs": 0,
            "batch_size": 0,
            "phase": "f1"
        }

    # Remove any suffixes starting with '_'
    main_parts = exp_name.split('_')[0].split('-')
    cfg = {}
    cfg["network_type"] = main_parts[0]
    try:
        cfg["lr"] = float(main_parts[1])
    except ValueError:
        cfg["lr"] = 0.0
    try:
        cfg["epochs"] = int(main_parts[2])
    except (ValueError, IndexError):
        cfg["epochs"] = 0
    try:
        cfg["batch_size"] = int(main_parts[3])
    except (ValueError, IndexError):
        cfg["batch_size"] = 0
    # Phase (e.g., f1, f2, f3)
    cfg["phase"] = main_parts[4] if len(main_parts) > 4 else "f1"
    return cfg



def generate_latex_table(sorted_experiments, experiment_results):
    """
    Generate a LaTeX table summarizing the experiments.
    """
    header = (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Network & Learning Rate & Epochs & Batch Size & Success Rate (\\%) \\\\\n"
        "\\midrule\n"
    )
    rows = []
    for exp in sorted_experiments:
        cfg = parse_experiment_name(exp)
        if cfg is None:
            continue
        net = cfg["network_type"]
        lr = cfg["lr"]
        ep = cfg["epochs"]
        bs = cfg["batch_size"]
        phase = cfg["phase"]
        percent = experiment_results[exp]
        rows.append(f"{net} ({phase}) & {lr} & {ep} & {bs} & {percent:.2f} \\\\")
    footer = "\\bottomrule\n\\end{tabular}"
    return header + "\n".join(rows) + "\n" + footer


def sort_key(exp_name):
    cfg = parse_experiment_name(exp_name)
    # Use tuple ordering: network_type, lr, epochs, batch_size, phase
    phase_num = int(cfg["phase"][1:]) if cfg["phase"].startswith("f") and cfg["phase"][1:].isdigit() else 0
    return (cfg["network_type"], cfg["lr"], cfg["epochs"], cfg["batch_size"], phase_num)

# Iterate over each experiment folder
for exp_name in os.listdir(experiments_dir):
    if exp_name.endswith("_unfinished"):
        logging.info(f"Skipping unfinished experiment: {exp_name}")
        continue
        
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
    ("MLP", "f1"): "lightblue",
    ("MLP", "f2"): "dodgerblue",
    ("MLP", "f3"): "navy",
    ("SIREN", "f1"): "sandybrown",
    ("SIREN", "f2"): "darkorange",
    ("SIREN", "f3"): "chocolate"
}

bar_colors = []
for exp in sorted_experiments:
    cfg = parse_experiment_name(exp)
    phase = cfg["phase"] if "phase" in cfg else "f1"
    color = colors_map.get((cfg["network_type"], phase), "gray")
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
    phase = cfg["phase"] if "phase" in cfg else "f1"
    label = f"lr: {cfg['lr']}\nep: {cfg['epochs']}\nbs: {cfg['batch_size']}\nphase: {phase}"
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
    Patch(facecolor=colors_map[("MLP", "f1")], label='MLP (f1)'),
    Patch(facecolor=colors_map[("MLP", "f2")], label='MLP (f2)'),
    Patch(facecolor=colors_map[("MLP", "f3")], label='MLP (f3)'),
    Patch(facecolor=colors_map[("SIREN", "f1")], label='SIREN (f1)'),
    Patch(facecolor=colors_map[("SIREN", "f2")], label='SIREN (f2)'),
    Patch(facecolor=colors_map[("SIREN", "f3")], label='SIREN (f3)')
]
ax.legend(handles=legend_elements, title="Network Type and Phase", loc="upper right")

plt.tight_layout()
plt.savefig("experiment_plot_FIRE.png")
plt.show()

latex_table = generate_latex_table(sorted_experiments, experiment_results)
print("\nLaTeX Table:\n")
print(latex_table)

print("Done! Plot saved as experiment_plot_FIRE.png")