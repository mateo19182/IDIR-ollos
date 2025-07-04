import os
import re
import logging
import matplotlib.pyplot as plt
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Plot experiment results (simplified)")
parser.add_argument(
    "--filters",
    nargs='+',
    type=str,
    default=[],
    help="Only process experiments whose directory name contains ALL of these substrings"
)
parser.add_argument(
    "-n",
    type=int,
    help="Show only the top N runs based on AUC",
    default=None
)
parser.add_argument(
    "--legend-labels",
    nargs='+',
    type=str,
    default=None,
    help="Custom legend labels for the experiments (in order of appearance)"
)
args = parser.parse_args()
filters = args.filters
legend_labels = args.legend_labels

# experiments_dir = os.path.join("out", "zzz", "RFMID")
experiments_dir = os.path.join("out", "base", "RFMID")
model_configs = {}

def ensure_all_difficulty_ranges(sorted_stats):
    expected_ranges = ['0_150', '150_300', '300_450', '450_600', '600_750']
    stats_dict = dict(sorted_stats)
    complete_stats = [(r, stats_dict.get(r, 0.0)) for r in expected_ranges]
    return complete_stats

def calculate_auc(percentages):
    x = np.linspace(0, 1, len(percentages))
    y = np.array(percentages) / 100.0
    return np.trapz(y, x)

def parse_model_name(model_name):
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

for model_name in os.listdir(experiments_dir):
    model_info = parse_model_name(model_name)
    if not model_info:
        continue
    if filters:
        if not all(filter_val in model_info.values() for filter_val in filters):
            continue
    model_path = os.path.join(experiments_dir, model_name)
    if not os.path.isdir(model_path):
        continue
    stats = {}
    for difficulty in os.listdir(model_path):
        if not re.match(r'\d+_\d+', difficulty):
            continue
        difficulty_path = os.path.join(model_path, difficulty)
        if not os.path.isdir(difficulty_path):
            continue
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
            continue
        percent = (success_count / total_instances) * 100
        stats[difficulty] = percent
    if stats:
        sorted_stats = sorted(stats.items(), key=lambda x: int(x[0].split('_')[0]))
        complete_stats = ensure_all_difficulty_ranges(sorted_stats)
        labels, percentages = zip(*complete_stats)
        model_configs[model_name] = {"labels": labels, "percentages": percentages}

# After collecting model_configs, sort by batch size
def get_batch_size(model_name):
    parsed = parse_model_name(model_name)
    if parsed:
        try:
            return int(parsed['batch_size'])
        except Exception:
            return float('inf')
    return float('inf')

# Sort model_configs by batch size
model_configs = dict(
    sorted(model_configs.items(), key=lambda item: get_batch_size(item[0]))
)

if args.n is not None:
    model_aucs = {model_name: calculate_auc(data["percentages"]) for model_name, data in model_configs.items()}
    sorted_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)
    top_n_models = sorted_models[:args.n]
    model_configs = {model: model_configs[model] for model, _ in top_n_models}

model_aucs = {
    model_name: calculate_auc(data["percentages"])
    for model_name, data in model_configs.items()
}

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_ylim(0, 100)

colors = plt.cm.tab10(np.linspace(0, 1, len(model_configs)))

for idx, ((model_name, data), color) in enumerate(zip(model_configs.items(), colors)):
    auc = model_aucs[model_name]
    # Always append AUC to the label, even if custom
    if legend_labels and idx < len(legend_labels):
        label = f"{legend_labels[idx]} (AUC: {auc:.3f})"
    else:
        label = f"{model_name} (AUC: {auc:.3f})"
    x = np.arange(len(data["labels"]))
    ax.plot(
        x,
        data["percentages"],
        color=color,
        marker='o',
        markersize=8,
        linewidth=2,
        label=label
    )

ax.set_xlabel("Rango de dificultade")
ax.set_ylabel("Taxa de éxito (%)")
ax.set_title("Taxas de éxito por dificultade da transformación con ReLU")

if model_configs:
    first_labels = list(next(iter(model_configs.values()))["labels"])
    ax.set_xticks(np.arange(len(first_labels)))
    ax.set_xticklabels(first_labels, fontsize=12)

ax.legend(loc='lower left',  title="Batch Size", fontsize=18, title_fontsize=20)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("experiment_plot_RFMID_simple.png")
plt.show()

print("Feito! Gráfica gardada como experiment_plot_RFMID_simple.png")