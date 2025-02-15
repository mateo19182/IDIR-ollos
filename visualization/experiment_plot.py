import os
import re
import logging
import matplotlib.pyplot as plt
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set the experiments directory (relative to project root)
experiments_dir = os.path.join("out", "zzz", "RFMID")
model_configs = {}

# Automatically traverse experiment directories
for model_name in os.listdir(experiments_dir):
    model_path = os.path.join(experiments_dir, model_name)
    if not os.path.isdir(model_path):
        continue
    logging.info(f"Processing model: {model_name}")
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

        # # Write the correct number to results.txt inside the difficulty folder
        # results_txt = os.path.join(difficulty_path, "results.txt")
        # with open(results_txt, "w") as rf:
        #     rf.write(f"Number of successful registrations: {success_count}/{total_instances}\n")
        percent = (success_count / total_instances) * 100
        stats[difficulty] = percent
        logging.info(f"Model {model_name}, Difficulty {difficulty}: {percent:.2f}%")

    if stats:
        # Sort difficulties by the lower bound
        sorted_stats = sorted(stats.items(), key=lambda x: int(x[0].split('_')[0]))
        labels, percentages = zip(*sorted_stats)
        model_configs[model_name] = {"labels": labels, "percentages": percentages}

# Use a modern style for plotting
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(16, 8))

# Plot each experiment configuration
for model_name, data in model_configs.items():
    x = np.arange(len(data["labels"]))
    ax.plot(x, data["percentages"], marker='o', linestyle='-', linewidth=2, label=model_name)

# Set axis labels and title
ax.set_xlabel("Difficulty Range")
ax.set_ylabel("Success Rate (%)")
ax.set_title("Success Rates per Difficulty of Transformation")

# Use the first model's difficulty labels for x-ticks if available
if model_configs:
    first_labels = list(next(iter(model_configs.values()))["labels"])
    ax.set_xticks(np.arange(len(first_labels)))
    ax.set_xticklabels(first_labels, fontsize=12)
# Change the legend position to bottom left
ax.legend(loc='lower left')

# Adjust layout and save/show plot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("experiment_plot.png")
plt.show()

print("Done! Plot saved as experiment_plot.png")
