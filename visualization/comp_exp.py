import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_metrics_file(path):
    """Parse metrics.txt and return dict with mean_distance, auc, threshold_90."""
    stats = {"bs_mean_distance": None, "mean_distance": None, "auc": None, "threshold_90": None}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Baseline Mean Distance:"):
                    stats["bs_mean_distance"] = float(line.split(":")[1])
                if line.startswith("Mean Distance:"):
                    stats["mean_distance"] = float(line.split(":")[1])
                elif line.startswith("Area Under the Curve (AUC):"):
                    stats["auc"] = float(line.split(":")[1])
                elif line.startswith("Threshold for 90% success rate:"):
                    val = line.split(":")[1].strip()
                    if "Not achieved" not in val:
                        stats["threshold_90"] = float(val)
                    else:
                        stats["threshold_90"] = None
                elif line.startswith("did not improve"):
                    stats["threshold_90"] = None   
    except FileNotFoundError:
        pass
    return stats

def parse_results_file(path):
    """Parse results.txt for parameter information, return as dict."""
    params = {}
    if not os.path.isfile(path):
        return params
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            params[key.strip()] = val.strip()
    return params

def gather_experiment_data(exp_dir):
    """Traverse subfolders in exp_dir, parse metrics, return list of stats (dict)."""
    data = []
    for sub in os.listdir(exp_dir):
        # print(sub)
        sub_path = os.path.join(exp_dir, sub)
        if os.path.isdir(sub_path):
            mfile = os.path.join(sub_path, "metrics.txt")
            stats = parse_metrics_file(mfile)
            stats["name"] = sub
            data.append(stats)
            print(f"Processed {stats}")
    return data

def compare_experiments(dir_list):
    """
    Compare an arbitrary list of experiment directories, collecting 'mean_distance',
    'auc', and 'threshold_90'. Plots bar charts per metric plus a success heatmap.
    """
    # Gather experiment data for each directory
    experiments = {}
    for d in dir_list:
        data = gather_experiment_data(d)
        data.sort(key=lambda x: x["name"])
        experiments[d] = data
        # Remove experiments with None mean_distance
    for d in dir_list:
        experiments[d] = [exp for exp in experiments[d] if exp["mean_distance"] is not None]
    # Find all subfolders for each experiment and compute intersection
    sets_of_subs = []
    for d in dir_list:
        sets_of_subs.append(set(x["name"] for x in experiments[d]))
    common_subfolders = sorted(set.intersection(*sets_of_subs), key=lambda x: int(x.split('_')[0]))
    print(f"Common subfolders: {common_subfolders}")
    # Build dictionaries by name for easy lookup
    exp_dicts_by_name = {d: {x["name"]: x for x in experiments[d]} for d in dir_list}

    # For each subfolder in 'common_subfolders', gather metrics from each directory
    bs_mean_dists = []
    mean_dists = []
    aucs = []
    thresholds_90 = []
    successes = []

    for sub in common_subfolders:
        bs_mean_dists.append([exp_dicts_by_name[d][sub]["bs_mean_distance"] for d in dir_list])
        mean_dists.append([exp_dicts_by_name[d][sub]["mean_distance"] for d in dir_list])
        aucs.append([exp_dicts_by_name[d][sub]["auc"] for d in dir_list])
        thr_vals = [exp_dicts_by_name[d][sub]["threshold_90"] or 0 for d in dir_list]
        thresholds_90.append(thr_vals)
        successes.append([1 if exp_dicts_by_name[d][sub]["threshold_90"] is not None else 0 for d in dir_list])

    # Convert to numpy arrays for convenience
    bs_mean_dists = np.array(bs_mean_dists)  # shape: (num_subfolders, num_dirs)
    mean_dists = np.array(mean_dists)      # shape: (num_subfolders, num_dirs)
    aucs = np.array(aucs)                  # shape: (num_subfolders, num_dirs)
    thresholds_90 = np.array(thresholds_90)# shape: (num_subfolders, num_dirs)
    successes = np.array(successes)        # shape: (num_subfolders, num_dirs)

    # Verify data is not empty
    if not common_subfolders:
        raise ValueError("No common subfolders found between experiments")
    if mean_dists.size == 0 or aucs.size == 0 or thresholds_90.size == 0 or successes.size == 0:
        raise ValueError("No data to plot")

    x = np.arange(len(common_subfolders))
    bar_width = 0.8 / len(dir_list)  # keep bars within total width 0.8
    fig, axs = plt.subplots(2, 2, figsize=(30, 18))
    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(dir_list)))

    # Helper to plot bars for a given ax, data matrix row=items, col=experiment
    def plot_bars(ax, data, title):
        for i, d in enumerate(dir_list):
            offset = (i - (len(dir_list) - 1)/2) * bar_width
            ax.bar(x + offset, data[:, i], width=bar_width, color=colors[i], label=os.path.basename(d))
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(common_subfolders, rotation=45, ha='right', fontsize=8)
        ax.legend()

    # Subplot (0, 0) : Mean Distance
    plot_bars(axs[0, 0], mean_dists, "Mean Distance")
    
    # Calculate statistics
    success_counts = np.sum(successes, axis=0)
    total_cases = successes.shape[0]
    success_rates = success_counts / total_cases * 100

    # Add stats text to figure
    stats_text = ""
    for i, d in enumerate(dir_list):
        stats_text += f"{os.path.basename(d)}: "
        stats_text += f"{success_rates[i]:.1f}% ({success_counts[i]}/{total_cases})\n"

    # Add text box to figure
    fig.text(0.75, 0.35, stats_text, 
             bbox=dict(facecolor='white', alpha=0.5, pad=1.0),
             fontsize=25,
             verticalalignment='center',
             horizontalalignment='center')
    
    # Subplot (0, 1) : AUC
    plot_bars(axs[0, 1], aucs, "AUC")

    # Subplot (1, 0) : Threshold 90%
    plot_bars(axs[1, 0], thresholds_90, "Threshold 90%")

    # Subplot (1, 1) : Success Heatmap (rows=subfolders, cols=experiments)
    cax = axs[1, 1].imshow(successes, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    
    
    
    axs[1, 1].set_title("Success (1=True, 0=False)")
    axs[1, 1].set_xticks(range(len(dir_list)))
    axs[1, 1].set_xticklabels([os.path.basename(d) for d in dir_list], rotation=45, ha='right')
    axs[1, 1].set_yticks(range(len(common_subfolders)))
    axs[1, 1].set_yticklabels(common_subfolders, rotation=45, ha='right')
    fig.colorbar(cax, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("comp_exp.png")

if __name__ == "__main__":
    """
    Usage:
        python compare.py /path/to/exp1 /path/to/exp2 /path/to/exp3 ...
    """

    # dir_1 = "out/new/good/RFMID/MLP-1e-05-2500-150000_r"
    # dir_2 = "out/new/good/RFMID/SIREN-1e-05-2000-150000_r+reg"
    # dir_3 = "out/new/good/RFMID/RFMIDu/MLP-1e-05-2000-62500"
    # dir_4 = "out/new/good/RFMID/RFMIDu/SIREN-1e-05-2000-62500"
    # not enought common subfolders RFMID
    dir_1 = "out/real/FIRE/MLP-0.0001-r-1000-65536"
    dir_2 = "out/real/FIRE/MLP-0.0001-r-1000-131072"
    dir_3 = "out/real/FIRE/MLP-0.0001-u-1000-65536"
    dir_4 = "out/real/FIRE/MLP-0.0001-u-1000-131072"
    dir_5 = "out/real/FIRE/SIREN-1e-05-u-1000-65536"
    dir_6 = "out/real/FIRE/SIREN-1e-06-r-1000-65536"
    # dir_7 = "out/new/good/FIRE/S/MLP-1e-05-2000-150000_S_r_baseline" 

    # dir_reg = "out/new/FIRE/MLP-1e-05-2000-150000_A_r"
    # dir_percentage = "out/new/RFMID/SIREN-1e-05-2000-150000_p"
    dirs = [dir_1, dir_2, dir_3, dir_4, dir_5, dir_6]
    compare_experiments(dirs)
    print("Comparison file saved as 'comp_exp.png'")