import os
import numpy as np
import matplotlib.pyplot as plt

def parse_metrics_file(path):
    """Parse metrics.txt and return dict with mean_distance, auc, threshold_90."""
    stats = {"mean_distance": None, "auc": None, "threshold_90": None}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
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
    """Traverse subfolders in exp_dir, parse metrics, return list of stats."""
    data = []
    for sub in os.listdir(exp_dir):
        sub_path = os.path.join(exp_dir, sub)
        if os.path.isdir(sub_path):
            mfile = os.path.join(sub_path, "metrics.txt")
            stats = parse_metrics_file(mfile)
            stats["name"] = sub
            data.append(stats)
    return data

def compare_experiments(dir_random, dir_weighted, dir_percentage):
    """Compare three experiments and plot them."""
    # Grab experiment data
    data_r = gather_experiment_data(dir_random)
    data_w = gather_experiment_data(dir_weighted)
    data_p = gather_experiment_data(dir_percentage)
    
    data_r.sort(key=lambda x: x["name"])
    data_w.sort(key=lambda x: x["name"])
    data_p.sort(key=lambda x: x["name"])
    
    names_r = [d["name"] for d in data_r]
    names_w = [d["name"] for d in data_w]
    names_p = [d["name"] for d in data_p]

    # Only subfolders present in all three experiments
    common = sorted(set(names_r).intersection(set(names_w)).intersection(set(names_p)))
    r_by_name = {d["name"]: d for d in data_r}
    w_by_name = {d["name"]: d for d in data_w}
    p_by_name = {d["name"]: d for d in data_p}

    # Extract data
    mean_dist_r = [r_by_name[n]["mean_distance"] for n in common]
    mean_dist_w = [w_by_name[n]["mean_distance"] for n in common]
    mean_dist_p = [p_by_name[n]["mean_distance"] for n in common]

    auc_r = [r_by_name[n]["auc"] for n in common]
    auc_w = [w_by_name[n]["auc"] for n in common]
    auc_p = [p_by_name[n]["auc"] for n in common]

    threshold_90_r = [r_by_name[n]["threshold_90"] or 0 for n in common]
    threshold_90_w = [w_by_name[n]["threshold_90"] or 0 for n in common]
    threshold_90_p = [p_by_name[n]["threshold_90"] or 0 for n in common]

    # Build success matrix for heatmap (1 if threshold_90 is not None)
    success_r = [1 if r_by_name[n]["threshold_90"] is not None else 0 for n in common]
    success_w = [1 if w_by_name[n]["threshold_90"] is not None else 0 for n in common]
    success_p = [1 if p_by_name[n]["threshold_90"] is not None else 0 for n in common]
    success_matrix = np.array([[r, w, p] for r, w, p in zip(success_r, success_w, success_p)])
    
    # Prepare figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    x = np.arange(len(common))
    bar_width = 0.25

    # (0, 0) Mean Distance
    axs[0, 0].bar(x - bar_width, mean_dist_r, width=bar_width, label="Random")
    axs[0, 0].bar(x, mean_dist_w, width=bar_width, label="Weighted")
    axs[0, 0].bar(x + bar_width, mean_dist_p, width=bar_width, label="Percentage")
    axs[0, 0].set_title("Mean Distance")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(common, rotation=45, ha='right')
    axs[0, 0].legend()

    # (0, 1) AUC
    axs[0, 1].bar(x - bar_width, auc_r, width=bar_width, label="Random")
    axs[0, 1].bar(x, auc_w, width=bar_width, label="Weighted")
    axs[0, 1].bar(x + bar_width, auc_p, width=bar_width, label="Percentage")
    axs[0, 1].set_title("AUC")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(common, rotation=45, ha='right')
    axs[0, 1].legend()

    # (1, 0) Threshold 90%
    axs[1, 0].bar(x - bar_width, threshold_90_r, width=bar_width, label="Random")
    axs[1, 0].bar(x, threshold_90_w, width=bar_width, label="Weighted")
    axs[1, 0].bar(x + bar_width, threshold_90_p, width=bar_width, label="Percentage")
    axs[1, 0].set_title("Threshold 90%")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(common, rotation=45, ha='right')
    axs[1, 0].legend()

    # (1, 1) Success Heatmap
    cax = axs[1, 1].imshow(success_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    axs[1, 1].set_title("Success (1=True, 0=False)")
    axs[1, 1].set_xticks([0, 1, 2])
    axs[1, 1].set_xticklabels(["Random", "Weighted", "Percentage"])
    axs[1, 1].set_yticks(range(len(common)))
    axs[1, 1].set_yticklabels(common, rotation=45, ha='right')
    fig.colorbar(cax, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("comp_exp.png")

if __name__ == "__main__":
    dir_random = "out/new/RFMID/SIREN-1e-05-2000-150000_r"
    dir_weighted = "out/new/RFMID/SIREN-1e-05-2000-150000_w"
    dir_percentage = "out/new/RFMID/SIREN-1e-05-2000-150000_p"
    compare_experiments(dir_random, dir_weighted, dir_percentage)
