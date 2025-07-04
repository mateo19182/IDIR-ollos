import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def extract_mean_distance(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if 'Mean mean_distances:' in line:
                return float(line.split(':')[1].strip())
    return None

def process_results_directory_lr(directory, filter_str):
    """
    Processes folders with names like:
      "MLP-0.0001-1500-10000_0.5-0-50"
    Extracts:
      - network type: e.g., 'MLP'
      - learning rate: the second field (can be in scientific notation)
    
    Returns a list of tuples: (network_type, lr, mean_distance)
    """
    results = []
    # Regex pattern to extract network type and learning rate,
    # allowing scientific notation for the learning rate.
    pattern = re.compile(r"^(?P<network>[A-Za-z0-9]+)-(?P<lr>\d*\.?\d+(?:[eE][-+]?\d+)?)-")
    
    for root, dirs, files in os.walk(directory):
        folder_name = os.path.basename(root)
        if filter_str and filter_str not in folder_name:
            continue
        if 'results.txt' in files:
            try:
                if '_' not in folder_name:
                    continue
                match = pattern.search(folder_name)
                if not match:
                    continue
                network_type = match.group("network")
                lr = float(match.group("lr"))
                results_txt = os.path.join(root, 'results.txt')
                mean_distance = extract_mean_distance(results_txt)
                if mean_distance is not None:
                    results.append((network_type, lr, mean_distance))
                    print(f"Added: {network_type}, lr: {lr}, Mean distance: {mean_distance:.2f}")
                else:
                    print("No mean distance found in", results_txt)
            except Exception as e:
                print("Error processing", root, ":", e)
                continue
    results.sort(key=lambda x: (x[0], x[1]))
    return results

def create_visualization_lr(results, directory, filter_str):
    """
    Create a bar chart where:
      - x-axis: learning rates (sorted and formatted)
      - y-axis: mean distances (log scale)
    Each bar is annotated with its mean distance value (displayed in a big font).
    If multiple network types exist, bars are colored according to their network.
    """
    # Sort results by learning rate
    results_sorted = sorted(results, key=lambda x: x[1])
    
    # Extract learning rates, distances, and network types
    lrs = [r[1] for r in results_sorted]
    distances = [r[2] for r in results_sorted]
    networks = [r[0] for r in results_sorted]
    
    x = np.arange(len(results_sorted))
    
    # Create a color mapping for networks using a colormap
    unique_networks = list(set(networks))
    cmap = plt.get_cmap("tab10")
    network_color = {net: cmap(i % 10) for i, net in enumerate(unique_networks)}
    bar_colors = [network_color[net] for net in networks]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(x, distances, color=bar_colors, edgecolor='k', linewidth=0.5)
    
    # Annotate each bar with its mean distance value (using a larger font)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f"{distances[i]:.2f}",
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
                
    # Use log scale for the y-axis to better handle wide-ranging values
    ax.set_yscale('log')
    
    # Configure x-axis ticks: format learning rates in scientific or fixed notation
    lr_labels = [f"{lr:.2e}" if lr < 0.001 or lr > 1 else f"{lr:.3f}" for lr in lrs]
    ax.set_xticks(x)
    ax.set_xticklabels(lr_labels, fontsize=12, rotation=45)
    
    ax.set_xlabel("Learning Rate", fontsize=14)
    ax.set_ylabel("Mean Distance (log scale)", fontsize=14)
    ax.set_title("Learning Rate vs Mean Distance", fontsize=16, pad=15)
    
    # Create a legend for network types
    handles = []
    for net, color in network_color.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, edgecolor='k'))
    ax.legend(handles, network_color.keys(), title="Network Type", fontsize=12, title_fontsize=13)
    
    # Display additional information about the directory and filter on the plot
    last_dir = os.path.basename(os.path.normpath(directory))
    info_text = f"Directory: {last_dir}\nFilter: {filter_str if filter_str else 'None'}"
    ax.text(0.01, 0.98, info_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="w", ec="0.5", alpha=0.9))
    
    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(directory), "grid_search_lr.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    return output_file

def main():
    directory = input("Enter the directory path for grid search results: ").strip()
    directory = os.path.expanduser(directory)
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return
    
    filter_str = input("Enter a filter string for folder names (e.g., 'MLP' or 'SIREN') or leave empty: ").strip()
    results = process_results_directory_lr(directory, filter_str)
    if not results:
        print("No valid results found.")
        return
    
    plot_file = create_visualization_lr(results, directory, filter_str)
    print(f"Heatmap visualization saved to: {plot_file}")

if __name__ == "__main__":
    main()