import os
from collections import defaultdict
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np

def extract_mean_distance(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if 'Mean mean_distances:' in line:
                return float(line.split(':')[1].strip())
    return None

def process_results_directory(directory, filter_str):
    results = []
    for root, dirs, files in os.walk(directory):
        folder_name = os.path.basename(root)
        if folder_name.endswith('_1'):
            continue
        if filter_str and filter_str not in folder_name:
            continue
        if 'results.txt' in files:
            try:
                # Expected folder name format: "SIREN-0.0001-1500-10000_0.4-0-50"
                if '_' not in folder_name:
                    continue
                prefix, regs_str = folder_name.split('_', 1)
                network_type = prefix.split('-')[0]
                regs = regs_str.split('-')
                if len(regs) != 3:
                    continue
                hyper_reg, jacobian_reg, bending_reg = regs
                results_txt = os.path.join(root, 'results.txt')
                mean_distance = extract_mean_distance(results_txt)
                if mean_distance is not None:
                    results.append((network_type, float(hyper_reg), float(jacobian_reg), float(bending_reg), mean_distance))
                    print(f"Added: {network_type}, regs: {hyper_reg}, {jacobian_reg}, {bending_reg}, Mean distance: {mean_distance:.2f}")
                else:
                    print("No mean distance found in", results_txt)
            except Exception as e:
                print("Error processing", root, ":", e)
                continue
    results.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return results

def create_latex_table(results):
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{|l|c|c|c|c|}")
    latex.append("\\hline")
    latex.append("Network Type & Hyper Reg & Jacobian Reg & Bending Reg & Mean Distance \\\\ \\hline")
    for network, hyper, jac, bending, mean_distance in results:
        latex.append(f"{network} & {hyper} & {jac} & {bending} & {mean_distance:.2f} \\\\ \\hline")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Grid Search on Regularization: Mean Distances}")
    latex.append("\\label{tab:grid_search_regularization}")
    latex.append("\\end{table}")
    return "\n".join(latex)

def create_visualization(results, directory, filter_str):
    """
    Creates a single heatmap with:
      - x-axis: Hyper Regularization (sorted low-to-high)
      - y-axis: (Jacobian, Bending) pairs sorted by (jacobian+bending, jacobian, bending)
    The cell annotation shows the averaged mean distance.
    Also, adds a text box with the last directory name and regex word.
    """
    # Unique hyper values (x-axis)
    hyper_set = sorted(set(r[1] for r in results))
    
    # Unique (jacobian, bending) pairs sorted by total regularization
    jac_bend_set = sorted(set((r[2], r[3]) for r in results), key=lambda x: (x[0] + x[1], x[0], x[1]))
    y_labels = [f"J: {jb[0]}, B: {jb[1]}" for jb in jac_bend_set]
    
    # Build a dictionary (jac, bend, hyper) -> list of mean_distance, then average duplicates
    data_dict = defaultdict(list)
    for _, hyper, jac, bend, mean_distance in results:
        data_dict[(jac, bend, hyper)].append(mean_distance)
    
    # Build pivot matrix: rows are (jac, bend) and cols are hyper values
    matrix = np.full((len(jac_bend_set), len(hyper_set)), np.nan)
    for i, (jac, bend) in enumerate(jac_bend_set):
        for j, hyper in enumerate(hyper_set):
            values = data_dict.get((jac, bend, hyper), [])
            if values:
                matrix[i, j] = np.mean(values)
    
    fig, ax = plt.subplots(figsize=(0.8 * len(hyper_set) + 3, 0.6 * len(jac_bend_set) + 3))
    cax = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_title("Grid Search Regularization Heatmap")
    ax.set_xlabel("Hyper Regularization")
    ax.set_ylabel("Jacobian & Bending Regularization")
    
    ax.set_xticks(np.arange(len(hyper_set)))
    ax.set_xticklabels(hyper_set, rotation=45)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    
    # Annotate each cell with the mean distance value
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='w', fontsize=8)
    
    # Add a colorbar to the side
    fig.colorbar(cax, ax=ax, label="Mean Distance")
    
    # Get last directory and regex word (filter string)
    last_dir = os.path.basename(os.path.normpath(directory))
    regex_word = filter_str if filter_str else "None"
    
    # Add an annotation box with directory and regex word info
    info_text = f"Last Directory: {last_dir}\nRegex: {regex_word}"
    ax.text(0.01, 0.99, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    
    # Incorporate last directory and regex word into the output file name
    safe_regex = regex_word.replace(" ", "_")
    output_file = os.path.join(os.path.dirname(directory), f"grid_search_single_heatmap_{last_dir}_{safe_regex}.png")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    return output_file

def plot_single_reg_effects(results, directory):
    """
    Plots bar graphs for each regularization type, showing mean distance as that reg increases,
    with the other two regularizations set to zero. Handles float and int values cleanly.
    """
    reg_types = [
        ("Hyper", 1, 2, 3),
        ("Jacobian", 2, 1, 3),
        ("Bending", 3, 1, 2)
    ]
    for reg_label, reg_idx, zero_idx1, zero_idx2 in reg_types:
        filtered = [r for r in results if r[zero_idx1] == 0 and r[zero_idx2] == 0]
        if not filtered:
            continue
        # Sort by the regularization value
        filtered.sort(key=lambda x: x[reg_idx])
        x_vals = [r[reg_idx] for r in filtered]
        y_vals = [r[4] for r in filtered]
        # Format: show as int if possible, else 2 decimals
        def fmt(x):
            return f"{int(x)}" if float(x).is_integer() else f"{x:.2f}"
        tick_labels = [fmt(x) for x in x_vals]
        plt.figure(figsize=(max(6, len(x_vals)), 5))
        plt.bar(range(len(x_vals)), y_vals, color='skyblue')
        plt.xticks(range(len(x_vals)), tick_labels, rotation=45)
        plt.xlabel(f"{reg_label} Regularization")
        plt.ylabel("Mean Distance")
        plt.title(f"Effect of {reg_label} Regularization (others=0)")
        plt.tight_layout()
        output_file = os.path.join(
            os.path.dirname(directory),
            f"barplot_{reg_label.lower()}_reg_effect.png"
        )
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved bar plot for {reg_label} Regularization to: {output_file}")

def plot_hyper_reg_comparison(results, directory):
    """
    Plots a grouped bar chart comparing Hyper Regularization effects for MLP and SIREN.
    """
    # Filter for MLP and SIREN, with jacobian and bending = 0
    mlp = [r for r in results if r[0] == "MLP" and r[2] == 0 and r[3] == 0]
    siren = [r for r in results if r[0] == "SIREN" and r[2] == 0 and r[3] == 0]

    # Collect all unique hyper values
    all_hyper = sorted(set([r[1] for r in mlp] + [r[1] for r in siren]))

    # Helper to format x labels
    def fmt(x):
        return f"{int(x)}" if float(x).is_integer() else f"{x:.2f}"
    tick_labels = [fmt(x) for x in all_hyper]

    # Map hyper values to mean distances for each network
    mlp_map = {r[1]: r[4] for r in mlp}
    siren_map = {r[1]: r[4] for r in siren}
    mlp_y = [mlp_map.get(h, np.nan) for h in all_hyper]
    siren_y = [siren_map.get(h, np.nan) for h in all_hyper]

    x = np.arange(len(all_hyper))
    width = 0.35

    plt.figure(figsize=(max(7, len(all_hyper) * 1.2), 6))
    plt.bar(x - width/2, mlp_y, width, label="Relu", color='skyblue')
    plt.bar(x + width/2, siren_y, width, label="SIREN", color='orange')
    plt.xticks(x, tick_labels, rotation=45)
    plt.xlabel("Hyper Regularization")
    plt.ylabel("Distancia Media")
    plt.title("Hyper Regularization Effect: Relu vs SIREN (others=0)")
    plt.legend()
    plt.tight_layout()
    output_file = os.path.join(
        os.path.dirname(directory),
        "barplot_hyper_reg_comparison_MLP_vs_SIREN.png"
    )
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved grouped bar plot for Hyper Regularization (MLP vs SIREN) to: {output_file}")

def main():
    directory = input("Enter the directory path for grid search results: ")
    directory = os.path.expanduser(directory)
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return
    
    filter_str = input("Enter a filter string for folder names (e.g., 'SIREN') or leave empty: ").strip()
    results = process_results_directory(directory, filter_str)
    if not results:
        print("No valid results found.")
        return
    plot_hyper_reg_comparison(results, directory)
    # Create PrettyTable output
    table = PrettyTable()
    table.field_names = ["Network Type", "Hyper Reg", "Jacobian Reg", "Bending Reg", "Mean Distance"]
    for result in results:
        table.add_row([result[0], f"{result[1]:g}", f"{result[2]:g}", f"{result[3]:g}", f"{result[4]:.2f}"])
    
    latex_table = create_latex_table(results)
    latex_file = os.path.join(os.path.dirname(directory), "grid_search_mean_distances_table.tex")
    with open(latex_file, "w") as f:
        f.write(latex_table)
    
    plot_file = create_visualization(results, directory, filter_str)
    plot_single_reg_effects(results, directory)
    plot_hyper_reg_comparison(results, directory)
    
    print("\nPretty Table Output:")
    print(table)
    print(f"\nLaTeX table saved to: {latex_file}")
    print(f"Heatmap visualization saved to: {plot_file}")

if __name__ == "__main__":
    main()