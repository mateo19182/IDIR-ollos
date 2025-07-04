import os
from pathlib import Path
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np

def extract_mean_distance(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if 'Mean mean_distances:' in line:
                return float(line.split(':')[1].strip())
    return None

def create_latex_table(results):
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{|l|c|c|}")
    latex.append("\\hline")
    latex.append("Network Type & Loss Function & Mean Distance \\\\ \\hline")
    
    for network_type, loss_function, mean_distance in results:
        latex.append(f"{network_type} & {loss_function} & {mean_distance:.2f} \\\\ \\hline")
    
    latex.append("\\end{tabular}")
    latex.append("\\caption{Mean Distances by Network Type and Loss Function}")
    latex.append("\\label{tab:mean_distances}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)

def create_visualization(results, directory):
    # Prepare data for plotting
    network_types = sorted(list(set(x[0] for x in results)))
    loss_functions = sorted(list(set(x[1] for x in results)))
    
    # Set up the plot
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    bar_width = 0.35
    opacity = 0.8
    
    # Create bars for each network type
    for i, network in enumerate(network_types):
        network_data = []
        for lf in loss_functions:
            # Find a matching entry; if none, use zero.
            matches = [x[2] for x in results if x[0] == network and x[1] == lf]
            value = matches[0] if matches else 0
            network_data.append(value)
        positions = np.arange(len(loss_functions)) + i * bar_width
        plt.bar(positions, network_data, bar_width,
                alpha=opacity,
                label=network)
    
    # Customize the plot
    plt.xlabel('Loss Function')
    plt.ylabel('Mean Distance')
    plt.title('Mean Distances by Network Type and Loss Function')
    plt.xticks(np.arange(len(loss_functions)) + bar_width/2, loss_functions)
    plt.legend()
    
    # Save the plot
    output_file = os.path.join(os.path.dirname(directory), "mean_distances_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def process_results_directory(directory):
    # Store results in a list
    results = []
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Debug print to show which folder is being processed
        print("Processing folder:", root)
        if 'results.txt' in files:
            try:
                # Parse folder name to get information
                dir_name = os.path.basename(root)
                # Get network type (first part before hyphen)
                network_type = dir_name.split('-')[0]
                # Get loss function: text after the last underscore
                loss_function = dir_name.split('_')[-1]
                
                # Get mean distance from results.txt
                results_path = os.path.join(root, 'results.txt')
                mean_distance = extract_mean_distance(results_path)
                
                if mean_distance is not None:
                    results.append((network_type, loss_function, mean_distance))
                    print(f"Added: {network_type}, {loss_function}, {mean_distance:.2f}")
                else:
                    print("No mean distance found in", results_path)
            except (IndexError, ValueError) as e:
                print("Error processing", root, ":", e)
                continue
    
    # Sort results by network type and loss function
    results.sort(key=lambda x: (x[0], x[1]))
    
    # Create and populate PrettyTable
    table = PrettyTable()
    table.field_names = ["Network Type", "Loss Function", "Mean Distance"]
    for result in results:
        table.add_row([result[0], result[1], f"{result[2]:.2f}"])
    
    # Generate LaTeX table
    latex_table = create_latex_table(results)
    latex_file = os.path.join(os.path.dirname(directory), "mean_distances_table.tex")
    with open(latex_file, "w") as f:
        f.write(latex_table)
    
    # Create visualization
    plot_file = create_visualization(results, directory)
    
    return table, latex_file, plot_file

def main():
    directory = input("Enter the directory path to process: ")
    directory = os.path.expanduser(directory)
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return
    
    table, latex_file, plot_file = process_results_directory(directory)
    print("\nPretty Table Output:")
    print(table)
    print(f"\nLaTeX table has been saved to: {latex_file}")
    print(f"Visualization has been saved to: {plot_file}")

if __name__ == "__main__":
    main()