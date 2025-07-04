import os
import re
import argparse
import matplotlib.pyplot as plt

def extract_threshold(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'Threshold for 90% success rate: (\d+\.\d+|Not achieved)', line)
            if match:
                value = match.group(1)
                return float(value) if value != 'Not achieved' else None
    return None

def collect_thresholds(base_path):
    thresholds = []
    total_instances = 0
    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)
        if os.path.isdir(dir_path):
            total_instances += 1
            metrics_path = os.path.join(dir_path, 'metrics.txt')
            if os.path.exists(metrics_path):
                threshold = extract_threshold(metrics_path)
                thresholds.append(threshold)
    return thresholds, total_instances

def plot_fire_registration_scores(thresholds_list, labels, output_file):
    plt.figure(figsize=(10, 6))
    
    for thresholds, label in zip(thresholds_list, labels):
        total_instances = len(thresholds)
        thresholds = [t for t in thresholds if t is not None]
        thresholds.sort()
        x_values = [0] + thresholds  # Start from 0
        y_values = [0] + [(i + 1) / total_instances * 100 for i in range(len(thresholds))]

        # Ensure the plot goes up to 100% and reaches the 25 mark
        if len(thresholds) < total_instances:
            x_values.append(25)
            y_values.append(y_values[-1])  # Stay at the last percentage value

        plt.plot(x_values, y_values, linestyle='-', label=label)
    
    plt.xlabel('Limiar de Erro')
    plt.ylabel('Porcentaxe de Pares de Imaxes Rexistrados con Éxito')
    plt.title('Puntuación de Rexistro FIRE con ReLU')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.xlim(0, 25)  # Ensure the x-axis starts at 0 and ends at 25
    plt.xticks(range(0, 26, 5))  # Adjust x-axis ticks to show every 5 units, including 25
    plt.legend(title="Batch Size")
    plt.savefig(output_file)
    plt.close() 
    print(f"Plot saved as {output_file}")

def extract_batch_size(dir_name):
    # Find all numbers in the directory name and return the last one as int
    matches = re.findall(r'\d+', dir_name)
    return int(matches[-1]) if matches else float('inf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FIRE registration scores.")
    parser.add_argument(
        "--base_dir", type=str, required=True,
        help="Base directory containing subdirectories with metrics.txt files."
    )
    parser.add_argument(
        "--dir_regex", type=str, default=".*",
        help="Regex to filter subdirectories in base_dir."
    )
    parser.add_argument(
        "--labels", type=str, nargs="*", default=None,
        help="Custom legend labels (space-separated, must match number of selected dirs)."
    )
    parser.add_argument(
        "--output_file", type=str, default="fire_registration_scores_combined.png",
        help="Output filename for the plot."
    )
    args = parser.parse_args()

    # Filter subdirectories using regex
    dir_names = [
        d for d in os.listdir(args.base_dir)
        if os.path.isdir(os.path.join(args.base_dir, d)) and re.match(args.dir_regex, d)
    ]

    # Sort by batch size (last number in dir name)
    dir_names = sorted(dir_names, key=extract_batch_size)

    thresholds_list = []
    labels = []

    for dir_name in dir_names:
        dir_path = os.path.join(args.base_dir, dir_name)
        thresholds, total_instances = collect_thresholds(dir_path)
        if thresholds:
            thresholds_list.append(thresholds)
            labels.append(dir_name)
        else:
            print(f"No valid thresholds found in {dir_path}.")

    # Use custom labels if provided and count matches
    if args.labels:
        if len(args.labels) != len(labels):
            raise ValueError("Number of custom labels must match number of selected directories.")
        labels = args.labels

    if thresholds_list:
        plot_fire_registration_scores(thresholds_list, labels, args.output_file)
    else:
        print("No valid thresholds found for any directory.")