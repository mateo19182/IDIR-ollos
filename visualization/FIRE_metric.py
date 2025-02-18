import os
import re
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
    
    plt.xlabel('Error Threshold')
    plt.ylabel('Percentage of Successfully Registered Image Pairs')
    plt.title('FIRE Registration Score Category S')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.xlim(0, 25)  # Ensure the x-axis starts at 0 and ends at 25
    plt.xticks(range(0, 26, 5))  # Adjust x-axis ticks to show every 5 units, including 25
    plt.legend()
    plt.savefig(output_file)
    plt.close() 
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    # Option 1: Specify multiple directories
    base_paths = [
        "out/new/good/FIRE/S/MLP-1e-05-2000-150000_S_r_baseline",
        "out/new/good/FIRE/S/SIREN-1e-05-1500-100000_S_++reg",
        "out/zzz/FIRE_S/S_baseline", 
        "out/zzz/FIRE_S/MLP-0.0001-u-1000-131072", 
        "out/zzz/FIRE_S/MLP-0.0001-r-1000-131072"
    ]
    # Option 2: Specify a single directory containing all subdirectories
    single_base_path = "out/zzz/FIRE_S/"

    thresholds_list = []
    labels = []
    
    # Uncomment the following block to use multiple directories
    # for i, base_path in enumerate(base_paths):
    #     thresholds, total_instances = collect_thresholds(base_path)
    #     if thresholds:
    #         thresholds_list.append(thresholds)
    #         labels.append(f'Plot {i+1}')
    #     else:
    #         print(f"No valid thresholds found in {base_path}.")
    
    # Uncomment the following block to use a single directory containing all subdirectories
    for dir_name in os.listdir(single_base_path):
        dir_path = os.path.join(single_base_path, dir_name)
        if os.path.isdir(dir_path):
            thresholds, total_instances = collect_thresholds(dir_path)
            if thresholds:
                thresholds_list.append(thresholds)
                labels.append(f'{dir_name}')
            else:
                print(f"No valid thresholds found in {dir_path}.")
    
    if thresholds_list:
        output_file = 'fire_registration_scores_combined.png'
        plot_fire_registration_scores(thresholds_list, labels, output_file)