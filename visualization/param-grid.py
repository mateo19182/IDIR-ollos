import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def extract_results(dir):
    """
    Traverse the FIRE directory and extract Mean AUC, Learning Rate, and Batch Size from each results.txt.
    
    Parameters:
    - fire_dir (str): Path to the out/grid/FIRE/ directory.
    
    Returns:
    - DataFrame: Pandas DataFrame containing the extracted data.
    """
    results = []
    
    # Iterate through each subdirectory in FIRE
    for subdir in os.listdir(dir):
        subdir_path = os.path.join(dir, subdir)
        if os.path.isdir(subdir_path):
            results_file = os.path.join(subdir_path, 'results.txt')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    mean_auc = None
                    lr = None
                    batch_size = None
                    for line in lines:
                        if "Mean auc" in line:
                            try:
                                mean_auc = float(line.strip().split(":")[1])
                            except ValueError:
                                print(f"Error parsing Mean AUC in {results_file}")
                        elif line.startswith("lr:"):
                            try:
                                lr = float(line.strip().split(":")[1])
                            except ValueError:
                                print(f"Error parsing learning rate in {results_file}")
                        elif line.startswith("batch_size:"):
                            try:
                                batch_size = int(line.strip().split(":")[1])
                            except ValueError:
                                print(f"Error parsing batch size in {results_file}")
                    
                    if mean_auc is not None and lr is not None and batch_size is not None:
                        results.append({
                            'Learning_Rate': lr,
                            'Batch_Size': batch_size,
                            'Mean_AUC': mean_auc
                        })
                    else:
                        print(f"Incomplete data in {results_file}")
    
    df = pd.DataFrame(results)
    return df

def visualize_results(df, output_path=None):
    """
    Create a heatmap visualization of Mean AUC based on Learning Rate and Batch Size.
    
    Parameters:
    - df (DataFrame): Pandas DataFrame containing Learning_Rate, Batch_Size, and Mean_AUC.
    - output_path (str, optional): Path to save the heatmap image. If None, displays the plot.
    """
    # Pivot the DataFrame to create a matrix suitable for heatmap
    pivot_table = df.pivot(index='Learning_Rate', columns='Batch_Size', values='Mean_AUC').fillna(float('nan'))
    
    # Sort the axes for better visualization
    pivot_table = pivot_table.sort_index(ascending=True)
    pivot_table = pivot_table.sort_index(axis=1, ascending=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis")
    plt.title('Mean AUC Heatmap for FIRE Experiments')
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {output_path}")
    else:
        plt.show()

def main():
    # Define the path to the FIRE directory
    current_directory = os.getcwd()
    dir = os.path.join(current_directory, 'out', 'grid', 'FIRE', 'SIREN')
    # Check if the FIRE directory exists
    if not os.path.exists(dir):
        print(f"The directory {dir} does not exist.")
        return
    
    # Extract results
    df = extract_results(dir)
    
    if df.empty:
        print("No results found. Please check the directory structure and results.txt files.")
        return
    
    print("Extracted Results:")
    print(df)
    
    # Save the aggregated results to a CSV file
    aggregated_results_file = os.path.join(dir, 'aggregated_results.csv')
    df.to_csv(aggregated_results_file, index=False)
    print(f"Aggregated results saved to {aggregated_results_file}")
    
    # Visualize the results
    heatmap_path = os.path.join(dir, f'{dir}_heatmap.png')
    visualize_results(df, output_path=heatmap_path)

if __name__ == "__main__":
    # main()
    visualize_results(pd.read_csv('/home/mateo/projects/ai/IDIR/out/grid/FIRE/aggregated_results.csv'), output_path=os.path.join(os.getcwd(), 'out', 'grid', 'FIRE', 'SIREN_heatmap.png'))