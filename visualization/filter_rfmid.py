import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import concurrent.futures  # Added for parallel processing
from tqdm import tqdm

def load_image_RFMID(folder):
    if not os.path.exists(folder):
        return None
    data = np.load(folder)
    og_img = data['og_img']  # imaxe orixinal
    geo_img = data['geo_img']  # imaxe cunha transformación xeométrica aleatoria
    clr_img = data['clr_img']  # imaxe cunha transformación de cor aleatoria
    full_img = data['full_img']  # imaxe coas dúas transformacións aplicadas
    mask = data['mask']
    geo_mask = data['geo_mask']
    matrix = data['matrix']
    inv_matrix = data['inv_matrix']
    images = [og_img, geo_img, clr_img, full_img]
    image_names = ['Original Image', 'Geometric Image', 'Color Image', 'Full Image']
    grayscale_images = np.dot(images, [0.2989, 0.5870, 0.1140])
    og_img = torch.tensor(grayscale_images[0], dtype=torch.float)
    geo_img = torch.tensor(grayscale_images[1], dtype=torch.float)
    clr_img = torch.tensor(grayscale_images[2], dtype=torch.float)
    full_img = torch.tensor(grayscale_images[3], dtype=torch.float)
    return (
        og_img,
        geo_img,
        clr_img,
        full_img,
        mask,
        geo_mask,
        matrix
    )

def compute_deformation_metric(matrix):
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix.squeeze(0)
    if matrix.shape[0] == 2 and matrix.shape[1] == 3:
        identity = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype=matrix.dtype)
    elif matrix.shape[0] == 3 and matrix.shape[1] == 3:
        identity = np.eye(3, dtype=matrix.dtype)
    else:
        identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    return np.linalg.norm(matrix - identity, 'fro')

def overlay_grid(ax, shape, grid_spacing=50, color='white', linestyle='--', linewidth=1):
    h, w = shape
    for x in range(0, w, grid_spacing):
        ax.plot([x, x], [0, h], color=color, linestyle=linestyle, linewidth=linewidth)
    for y in range(0, h, grid_spacing):
        ax.plot([0, w], [y, y], color=color, linestyle=linestyle, linewidth=linewidth)

def overlay_deformed_grid(ax, matrix, shape, grid_spacing=50, color='red', linestyle='--', linewidth=1):
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix.squeeze(0)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, np.array([0, 0, 1], dtype=matrix.dtype)])
    h, w = shape
    xs = np.arange(0, w, grid_spacing)
    ys = np.linspace(0, h, num=100)
    for x in xs:
        pts = []
        for y in ys:
            point = np.array([x, y, 1])
            trans_point = matrix.dot(point)
            pts.append(trans_point[:2])
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)
    ys = np.arange(0, h, grid_spacing)
    xs_line = np.linspace(0, w, num=100)
    for y in ys:
        pts = []
        for x in xs_line:
            point = np.array([x, y, 1])
            trans_point = matrix.dot(point)
            pts.append(trans_point[:2])
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)

def visualize_deformation(file_path, og_img, geo_img, matrix, metric, accepted=True):
    label = "Accepted" if accepted else "Rejected"
    title = f"{label} (Metric: {metric:.4f})\n{os.path.basename(file_path)}"
    if isinstance(og_img, torch.Tensor):
        og_img_np = og_img.numpy()
    else:
        og_img_np = og_img
    if isinstance(geo_img, torch.Tensor):
        geo_img_np = geo_img.numpy()
    else:
        geo_img_np = geo_img
    shape = og_img_np.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(og_img_np, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    overlay_grid(axes[0], shape, grid_spacing=50, color='cyan')
    axes[1].imshow(geo_img_np, cmap='gray')
    axes[1].set_title("Geometric Image")
    axes[1].axis('off')
    overlay_deformed_grid(axes[1], matrix, shape, grid_spacing=50, color='magenta')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{os.path.basename(file_path)}.png")

def process_file(file_path, threshold):
    try:
        og_img, geo_img, clr_img, full_img, mask, geo_mask, matrix = load_image_RFMID(file_path)
        metric = compute_deformation_metric(matrix)
        if threshold is not None:
            status = "accepted" if metric <= threshold else "rejected"
        else:
            status = "binned"
        return file_path, metric, og_img, geo_img, matrix, status
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Filter RFMID dataset for easy deformations based on a transformation metric. "
                    "If --threshold is not set, the files will be binned in metric ranges of 150."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to folder containing .npz files of the RFMID dataset.")
    # Set threshold default to None so that not providing it triggers binning.
    parser.add_argument("--threshold", type=float, default=None,
                        help="If set, threshold for the deformation metric; otherwise, files are binned in groups of 150.")
    parser.add_argument("--nvisual", type=int, default=0,
                        help="Number of examples to visualize for each group.")
    parser.add_argument("-n", type=int, default=0,
                        help="Number of instances to process. A value of 0 processes all instances.")
    args = parser.parse_args()

    all_metrics = []
    results_list = []

    file_paths = [os.path.join(args.dataset_dir, f) for f in os.listdir(args.dataset_dir) if f.endswith(".npz")]
    if args.n > 0:
        file_paths = file_paths[:args.n]
    total_files = len(file_paths)
    print(f"Found {total_files} .npz files in '{args.dataset_dir}' to process.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, file_paths, [args.threshold] * len(file_paths)),
                          total=len(file_paths),
                          desc="Processing files"))
    for idx, result in enumerate(results):
        if result is None:
            continue
        file_path, metric, og_img, geo_img, matrix, status = result
        all_metrics.append(metric)
        results_list.append(result)
        if args.threshold is not None:
            status_str = "Accepted" if metric <= args.threshold else "Rejected"
            print(f"[{idx+1}/{total_files}] {os.path.basename(file_path)}: Metric = {metric:.4f} -> {status_str}")
        else:
            print(f"[{idx+1}/{total_files}] {os.path.basename(file_path)}: Metric = {metric:.4f}")

    if args.threshold is not None:
        accepted_files = [r[0] for r in results_list if r[1] <= args.threshold]
        median_metric = np.median(all_metrics) if all_metrics else float('nan')
        std_metric = np.std(all_metrics) if all_metrics else float('nan')
        print("\nSummary:")
        print(f"Total processed: {total_files}")
        print(f"Accepted (easy deformations): {len(accepted_files)}")
        print(f"Median metric: {median_metric:.4f}")
        print(f"Standard Deviation: {std_metric:.4f}")

        print("\nVisualizing accepted examples:")
        accepted_visual_examples = [r for r in results_list if r[1] <= args.threshold][:args.nvisual]
        for file_path, metric, og_img, geo_img, matrix, _ in accepted_visual_examples:
            visualize_deformation(file_path, og_img, geo_img, matrix, metric, accepted=True)

        accepted_file_path = os.path.join(args.dataset_dir, "accepted_files_" + str(args.threshold) + ".txt")
        with open(accepted_file_path, "w") as f:
            for af in accepted_files:
                f.write(af + "\n")
            f.write("accepted " + str(len(accepted_files)) + " out of " + str(total_files) + " files")
        print(f"Accepted file list saved to: {accepted_file_path}")

    else:
        # Bin files in groups of 150 based on the metric.
        bins = {}
        for file_path, metric, og_img, geo_img, matrix, _ in results_list:
            bin_index = int(metric // 150)
            lower = bin_index * 150
            upper = lower + 150
            key = f"{lower}_{upper}"
            bins.setdefault(key, []).append((file_path, metric, og_img, geo_img, matrix))
        # Write each bin to its own file.
        for key, files in bins.items():
            bin_file_path = os.path.join(args.dataset_dir, f"accepted_files_{key}.txt")
            with open(bin_file_path, "w") as f:
                for item in files:
                    f.write(item[0] + "\n")
                f.write(f"accepted {len(files)} out of {total_files} files\n")
            print(f"Bin {key}: {len(files)} files. List saved to: {bin_file_path}")

        # Optionally, visualize examples from each bin.
        for key, files in bins.items():
            print(f"\nVisualizing examples for bin {key}:")
            for file_path, metric, og_img, geo_img, matrix in files[:args.nvisual]:
                visualize_deformation(file_path, og_img, geo_img, matrix, metric, accepted=True)

if __name__ == "__main__":
    main()