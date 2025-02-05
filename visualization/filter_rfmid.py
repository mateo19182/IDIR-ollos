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
    og_img = data['og_img'] #imaxe orixinal
    geo_img = data['geo_img'] #imaxe cunha transformación xeométrica aleatoria (dentro duns parámetros)
    clr_img = data['clr_img'] #imaxe cunha transformación de cor aleatoria (dentro duns parámetros)
    full_img = data['full_img'] #imaxe coas dúas transformacións aplicadas
    mask = data['mask'] #máscara orixinal, aplicable á imaxe orixinal é a de cor
    geo_mask = data['geo_mask'] #máscara coa transformación xeométrica, aplicable á geo e á full

    matrix = data['matrix'] #matriz de transformación coa cal se conseguen a geo e a full
    inv_matrix = data['inv_matrix'] #matriz inversa que permite pasar da geo/full á og/clr

    images = [og_img, geo_img, clr_img, full_img] 
    image_names = ['Original Image', 'Geometric Image', 'Color Image', 'Full Image']
    #display_images(images, image_names)
    grayscale_images = np.dot(images, [0.2989, 0.5870, 0.1140]) #convertir a grayscale

    og_img = torch.tensor(grayscale_images[0], dtype=torch.float)
    geo_img = torch.tensor(grayscale_images[1], dtype=torch.float)
    clr_img = torch.tensor(grayscale_images[2], dtype=torch.float)
    full_img = torch.tensor(grayscale_images[3], dtype=torch.float)
    #print(og_img.shape) = torch.Size([1708, 1708])
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
    """
    Compute the Frobenius norm of the difference between the provided transformation
    matrix and its corresponding identity matrix.
    
    Args:
        matrix (np.ndarray): Transformation matrix from the .npz file.
        
    Returns:
        float: The computed deformation metric.
    """
    # Remove extra dimensions if necessary (e.g., (1, 3, 3) -> (3, 3))
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix.squeeze(0)
    
    # Determine identity based on the expected shape of the transformation matrix.
    if matrix.shape[0] == 2 and matrix.shape[1] == 3:
        identity = np.array([[1, 0, 0],
                             [0, 1, 0]], dtype=matrix.dtype)
    elif matrix.shape[0] == 3 and matrix.shape[1] == 3:
        identity = np.eye(3, dtype=matrix.dtype)
    else:
        identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    
    return np.linalg.norm(matrix - identity, 'fro')

def overlay_grid(ax, shape, grid_spacing=50, color='white', linestyle='--', linewidth=1):
    """
    Overlay a standard (regular, undistorted) grid on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): The axes on which to draw.
        shape (tuple): (height, width) of the image.
        grid_spacing (int): Spacing in pixels between grid lines.
        color (str): Color of the grid lines.
        linestyle (str): The linestyle (e.g. '--').
        linewidth (int or float): The width of the grid lines.
    """
    h, w = shape
    # Vertical lines
    for x in range(0, w, grid_spacing):
        ax.plot([x, x], [0, h], color=color, linestyle=linestyle, linewidth=linewidth)
    # Horizontal lines
    for y in range(0, h, grid_spacing):
        ax.plot([0, w], [y, y], color=color, linestyle=linestyle, linewidth=linewidth)

def overlay_deformed_grid(ax, matrix, shape, grid_spacing=50, color='red', linestyle='--', linewidth=1):
    """
    Overlay a grid deformed by the transformation matrix on the given axes.
    
    Args:
        ax (matplotlib.axes.Axes): The axes on which to draw.
        matrix (np.ndarray): The transformation matrix.
        shape (tuple): (height, width) of the image.
        grid_spacing (int): Spacing in pixels between grid lines.
        color (str): Color of the deformed grid lines.
        linestyle (str): The linestyle (e.g. '--').
        linewidth (int or float): The width of the grid lines.
    """
    # Ensure matrix is in a 3x3 form.
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix.squeeze(0)
    if matrix.shape == (2, 3):
        # Augment the matrix to 3x3 by adding the homogeneous coordinate row.
        matrix = np.vstack([matrix, np.array([0, 0, 1], dtype=matrix.dtype)])
    
    h, w = shape
    # Vertical grid lines: for a fixed x, vary y.
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
        
    # Horizontal grid lines: for a fixed y, vary x.
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
    """
    Visualize the original and geometric images with an informative title
    and overlay grid lines to illustrate the deformation.
    
    Args:
        file_path (str): The source file path.
        og_img (torch.Tensor or np.ndarray): Original image.
        geo_img (torch.Tensor or np.ndarray): Geometric (transformed) image.
        matrix (np.ndarray): The transformation matrix used to generate the deformation.
        metric (float): Computed deformation metric.
        accepted (bool): Flag indicating whether the deformation is 'easy' or not.
    """
    label = "Accepted" if accepted else "Rejected"
    title = f"{label} (Metric: {metric:.4f})\n{os.path.basename(file_path)}"

    # Convert tensors to numpy if needed.
    if isinstance(og_img, torch.Tensor):
        og_img_np = og_img.numpy()
    else:
        og_img_np = og_img

    if isinstance(geo_img, torch.Tensor):
        geo_img_np = geo_img.numpy()
    else:
        geo_img_np = geo_img

    # Assume both images have the same shape.
    shape = og_img_np.shape

    # Create side-by-side plots for the two images.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image with standard grid overlay.
    axes[0].imshow(og_img_np, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    overlay_grid(axes[0], shape, grid_spacing=50, color='cyan')
    
    # Geometric image with deformed grid overlay.
    axes[1].imshow(geo_img_np, cmap='gray')
    axes[1].set_title("Geometric Image")
    axes[1].axis('off')
    overlay_deformed_grid(axes[1], matrix, shape, grid_spacing=50, color='magenta')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{os.path.basename(file_path)}.png")

def process_file(file_path, threshold):
    """
    Process a single .npz file:
      - Load images and transformation matrix.
      - Compute the deformation metric.
      - Return a tuple with the file path, metric, images and matrix,
        along with a status flag ("accepted" or "rejected").

    Args:
        file_path (str): Path to the .npz file.
        threshold (float): The metric threshold.

    Returns:
        tuple: (file_path, metric, og_img, geo_img, matrix, status) or None if error.
    """
    try:
        og_img, geo_img, clr_img, full_img, mask, geo_mask, matrix = load_image_RFMID(file_path)
        metric = compute_deformation_metric(matrix)
        status = "accepted" if metric <= threshold else "rejected"
        return file_path, metric, og_img, geo_img, matrix, status
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Filter RFMID dataset for easy deformations based on a transformation metric."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to folder containing .npz files of the RFMID dataset.")
    parser.add_argument("--threshold", type=float, default=100,
                        help="Threshold for the deformation metric; below this value is considered 'easy'.")
    parser.add_argument("--nvisual", type=int, default=0,
                        help="Number of examples to visualize for each accepted and rejected group.")
    parser.add_argument("-n", type=int, default=0,
                        help="Number of instances to process. A value of 0 processes all instances.")
    args = parser.parse_args()

    accepted_files = []
    rejected_files = []
    accepted_visual_examples = []
    rejected_visual_examples = []
    all_metrics = []
    
    # Create a full list of file paths.
    file_paths = [os.path.join(args.dataset_dir, f) for f in os.listdir(args.dataset_dir) if f.endswith(".npz")]

    # Limit the number of files if -n is provided (non-zero)
    if args.n > 0:
        file_paths = file_paths[:args.n]
    
    total_files = len(file_paths)
    print(f"Found {total_files} .npz files in '{args.dataset_dir}' to process.")

    # Process files concurrently.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the process_file function over all file paths with progress bar
        results = list(tqdm(executor.map(process_file, file_paths, [args.threshold]*len(file_paths)), 
                          total=len(file_paths),
                          desc="Processing files"))

    for idx, result in enumerate(results):
        if result is None:
            continue
        file_path, metric, og_img, geo_img, matrix, status = result
        all_metrics.append(metric)
        
        if metric <= args.threshold:
            accepted_files.append(file_path)
            if len(accepted_visual_examples) < args.nvisual:
                accepted_visual_examples.append((file_path, og_img, geo_img, matrix, metric))
        else:
            rejected_files.append(file_path)
            if len(rejected_visual_examples) < args.nvisual:
                rejected_visual_examples.append((file_path, og_img, geo_img, matrix, metric))
        
        status_str = "Accepted" if metric <= args.threshold else "Rejected"
        print(f"[{idx+1}/{total_files}] {os.path.basename(file_path)}: Metric = {metric:.4f} -> {status_str}")
    
    # Compute median and standard deviation for the metric.
    median_metric = np.median(all_metrics) if all_metrics else float('nan')
    std_metric = np.std(all_metrics) if all_metrics else float('nan')
    
    print("\nSummary:")
    print(f"Total processed: {total_files}")
    print(f"Accepted (easy deformations): {len(accepted_files)}")
    print(f"Rejected (complex deformations): {len(rejected_files)}")
    print(f"Median metric: {median_metric:.4f}")
    print(f"Standard Deviation: {std_metric:.4f}")

    # Visualize a selection of accepted files.
    print("\nVisualizing accepted examples:")
    for file_path, og_img, geo_img, matrix, metric in accepted_visual_examples:
        visualize_deformation(file_path, og_img, geo_img, matrix, metric, accepted=True)
    
    # Visualize a selection of rejected files.
    print("\nVisualizing rejected examples:")
    for file_path, og_img, geo_img, matrix, metric in rejected_visual_examples:
        visualize_deformation(file_path, og_img, geo_img, matrix, metric, accepted=False)
    
    # Optionally, save the accepted file paths to a text file.
    accepted_file_path = os.path.join(args.dataset_dir, "accepted_files_" + str(args.threshold) + ".txt")
    with open(accepted_file_path, "w") as f:
        for af in accepted_files:
            f.write(af + "\n")
        f.write("accepted " + str(len(accepted_files)) + " out of " + str(total_files) + " files")
    print(f"Accepted file list saved to: {accepted_file_path}")

if __name__ == "__main__":
    main() 