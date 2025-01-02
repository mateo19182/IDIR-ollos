import imageio
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import kornia.geometry.linalg as kn
import cv2
import SimpleITK as sitk
import pystrum
from scipy import integrate
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../visualization'))
import fig_vis


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(means, np.mean(difference))
    stds = np.append(stds, np.std(difference))

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    means = means[::-1]
    stds = stds[::-1]

    return means, stds

def compute_landmarks(network, landmarks_pre, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

    output = network(coordinate_tensor.cuda())

    delta = output.cpu().detach().numpy() * (scale_of_axes)

    return landmarks_pre + delta, delta

def load_image_DIRLab(variation=1, folder=r"D:/Data/DIRLAB/Case"):
    # Size of data, per image pair
    image_sizes = [
        0,
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    # Scale of data, per image pair
    voxel_sizes = [
        0,
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    shape = image_sizes[variation]

    folder = folder + str(variation) + r"Pack" + os.path.sep

    # Images
    dtype = np.dtype(np.int16)

#   with open(folder + r"Images/case" + str(variation) + "_T00_s.img", "rb") as f:
    with open(folder + r"Images/case" + str(variation) + "_T00.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_insp = data.reshape(shape)

#   with open(folder + r"Images/case" + str(variation) + "_T50_s.img", "rb") as f:
    with open(folder + r"Images/case" + str(variation) + "_T50.img", "rb") as f:
        data = np.fromfile(f, dtype)
    image_exp = data.reshape(shape)

#   imgsitk_in = sitk.ReadImage(folder + r"Masks/case" + str(variation) + "_T00_s.mhd")
    imgsitk_in = sitk.ReadImage(folder + r"Masks/case" + str(variation) + "_T00.mhd")
    print(imgsitk_in)
    mask = np.clip(sitk.GetArrayFromImage(imgsitk_in), 0, 1)
    image_insp = torch.FloatTensor(image_insp)
    image_exp = torch.FloatTensor(image_exp)

    # Landmarks
    with open(
        folder + r"ExtremePhases/Case" + str(variation) + "_300_T00_xyz.txt"
    ) as f:
        landmarks_insp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    with open(
        folder + r"ExtremePhases/Case" + str(variation) + "_300_T50_xyz.txt"
    ) as f:
        landmarks_exp = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    landmarks_insp[:, [0, 2]] = landmarks_insp[:, [2, 0]]
    landmarks_exp[:, [0, 2]] = landmarks_exp[:, [2, 0]]

    return (
        image_insp,
        image_exp,
        landmarks_insp,
        landmarks_exp,
        mask,
        voxel_sizes[variation],
    )

def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_coordinate_slice(dims=(28, 28), dimension=0, slice_pos=0, gpu=True):
    """Make a coordinate tensor."""

    dims = list(dims)
    dims.insert(dimension, 1)

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor[dimension] = torch.linspace(slice_pos, slice_pos, 1)
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

def make_masked_coordinate_tensor(mask, dims=(28, 28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

#----------------------------------------------------------------------

def create_unique_dir(base_dir):
    suffix = 0
    while True:
        dir_name = f"{base_dir}{f'_{suffix}' if suffix else ''}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        suffix += 1

def make_masked_coordinate_tensor_2d(mask, dims):
    """Make a coordinate tensor."""
    mask = np.ceil(mask).clip(0, 1)
    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(2)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=2)
    coordinate_tensor = coordinate_tensor.view([-1, 2])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]
    coordinate_tensor = coordinate_tensor.cuda()
    return coordinate_tensor

def make_coordinate_tensor_2d(dims=(28, 28), gpu=True):
    """Make a 2D coordinate grid."""
    
    # Create a meshgrid for the dimensions
    x = torch.linspace(-1, 1, dims[1])
    y = torch.linspace(-1, 1, dims[0])
    xv, yv = torch.meshgrid(x, y, indexing="ij")
    
    # Stack the coordinate grids
    coordinate_grid = torch.stack([xv, yv], dim=-1) # Shape: [dims[0], dims[1], 2]
    
    # Flatten the grid to have a list of coordinates
    coordinate_grid = coordinate_grid.view(-1, 2) # Shape: [dims[0]*dims[1], 2]
    
    # Move to GPU if requested
    if gpu:
        coordinate_grid = coordinate_grid.cuda()
    return coordinate_grid

def weight_mask(mask, fixed_image, save=False):
    # Ensure inputs are on CPU
    # mask = mask.cpu().float()
    # img = fixed_image.cpu().float()
    
    # Apply histogram equalization
    img_np = fixed_image.cpu().numpy().astype(np.uint8)
    # img_equalized = cv2.equalizeHist(img_np)
    img_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_np)  
    img = torch.tensor(img_clahe, dtype=torch.float)
    
    # Normalize the image to have values between 0 and 1
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(img.numpy(), cv2.CV_64F, 1, 0, ksize=17)
    grad_y = cv2.Sobel(img.numpy(), cv2.CV_64F, 0, 1, ksize=17)
    
    # Compute gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    disk_mask = get_optical_disk_mask(fixed_image.cpu().numpy())
    
    if disk_mask is None:
        disk_mask = np.zeros_like(grad_mag, dtype=grad_mag.dtype)
    else:
        disk_mask = disk_mask.astype(grad_mag.dtype)
    
    grad_mag = grad_mag + (disk_mask)  # not workinggg, si multiplico si que va pero quiero sumarr
    
    # Apply mask
    grad_mag *= mask
    
    # Avoid computing gradients on the borders of the mask
    mask_np = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(grad_mag, contours, -1, 0, thickness=5)

    # # Increase the contrast by making darker parts more dark
    # grad_mag = np.power(grad_mag, 1.5)  # the higher the power, the less the dark parts will be sampled
    # # Dilate the bright parts of the gradient magnitude
    # # grad_mag = cv2.morphologyEx(grad_mag, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # # Apply Gaussian blur to the gradient magnitude
    # grad_mag = cv2.GaussianBlur(grad_mag, (51, 51), 0)
    # grad_mag = np.power(grad_mag, 1.25) 
    
    # Normalize to create a probability distribution
    eps = 1e-8
    total = grad_mag.sum() + eps
    weights_np = grad_mag / total
    
    if save:
        fig_vis.save_weight_map_as_image(weights_np)

    # Flatten and apply mask
    mask = mask.flatten() > 0
    weights = torch.from_numpy(weights_np).flatten()[mask].to('cuda')
    
    return weights

def get_optical_disk_mask(image, initial_thresh=175, max_iter=5):
    """
    Repeatedly thresholds the image, checking if the largest contour is within
    an acceptable size range. If not, adjusts threshold and tries again.
    """
    threshold_val = initial_thresh
    desired_min_size = 25
    desired_max_size = 600
    step = 10

    for _ in range(max_iter):
        # Threshold the image
        _, thresh = cv2.threshold(image.astype(np.uint8), threshold_val, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"No contours found at threshold={threshold_val}, adjusting upward.")
            threshold_val = min(threshold_val + step, 255)
            continue
        
        # Find the largest contour (assumed to be the optic disc)
        largest_contour = max(contours, key=cv2.contourArea)
        size = len(largest_contour)
        print(f"Threshold={threshold_val} -> Largest contour size={size}")

        if size > desired_max_size:
            print("Contour too big, increasing threshold.")
            threshold_val = min(threshold_val + step, 255)
        elif size < desired_min_size:
            print("Contour too small, decreasing threshold.")
            threshold_val = max(threshold_val - step, 0)
        else:
            print("Contour size within desired range, creating mask.")
            # Create a mask for the optic disc
            mask = np.zeros(image.shape, np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            # Dilate the mask to include surrounding areas
            mask = cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=2)
            # Apply Gaussian blur to smooth the mask
            mask = cv2.GaussianBlur(mask, (25, 25), 0)
            cv2.imwrite('optic_disk_mask.png', mask)
            print("Optic disk mask saved as 'optic_disk_mask.png'")
            return mask
    
    print("Unable to find suitable contour within iteration limit.")
    return None

def bilinear_interpolation(input_array, x_indices, y_indices):
    # input_array.shape = #torch.Size([2912, 2912])
    # x_indices.shape = torch.Size([1000000])

    # Convert indices from [-1, 1] to [0, width-1] or [0, height-1]
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5

    # Get the four surrounding pixel coordinates
    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp the coordinates to be within the image bounds
    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)

    # Calculate the interpolation weights
    x = x_indices - x0
    y = y_indices - y0
    
    # Perform bilinear interpolation
    output = (
        input_array[x0, y0] * (1 - x) * (1 - y) +
        input_array[x1, y0] * x * (1 - y) +
        input_array[x0, y1] * (1 - x) * y +
        input_array[x1, y1] * x * y
    )
    
    return output

def simple_bilinear_interpolation_point(dfv, x, y, scale):
    # Scale the coordinates
    x_scaled = x * scale
    y_scaled = y * scale

    # Get the integer parts
    x0 = int(np.floor(x_scaled))
    x1 = x0 + 1
    y0 = int(np.floor(y_scaled))
    y1 = y0 + 1

    # Ensure the indices are within bounds
    x0 = np.clip(x0, 0, dfv.shape[1] - 1)
    x1 = np.clip(x1, 0, dfv.shape[1] - 1)
    y0 = np.clip(y0, 0, dfv.shape[0] - 1)
    y1 = np.clip(y1, 0, dfv.shape[0] - 1)

    # Get the fractional parts
    x_frac = x_scaled - x0
    y_frac = y_scaled - y0

    # Get the values at the four surrounding points
    Q11 = dfv[y0, x0]
    Q21 = dfv[y0, x1]
    Q12 = dfv[y1, x0]
    Q22 = dfv[y1, x1]

    # Perform bilinear interpolation
    R1 = (1 - x_frac) * Q11 + x_frac * Q21
    R2 = (1 - x_frac) * Q12 + x_frac * Q22
    P = (1 - y_frac) * R1 + y_frac * R2

    return P

def transform_point(x, y, transformation_matrix):
    # Create a homogeneous coordinate vector
    point = np.array([x, y, 1])
    # Perform the matrix multiplication
    transformed_point = np.dot(transformation_matrix[0], point)
    # Extract the x and y coordinates (discarding the homogeneous coordinate)
    new_x, new_y = transformed_point[:2]
    return new_x, new_y

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

def load_image_FIRE(index, folder):
    images_folder = os.path.join(folder, 'Images')
    files = os.listdir(images_folder)
    files.sort()
    fixed_image = imageio.imread(os.path.join(images_folder, files[index*2]))
    moving_image = imageio.imread(os.path.join(images_folder, files[(index*2)+1]))
    #print(os.path.join(images_folder, files[index*2]))

    ground_folder = os.path.join(folder, 'Ground Truth')
    files = os.listdir(ground_folder)
    files.sort()
    ground_truth = []
    #print(os.path.join(ground_folder, files[index]))
    with open(os.path.join(ground_folder, files[index]), 'r') as f:
        for line in f:
            line = line.strip().split()
            ground_truth.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])    
    images = [fixed_image, moving_image] 
    image_names = ['fixed_image', 'moving_image']
    #display_images(images, image_names)
    grayscale_images = np.dot(images, [0.2989, 0.5870, 0.1140])
    green_channel_images = [img[:, :, 1] for img in images]
    fixed_image = torch.tensor(grayscale_images[0], dtype=torch.float)
    moving_image = torch.tensor(grayscale_images[1], dtype=torch.float)
    return (
        fixed_image,
        moving_image,
        ground_truth,
        grayscale_images[0], 
        grayscale_images[1]
    )

def plot_loss_curves(data_loss_list, total_loss_list, epochs, save_path):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, data_loss_list, label='Data Loss',  linestyle='-', color='blue')
    plt.plot(epochs_range, total_loss_list, label='Total Loss',  linestyle='--', color='red')
    plt.title('Loss Curves over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(os.path.join(save_path,'loss.svg'), format='svg')

def calculate_metrics(thresholds, success_rates, dists, og_dists, save_path):
    # Calculate AUC
    auc = integrate.trapezoid(success_rates, thresholds)
    
    threshold_90 = None
    for threshold, rate in zip(thresholds, success_rates):
        if rate >= 0.9:
            threshold_90 = threshold
            break

    mean_dist = np.mean(dists)
    
    metrics = os.path.join(save_path, 'metrics.txt')
    with open(metrics, 'w') as f:
        f.write(f"Mean Distance: {mean_dist:.4f}\n")
        f.write(f"Area Under the Curve (AUC): {auc:.4f}\n")
        if threshold_90 is not None:
            f.write(f"Threshold for 90% success rate: {threshold_90:.4f}\n")
        else:
            f.write("Threshold for 90% success rate: Not achieved\n")
        if sum(dists)<(sum(og_dists)*0.99):
            f.write("Successful\n")
            success = True
        else:
            f.write("Unsuccessful\n")
            success = False
   
    return [auc, mean_dist, success_rates, success]

def test_FIRE(dfv, ground_truth, vol_shape, save_path, reg_img, fixed_image, moving_image):
    scale = vol_shape[0]/2912
    dists, og_dists, success_rates = [], [], []
    thresholds = np.arange(0, 25, 0.1)  # 0.1 to 25.0 in steps of 0.1
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    mapx, mapy = np.meshgrid(np.arange(-1,1,2/vol_shape[0]), np.arange(-1,1,2/vol_shape[0]))
    dfs = np.stack([mapy, mapx], axis=2)
    
    #mapx = mapx - 0.2
    #mapy = mapy - 0.1
    #dfm = np.stack([mapy, mapx], axis=2).reshape((vol_shape[0]*vol_shape[1], 2)) 
    #dfv = np.stack([mapy, mapx], axis=2)

    grid =  torch.from_numpy(pystrum.pynd.ndutils.bw_grid((2912, 2912), spacing=64, thickness=3))
    tr1 = bilinear_interpolation(grid, torch.from_numpy( dfv[:, 0]), torch.from_numpy(dfv[:, 1]))
    img = bilinear_interpolation(moving_image, torch.from_numpy(dfv[:, 0]), torch.from_numpy(dfv[:, 1]))
    tr1 = tr1.reshape(vol_shape).numpy()

    img = cv2.resize(img.reshape(vol_shape).numpy(), (2912, 2912))

    axes[0].imshow(fixed_image, cmap='gray')
    axes[0].set_title('Fixed Image')    
    axes[1].imshow(img, cmap='gray')
    axes[1].set_title('Registered Image')
    axes[2].imshow(moving_image, cmap='gray')
    axes[2].set_title('Moving Image')
    axes[3].imshow(tr1, cmap='gray')
    axes[3].set_title('grid')

    dfv=dfv.reshape((vol_shape[0], vol_shape[1], 2))

    for points in ground_truth:
        fixed_x= float(points[0])
        fixed_y=float(points[1])
        moving_x = float(points[2])
        moving_y = float(points[3])

        #dy, dx = dfv[int(np.round(y*scale)), int(np.round(x*scale))]
        # oy, ox = dfs[int(np.round(y_truth*scale)), int(np.round(x_truth*scale))]
        # oy, ox = simple_bilinear_interpolation_point(dfs, x_truth, y_truth, scale)
        dy, dx = simple_bilinear_interpolation_point(dfv, moving_x, moving_y, scale)

        # dx = ox+(ox-dx)
        # dy = oy+(oy-dy)

        x_res= (dx  + 1 ) * (2912 - 1) * 0.5
        y_res= (dy  + 1 ) * (2912 - 1) * 0.5

        #print("x: {} y: {} x_truth: {} y_truth: {} x_res: {} y_res:{} ".format(x, y, x_truth, y_truth, x_res, y_res))
        dist = np.linalg.norm(np.array((moving_x, moving_y)) - np.array((x_res, y_res)))
        og_dist = np.linalg.norm(np.array((moving_x, moving_y)) - np.array((fixed_x, fixed_y)))

        axes[0].scatter(fixed_x, fixed_y, c='w', s=2)  
        axes[1].scatter(fixed_x, fixed_y, c='w', s=1) 
        axes[2].scatter(moving_x, moving_y, c='g', s=2)  
        axes[2].scatter(x_res, y_res, c='b', s=1)  
        axes[2].annotate(f'{dist:.2f}', (x_res, y_res))
        axes[2].plot([moving_x, x_res], [moving_y, y_res], linestyle='-', color='red', linewidth=0.2)

        dists.append(dist)
        og_dists.append(og_dist)

    # with open(os.path.join(save_path,'dists.txt'), 'w') as f:
    #     for item in dists:
    #         f.write("%s\n" % item)
    #     f.write("Mean: %s\n" % np.mean(dists))
        
    for threshold in thresholds:
        res = 0
        for dist in dists:
            if dist < threshold:
                res+=1
        success_rates.append(res/len(dists))

    fig_path = os.path.join(save_path, 'plot.png')
    plt.savefig(fig_path, format='png')
    print("Plot saved at: ", fig_path)

    # plt.figure()
    # plt.plot(thresholds, success_rates)
    # plt.xlabel('Threshold')
    # plt.ylabel('Success Rate')
    # plt.title('Success Rate vs Threshold')
    # plt.ylim([0, 1]) 
    # fig_path = os.path.join(save_path, 'eval.png')
    # plt.savefig(fig_path, format='png')
    return calculate_metrics(thresholds, success_rates, dists, og_dists, save_path)

def test_RFMID(dfv, matrix, vol_shape, save_path, reg_img, fixed_image, moving_image, mask):
    scale = vol_shape[0]/mask.shape[0]
    # print(scale)
    dists, og_dists, success_rates = [], [], []
    thresholds = np.arange(0.1, 25.1, 0.1)  # 0.1 to 25.0 in steps of 0.1
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    # if np.array_equal(img, fixed_image.numpy()):
    #     print("The images are the same.")
    # else:
    #     print("The images are different.")
    # mapx, mapy = np.meshgrid(np.arange(-1,1,2/vol_shape[0]), np.arange(-1,1,2/vol_shape[0]))
    # dfs = np.stack([mapy, mapx], axis=2)
    # mapx = mapx - 0.2
    # mapy = mapy
    # dfm = np.stack([mapy, mapx], axis=2)
    # dfm=dfm.reshape((vol_shape[0]*vol_shape[1], 2))

    grid =  torch.from_numpy(pystrum.pynd.ndutils.bw_grid((vol_shape[0], vol_shape[1]), spacing=64, thickness=3))
    tr1 = bilinear_interpolation(grid, torch.from_numpy(dfv[:, 0]), torch.from_numpy(dfv[:, 1]))
    img = bilinear_interpolation(moving_image, torch.from_numpy(dfv[:, 0]), torch.from_numpy(dfv[:, 1]))
    tr1 = tr1.reshape(vol_shape).numpy()
    img = cv2.resize(img.reshape(vol_shape).numpy(), (mask.shape[0], mask.shape[1]))

    axes[0].imshow(fixed_image, cmap='gray')
    axes[0].set_title('Fixed Image')
    axes[1].imshow(img, cmap='gray')
    axes[1].set_title('Registered Image')
    axes[2].imshow(moving_image, cmap='gray')
    axes[2].set_title('Moving Image')    
    axes[3].imshow(tr1, cmap='gray')
    axes[3].set_title('grid')

    
    dfv=dfv.reshape((vol_shape[0], vol_shape[1], 2))
    # dfm=dfm.reshape((vol_shape[0], vol_shape[1], 2))

    step = 200
    x, y = np.meshgrid(
        np.arange(step, mask.shape[0] - step, step),
        np.arange(step, mask.shape[1] - step, step)
    )
    xy_points = np.column_stack((x.ravel(), y.ravel()))

    ground_truth = []
    for points in xy_points:
        if mask[points[0], points[1]]:
            x, y= transform_point( points[0], points[1], matrix)
            ground_truth.append([points[0], points[1], x, y])


    for points in ground_truth:
        fixed_x= float(points[0])
        fixed_y=float(points[1])
        moving_x = float(points[2])
        moving_y = float(points[3])     

        # dy, dx = dfv[int(np.round(y*scale)), int(np.round(x*scale))]
        # my, mx = dfm[int(np.round(y_truth*scale)), int(np.round(x_truth*scale))]
        # oy, ox = simple_bilinear_interpolation_point(dfs, x_truth, y_truth, scale)
        dy, dx = simple_bilinear_interpolation_point(dfv, fixed_x, fixed_y, scale)
        
        # dx = ox+(ox-dx)
        # dy = oy+(oy-dy)

        x_res= (dx  + 1 ) * (1708 - 1) * 0.5
        y_res= (dy  + 1 ) * (1708 - 1) * 0.5
        #print("x: {} y: {} x_truth: {} y_truth: {} x_res: {} y_res:{} ".format(x, y, x_truth, y_truth, x_res, y_res))
        dist = np.linalg.norm(np.array((moving_x, moving_y)) - np.array((x_res, y_res)))
        og_dist = np.linalg.norm(np.array((fixed_x, fixed_y)) - np.array((moving_x, moving_y)))
        
        axes[0].scatter(fixed_x, fixed_y, c='w', s=2)  
        axes[1].scatter(fixed_x, fixed_y, c='w', s=1) 
        axes[2].scatter(moving_x, moving_y, c='g', s=2)  
        axes[2].scatter(x_res, y_res, c='b', s=1)  
        axes[2].annotate(f'{dist:.2f}', (x_res, y_res))
        axes[2].plot([moving_x, x_res], [moving_y, y_res], linestyle='-', color='red', linewidth=0.2)
        # axes[2].plot([fixed_x, x_res], [fixed_y, y_res], linestyle='-', color='yellow', linewidth=0.2)

        dists.append(dist)
        og_dists.append(og_dist)

    # with open(os.path.join(save_path,'dists.txt'), 'w') as f:
    #     for item in dists:
    #         f.write("%s\n" % item)
    #     f.write("Mean: %s\n" % np.mean(dists))

    for threshold in thresholds:
        res = 0
        for dist in dists:
            if dist < threshold:
                res+=1
        success_rates.append(res/len(dists))

    fig_path = os.path.join(save_path, 'plot.png')
    plt.savefig(fig_path, format='png')
    print("Plot saved at: ", fig_path)
    # plt.figure()
    # plt.plot(thresholds, success_rates)
    # plt.xlabel('Threshold')
    # plt.ylabel('Success Rate')
    # plt.title('Success Rate vs Threshold')
    # plt.ylim([0, 1]) 
    # fig_path = os.path.join(save_path, 'eval.png')
    # plt.savefig(fig_path, format='png')
    

    # fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # # Create and display checkerboard image
    # checker_img = fig_vis.create_checkerboard(fixed_image, img)
    # ax[0].imshow(checker_img, cmap='gray')
    # ax[0].set_title('Checkerboard: Fixed vs. Registered')

    # color_mixed = fig_vis.color_overlay(fixed_image, img)
    # ax[1].imshow(color_mixed)
    # ax[1].set_title('Color Overlay: Red=Fixed, Green=Registered (Yellow=Match)')
    # # Save the figure
    # fig.tight_layout()
    # fig_path = os.path.join(save_path, 'combined_visualization.png')
    # plt.savefig(fig_path, format='png')

    return calculate_metrics(thresholds, success_rates, dists, og_dists, save_path)


def clean_memory():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for obj in dir():
            if torch.is_tensor(eval(obj)):
                del globals()[obj]
