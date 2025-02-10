import imageio
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import cv2
import SimpleITK as sitk
from scipy import integrate
from skimage import io, filters
import sys
import math
from models import models
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '../visualization'))
import fig_vis

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def make_uniform_coordinate_tensor(mask, dims, batch_size):
    #Fibonacci lattice points in polar form place points on a circle by assigning each point 
    # a radius (sqrt i / N) and an angle (2πi / φ²)
    # This creates a relatively uniform distribution of points across a circular area.
    # https://observablehq.com/@meetamit/fibonacci-lattices
    # Step 1: Determine the mask radius and valid region
    mask = np.ceil(mask).clip(0, 1)
    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(2)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=2)
    coordinate_tensor = coordinate_tensor.view([-1, 2])
    masked_coords = coordinate_tensor[mask.flatten() > 0, :]
    mask_radius = torch.norm(masked_coords, dim=1).max()

    # Step 2: Generate Fibonacci lattice points with small random perturbation
    indices = torch.arange(0, batch_size, dtype=torch.float32) + 0.5
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    r = torch.sqrt(indices / batch_size) * mask_radius
    theta = 2 * math.pi * indices / (phi ** 2)
    
    # Add small random perturbation
    r = r + torch.randn_like(r) * 0.005  # Small radial perturbation
    theta = theta + torch.randn_like(theta) * 0.005  # Small angular perturbation

    # Step 3: Convert to Cartesian coordinates
    x_coords = r * torch.cos(theta)
    y_coords = r * torch.sin(theta)
    even_coords = torch.stack((x_coords, y_coords), dim=1)

    # Step 4: Map coordinates back to mask indices
    x_indices = ((even_coords[:, 0] + 1) / 2 * (dims[0] - 1)).long().clamp(0, dims[0] - 1)
    y_indices = ((even_coords[:, 1] + 1) / 2 * (dims[1] - 1)).long().clamp(0, dims[1] - 1)

    # Step 5: Find points outside the mask and project them to the closest valid point
    # mask = torch.from_numpy(mask)
    # valid_mask = mask[x_indices, y_indices] > 0
    # invalid_indices = torch.where(~valid_mask)[0]

    # if len(invalid_indices) > 0: 
    #     valid_points = coordinate_tensor[mask.flatten() > 0, :]
    #     invalid_points = even_coords[invalid_indices]
    #     distances = torch.cdist(invalid_points, valid_points, p=2) # too memory xpensive
    #     closest_valid_indices = torch.argmin(distances, dim=1)
    #     closest_valid_points = valid_points[closest_valid_indices]
    #     even_coords[invalid_indices] = closest_valid_points
    # Step 5: Find valid points using the mask
    mask = torch.from_numpy(mask)
    valid_mask = mask[x_indices, y_indices] > 0
    valid_coords = even_coords[valid_mask]

    # Step 6: Select required number of points
    if len(valid_coords) < batch_size:
        # If we don't have enough valid points, repeat existing ones
        num_repeats = (batch_size + len(valid_coords) - 1) // len(valid_coords)
        valid_coords = valid_coords.repeat(num_repeats, 1)
    
    # Step 6: Select the required number of points
    selected_coords = even_coords[:batch_size]

    return selected_coords.cuda()

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
    # vessel_mask = get_vessel_mask(fixed_image.cpu().numpy())
    # vessel_mask = sift_based_vessel_mask(fixed_image.cpu().numpy())
    if disk_mask is None:
        disk_mask = np.zeros_like(grad_mag, dtype=grad_mag.dtype)
    else:
        disk_mask = disk_mask.astype(grad_mag.dtype)
    
    
    factor = grad_mag.max() * 0.00025
    
    # grad_mag = grad_mag + (disk_mask)  # not workinggg, si multiplico si que va pero quiero sumarr
    grad_mag += factor*disk_mask  # or some other factor
    
    # Apply mask
    grad_mag *= mask
    
    # Avoid computing gradients on the borders of the mask
    mask_np = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(grad_mag, contours, -1, 0, thickness=5)

    # Increase the contrast by making darker parts more dark
    grad_mag = np.power(grad_mag, 1.5)  # the higher the power, the less the dark parts will be sampled
    # Dilate the bright parts of the gradient magnitude
    # grad_mag = cv2.morphologyEx(grad_mag, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Apply Gaussian blur to the gradient magnitude
    grad_mag = cv2.GaussianBlur(grad_mag, (75, 75), 0)
    # grad_mag = np.power(grad_mag, 1.1) 
    
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

def get_optical_disk_mask(image, initial_thresh=175, max_iter=10):
    """
    Repeatedly thresholds the image, checking if the largest contour is within
    an acceptable size range. If not, adjusts threshold and tries again.
    """
    threshold_val = initial_thresh
    desired_min_size = 150
    desired_max_size = 750
    step = 5

    for _ in range(max_iter):
        # Threshold the image
        _, thresh = cv2.threshold(image.astype(np.uint8), threshold_val, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # print(f"No contours found at threshold={threshold_val}, adjusting upward.")
            threshold_val = min(threshold_val + step, 255)
            continue
        
        # Find the largest contour (assumed to be the optic disc)
        largest_contour = max(contours, key=cv2.contourArea)
        size = len(largest_contour)
        print(f"Threshold={threshold_val} -> Largest contour size={size}")

        if size > desired_max_size:
            # print("Contour too big, increasing threshold.")
            threshold_val = min(threshold_val + step, 255)
        elif size < desired_min_size:
            # print("Contour too small, decreasing threshold.")
            threshold_val = max(threshold_val - step, 0)
        else:
            print(f"Contour size ={size}, creating mask at optic_disk_mask.png")
            # Create a mask for the optic disc
            mask = np.zeros(image.shape, np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            # Dilate the mask to include surrounding areas
            mask = cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=2)
            # Apply Gaussian blur to smooth the mask
            mask = cv2.GaussianBlur(mask, (25, 25), 0)
            cv2.imwrite('optic_disk_mask.png', mask)
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
    print(os.path.join(images_folder, files[index*2]))
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
    plt.close()

def calculate_metrics(thresholds, success_rates, dists, og_dists, save_path):
    # Calculate AUC
    auc = integrate.trapezoid(success_rates, thresholds)
    
    threshold_90 = None
    for threshold, rate in zip(thresholds, success_rates):
        if rate >= 0.9:
            threshold_90 = threshold
            break

    mean_dist = np.mean(dists)
    og_mean_dist = np.mean(og_dists)
    
    metrics = os.path.join(save_path, 'metrics.txt')
    with open(metrics, 'w') as f:
        f.write(f"Baseline Mean Distance: {og_mean_dist:.4f}\n")
        f.write(f"Mean Distance: {mean_dist:.4f}\n")
        f.write(f"Area Under the Curve (AUC): {auc:.4f}\n")
        if threshold_90 is not None:
            f.write(f"Threshold for 90% success rate: {threshold_90:.4f}\n")
        else:
            f.write("Threshold for 90% success rate: Not achieved\n")
        if mean_dist < (og_mean_dist*1.1):
            f.write("improved\n")
            success = True
        else:
            f.write("did not improve\n")
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

    grid =  torch.from_numpy(bw_grid((2912, 2912), spacing=64, thickness=3))
    tr1 = bilinear_interpolation(grid, torch.from_numpy( dfv[:, 0]), torch.from_numpy(dfv[:, 1]))
    img = bilinear_interpolation(moving_image, torch.from_numpy(dfv[:, 0]), torch.from_numpy(dfv[:, 1]))
    tr1 = tr1.reshape(vol_shape).numpy()

    img = cv2.resize(img.reshape(vol_shape).numpy(), (2912, 2912))

    fig.suptitle('Image Registration Results', fontsize=16)
    fig.text(0.5, 0.02, 
         'The grid transformation applied to the moving image produces the registered image.\nBlue points show transformed positions from white (fixed) to green (moving) landmarks.',
         ha='center', fontsize=7)
         
    plt.subplots_adjust(bottom=0.11) 
    
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
        dy, dx = simple_bilinear_interpolation_point(dfv, fixed_x, fixed_y, scale)

        # dx = ox+(ox-dx)
        # dy = oy+(oy-dy)

        x_res= (dx  + 1 ) * (2912 - 1) * 0.5
        y_res= (dy  + 1 ) * (2912 - 1) * 0.5

        #print("x: {} y: {} x_truth: {} y_truth: {} x_res: {} y_res:{} ".format(x, y, x_truth, y_truth, x_res, y_res))
        dist = np.linalg.norm(np.array((moving_x, moving_y)) - np.array((x_res, y_res)))
        og_dist = np.linalg.norm(np.array((moving_x, moving_y)) - np.array((fixed_x, fixed_y)))

        axes[0].scatter(fixed_x, fixed_y, c='w', s=2)  
        axes[1].scatter(fixed_x, fixed_y, c='w', s=1) 
        axes[1].scatter(x_res, y_res, c='b', s=1)  
        axes[1].scatter(moving_x, moving_y, c='g', s=2)  
        axes[1].annotate(f'{dist:.1f}', (x_res, y_res))
        axes[1].plot([moving_x, x_res], [moving_y, y_res], linestyle='-', color='red', linewidth=0.2)
        axes[1].plot([x_res, fixed_x], [y_res, fixed_y], linestyle='-', color='white', linewidth=0.2)
        axes[2].scatter(moving_x, moving_y, c='g', s=2)  

        dists.append(dist)
        og_dists.append(og_dist)

    with open(os.path.join(save_path,'dists.txt'), 'w') as f:
        for i, o in zip(dists, og_dists):
            f.write("og:%s | now:%s\n" % (o, i))
        f.write("ogMean: %s\n" % np.mean(og_dists))
        f.write("nowMean: %s\n" % np.mean(dists))
        
        
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
    mapx, mapy = np.meshgrid(np.arange(-1,1,2/vol_shape[0]), np.arange(-1,1,2/vol_shape[0]))
    dfs = np.stack([mapy, mapx], axis=2)
    # mapx = mapx - 0.2
    # mapy = mapy
    # dfm = np.stack([mapy, mapx], axis=2)
    # dfm=dfm.reshape((vol_shape[0]*vol_shape[1], 2))

    grid =  torch.from_numpy(bw_grid((vol_shape[0], vol_shape[1]), spacing=64, thickness=3))
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

        x_res= (dx  + 1 ) * (vol_shape[0] - 1) * 0.5 * (1/scale)
        y_res= (dy  + 1 ) * (vol_shape[0] - 1) * 0.5 * (1/scale)
        #print("x: {} y: {} x_truth: {} y_truth: {} x_res: {} y_res:{} ".format(x, y, x_truth, y_truth, x_res, y_res))
        dist = np.linalg.norm(np.array((moving_x, moving_y)) - np.array((x_res, y_res)))
        og_dist = np.linalg.norm(np.array((fixed_x, fixed_y)) - np.array((moving_x, moving_y)))
        
        axes[0].scatter(fixed_x, fixed_y, c='w', s=2)  
        axes[1].scatter(fixed_x, fixed_y, c='w', s=1) 
        axes[1].scatter(x_res, y_res, c='b', s=1)  
        axes[1].scatter(moving_x, moving_y, c='g', s=2)  
        axes[1].annotate(f'{dist:.1f}', (x_res, y_res))
        axes[1].plot([moving_x, x_res], [moving_y, y_res], linestyle='-', color='red', linewidth=0.2)
        axes[1].plot([x_res, fixed_x], [y_res, fixed_y], linestyle='-', color='white', linewidth=0.2)
        axes[2].scatter(moving_x, moving_y, c='g', s=2)  

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

def bw_grid(vol_shape, spacing, thickness=1):
    """Draw a black and white grid."""
    if not isinstance(spacing, (list, tuple)):
        spacing = [spacing] * len(vol_shape)
    spacing = [f + 1 for f in spacing]
    
    grid_image = np.zeros(vol_shape)
    for d, v in enumerate(vol_shape):
        ranges = [np.arange(0, f) for f in vol_shape] 
        for t in range(thickness):
            ranges[d] = np.append(np.arange(0 + t, v, spacing[d]), -1)
            grid_image[tuple(np.meshgrid(*ranges, indexing='ij'))] = 1
            
    return grid_image

def evaluate_initial_loss(model, sample_size=None):
    """
    Evaluate the initial registration loss for a given model instance,
    including regularization losses if enabled.
    
    Note: Gradients are required for regularization terms so we disable
    the torch.no_grad() context.
    """
    model.network.eval()

    # Remove the torch.no_grad context to allow gradient computation.
    # Alternatively, ensure that the input coordinates require gradients.
    sample_size = sample_size or model.batch_size
    total_coords = model.possible_coordinate_tensor.shape[0]
    indices = torch.randperm(total_coords, device=model.possible_coordinate_tensor.device)[:sample_size]
    
    # Make sure the coordinates require gradients for the regularizers.
    coords = model.possible_coordinate_tensor[indices, :].clone().detach().requires_grad_(True)

    # Compute the network output and transformed coordinates.
    output = model.network(coords)
    coord_temp = coords + output
    output_rel = coord_temp - coords  # Learned displacement

    # Sample the moving and fixed images using bilinear interpolation.
    transformed_img = bilinear_interpolation(
        model.moving_image,
        coord_temp[:, 0],
        coord_temp[:, 1]
    )
    fixed_samples = bilinear_interpolation(
        model.fixed_image,
        coords[:, 0],
        coords[:, 1]
    )

    # Compute the data matching loss.
    loss_val = model.criterion(transformed_img, fixed_samples)

    # Import the regularizers module here to avoid potential circular imports.
    from objectives import regularizers

    # Add Jacobian regularization loss if enabled.
    if getattr(model, 'jacobian_regularization', False):
        jacobian_loss = model.alpha_jacobian * regularizers.compute_jacobian_loss_2d(
            coords, output_rel, batch_size=sample_size
        )
        loss_val += jacobian_loss

    # Add hyper elastic regularization loss if enabled.
    if getattr(model, 'hyper_regularization', False):
        if hasattr(regularizers, "compute_hyper_elastic_loss_2d"):
            hyper_loss = model.alpha_hyper * regularizers.compute_hyper_elastic_loss_2d(
                coords, output_rel, batch_size=sample_size
            )
            loss_val += hyper_loss
        else:
            print("Warning: hyper_regularization enabled but compute_hyper_elastic_loss_2d not available.")

    # Add bending regularization loss if enabled.
    if getattr(model, 'bending_regularization', False):
        bending_loss = model.alpha_bending * regularizers.compute_bending_energy_2d(
            coords, output_rel, batch_size=sample_size
        )
        loss_val += bending_loss

    return loss_val.item()


def select_best_initialization(moving_image, fixed_image, config, num_trials=5, sample_size=None, plot=True):
    """
    Try multiple network initializations (taking the regularization losses into account)
    and select the one with the lowest initial total loss.

    Additionally, save a combined image of the initial deformations on a grid for each trial,
    alongside a reference grid with overlaid dashed lines to indicate image boundaries 
    and the center. This makes it easier to judge deformations that move the grid out of 
    the image region.
    
    Parameters:
      moving_image (torch.Tensor): The moving image.
      fixed_image (torch.Tensor): The fixed image.
      config (dict): Configuration dictionary. Assumes 'image_shape' and 'save_folder' are included.
      num_trials (int): Number of candidate initialization trials.
      sample_size (int, optional): Number of points for loss evaluation.
      plot (bool): Whether to generate and save visualization plots. Default True.
    
    Returns:
      best_model: The model instance with the lowest evaluated loss.
      best_loss (float): The corresponding loss value.
    """
    import matplotlib.pyplot as plt

    best_loss = float('inf')
    best_model = None

    # Compute the volume shape (e.g. (1708, 1708)) and reference grid (undeformed).
    vol_shape = tuple(config["image_shape"])  # (height, width)
    ref_grid = bw_grid(vol_shape, spacing=64, thickness=3)  # original undeformed grid

    # Lists to store deformation images and their corresponding loss and seed
    deformation_images = []
    trial_losses = []
    trial_seeds = []

    for trial in range(num_trials):
        config_copy = config.copy()
        # Use a fully random seed if random_init is True, else add a trial offset.
        if config.get("random_init", False):
            new_seed = random.randint(0, 2**32 - 1)
            config_copy["seed"] = new_seed
        else:
            base_seed = config.get("seed", 1)
            config_copy["seed"] = base_seed + trial

        # Remove "random_init" so that ImplicitRegistrator2d does not receive an unknown keyword.
        config_copy.pop("random_init", None)
        
        # Set the global seed for this trial.
        torch.manual_seed(config_copy["seed"])
        
        # Create a new model instance with the current configuration.
        model = models.ImplicitRegistrator2d(moving_image, fixed_image, **config_copy)
        
        # Evaluate the initial loss (including regularization) for this candidate.
        loss_value = evaluate_initial_loss(model, sample_size)
        print(f"Trial {trial + 1}/{num_trials} (seed: {config_copy['seed']}): Initial total loss = {loss_value:.6f}")
        trial_losses.append(loss_value)
        trial_seeds.append(config_copy["seed"])

        if plot:
            # --- Begin Deformation Visualization Code for each trial ---
            try:
                # Get the model's current (initial) registration output.
                registered_img, dfv = model(output_shape=vol_shape)
                # Create a grid image using the bw_grid function.
                grid = torch.from_numpy(bw_grid(vol_shape, spacing=64, thickness=3))
                # Ensure dfv is a numpy array.
                if isinstance(dfv, torch.Tensor):
                    dfv_np = dfv.detach().cpu().numpy()
                else:
                    dfv_np = dfv
                # Compute the deformed grid using bilinear interpolation.
                transformed_grid = bilinear_interpolation(
                    grid,
                    torch.from_numpy(dfv_np[:, 0]),
                    torch.from_numpy(dfv_np[:, 1])
                )
                # Reshape to the original volume shape.
                transformed_grid = transformed_grid.reshape(vol_shape)
                deformation_images.append(transformed_grid)
            except Exception as e:
                print("Error computing deformation for trial", trial + 1, ":", e)
                # Append a placeholder (e.g., reference grid) if error occurs.
                deformation_images.append(ref_grid)
            # --- End Deformation Visualization Code ---

        if loss_value < best_loss:
            best_loss = loss_value
            best_model = model

    if plot:
        # --- Combine all trial deformations into a single figure ---
        # Create one extra subplot for the reference grid.
        num_columns = num_trials + 1  
        fig, axes = plt.subplots(1, num_columns, figsize=(3 * num_columns, 3))
        
        # Plot the reference grid in the first subplot.
        axes[0].imshow(ref_grid, cmap='gray')
        axes[0].set_title("Reference Grid", fontsize=9)
        axes[0].axis("off")
        
        # Retrieve image dimensions for reference lines
        height, width = vol_shape

        # Identify the best trial index.
        best_trial_index = trial_losses.index(best_loss) if trial_losses else None

        # Plot each candidate's deformation.
        for idx in range(num_trials):
            ax = axes[idx + 1]
            ax.imshow(deformation_images[idx], cmap='gray')
            title = f"Trial {idx + 1}\nLoss: {trial_losses[idx]:.4f}\nSeed: {trial_seeds[idx]}"
            if best_trial_index is not None and idx == best_trial_index:
                title += "\n(Best)"
                # Highlight best candidate border
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0) 
            
            ax.axhline(y=0, color='lime', linewidth=1)
            ax.axhline(y=height - 1, color='lime', linewidth=1)
            ax.axvline(x=0, color='lime', linewidth=1)
            ax.axvline(x=width - 1, color='lime', linewidth=1)
            ax.axhline(y=height / 2, color='cyan', linewidth=0.8)
            ax.axvline(x=width / 2, color='cyan', linewidth=0.8)

        fig.suptitle("Initial Deformations from Lottery Initialization", fontsize=12, y=1.1)
        os.makedirs(config["save_folder"], exist_ok=True)
        combined_filename = os.path.join(config["save_folder"], "initial_deformations_combined.png")
        fig.savefig(combined_filename, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f"Combined initial deformations saved at: {combined_filename}")

    return best_model, best_loss
