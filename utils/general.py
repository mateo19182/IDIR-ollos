import imageio
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import SimpleITK as sitk
import pystrum
from matplotlib.colors import Normalize


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
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

def make_coordinate_tensor(dims=(28, 28, 28), gpu=True):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

def make_masked_coordinate_tensor(mask, dims=(28, 28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

#----------------------------------------------------------------------

def display_images(images, image_names, cmap='color'):
    n = len(images)
    cols = 4
    rows = np.ceil(n / cols).astype(int)
    print(images)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    for ax, img, name in zip(axs.flatten(), images, image_names):
        ax.imshow(img, cmap=cmap if cmap == 'gray' else None)
        ax.set_title(f'{name} - Shape: {img.shape}')
        ax.axis('off')  # Hide axes ticks

    if n % cols != 0:
        for ax in axs.flatten()[n:]:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

def make_masked_coordinate_tensor_2d(mask, dims):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(2)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
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
    xv, yv = torch.meshgrid(x, y)
    
    # Stack the coordinate grids
    coordinate_grid = torch.stack([xv, yv], dim=-1) # Shape: [dims[0], dims[1], 2]
    
    # Flatten the grid to have a list of coordinates
    coordinate_grid = coordinate_grid.view(-1, 2) # Shape: [dims[0]*dims[1], 2]
    
    # Move to GPU if requested
    if gpu:
        coordinate_grid = coordinate_grid.cuda()
    return coordinate_grid

def bilinear_interpolation(input_array, x_indices, y_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)

    x = x_indices - x0
    y = y_indices - y0

    output = (
        input_array[x0, y0] * (1 - x) * (1 - y)
        + input_array[x1, y0] * x * (1 - y)
        + input_array[x0, y1] * (1 - x) * y
        + input_array[x1, y1] * x * y
    )
    return output

def load_image_RFMID(folder):

    data = np.load(folder)
    og_img = data['og_img'] #imaxe orixinal
    original = data['og_img'] #imaxe orixinal
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
        original
    )

def plot_loss_curves(data_loss_list, total_loss_list, epochs):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, data_loss_list, label='Data Loss',  linestyle='-', color='blue')
    plt.plot(epochs_range, total_loss_list, label='Total Loss',  linestyle='--', color='red')
    plt.title('Loss Curves over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()


def load_image_FIRE(index, folder):
    images_folder = os.path.join(folder, 'Images')
    files = os.listdir(images_folder)
    files.sort()
    fixed_image = imageio.imread(os.path.join(images_folder, files[index*2]))
    moving_image = imageio.imread(os.path.join(images_folder, files[(index*2)+1]))
    ground_folder = os.path.join(folder, 'Ground Truth')
    files = os.listdir(ground_folder)
    files.sort()
    with open(os.path.join(ground_folder, files[index*2]), 'r') as f:
        ground_truth = f.read()
    images = [fixed_image, moving_image] 
    image_names = ['fixed_image', 'moving_image']
    #display_images(images, image_names)
    grayscale_images = np.dot(images, [0.2989, 0.5870, 0.1140])
    green_channel_images = [img[:, :, 1] for img in images]
    fixed_image = torch.tensor(green_channel_images[0], dtype=torch.float)
    moving_image = torch.tensor(green_channel_images[1], dtype=torch.float)
    return (
        fixed_image,
        moving_image,
        ground_truth,
        grayscale_images[0], 
        grayscale_images[1]
    )

def test_accuracy(transformation, ground_truth):
    scale = 500/2912
    lines = ground_truth.strip().split('\n')
    dists = []
    error_threshold = 1
    for line in lines:
        points = line.split()
        x= float(points[0])*scale
        y=float(points[1])*scale
        x_truth = float(points[2])*scale
        y_truth = float(points[3])*scale
        transformation=transformation.reshape((500, 500, 2))
        x_t, y_t = transformation[int(x), int(y)]
        x_res, y_res = (x_t*x) + x, (y_t*y) + y
        print("x: {} y: {} x_truth: {} y_truth: {} x_res: {} y_res:{}".format(x, y, x_truth, y_truth, x_res, y_res))
        dist = np.linalg.norm(np.array((x_truth, y_truth)) - np.array((x_res, y_res)))
        dists.append(dist)
    print(dists)

def block_average(arr, block_size):
    shape = (arr.shape[0] // block_size, arr.shape[1] // block_size)
    return arr.reshape(shape[0], block_size, shape[1], block_size).mean(axis=(1,3))

def display_dfv(image, dfv,fixed_image, moving_image):
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]

    u = dfv[:, 0].reshape(image.shape)
    v = dfv[:, 1].reshape(image.shape)


    
    M = np.hypot(u, v)
    angles = np.arctan2(v, u)

    skip_factor = 5
    skip = (slice(None, None, skip_factor), slice(None, None, skip_factor))

    u_avg = block_average(u, skip_factor)
    v_avg = block_average(v, skip_factor)
    M_avg = np.hypot(u_avg, v_avg)
    angles_avg = np.arctan2(v_avg, u_avg)

    x_avg = block_average(x, skip_factor)
    y_avg = block_average(y, skip_factor)

    fig, axs = plt.subplots(1, 4, figsize=(15, 5)) 

    axs[0].imshow(np.flip(fixed_image, axis=0), cmap='gray')
    axs[0].set_title('Fixed Image')
    axs[0].axis('off') 

    axs[1].imshow(np.flip(moving_image, axis=0), cmap='gray')
    axs[1].set_title('Moving Image')
    axs[1].axis('off') 
    #color= 'red'

    grid_small = torch.from_numpy(pystrum.pynd.ndutils.bw_grid(vol_shape=(2912, 2912), spacing=16))
    dfv = torch.from_numpy(dfv)
    tr1 = bilinear_interpolation(grid_small, dfv[:, 0], dfv[:, 1])
    tr1 = tr1.reshape([500, 500])

    axs[2].imshow(tr1.numpy(), cmap='gray', origin='lower')
    axs[2].set_title('grid deformation')
    axs[2].axis('off') 

    axs[3].imshow(image, cmap='gray', origin='lower')
    quiver = axs[3].quiver(x_avg, y_avg, u_avg, v_avg, angles_avg, cmap='hsv')
    #quiver = axs[3].quiver(x[skip], y[skip], u[skip], -v[skip],angles[skip], cmap='hsv')
    #quiver = axs[3].quiver(x[skip], y[skip], u[skip], v[skip],M[skip], cmap='jet')
    axs[3].set_title('Vector Field Visualization')
    axs[3].axis('off') 


    cbar = fig.colorbar(quiver, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Direction')
    cbar.set_ticks([-np.pi, 0, np.pi])
    #plt.text(0.1, 0.5, """0: Red (rightward),π/2: Cyan or Green (upward), π: Blue (leftward), -π/2: Magenta or Yellow (downward) """, fontsize=12)

    plt.tight_layout()
    plt.show()

    # 0 radians (0 degrees): Red (rightward)
    # π/2 radians (90 degrees): Cyan or Green (upward)
    # π radians (180 degrees): Blue (leftward)
    # 3π/2 radians (270 degrees): Magenta or Yellow (downward)


def display_grid(dfv, shape=[500, 500]):
    grid_small = torch.from_numpy(pystrum.pynd.ndutils.bw_grid(vol_shape=(2912, 2912), spacing=8))
    grid_medium =  torch.from_numpy(pystrum.pynd.ndutils.bw_grid(vol_shape=(2912, 2912), spacing=16))
    grid_big =  torch.from_numpy(pystrum.pynd.ndutils.bw_grid(vol_shape=(2912, 2912), spacing=32))

    dfv = torch.from_numpy(dfv)

    tr1 = bilinear_interpolation(grid_small, dfv[:, 0], dfv[:, 1])
    tr1 = tr1.reshape(shape)

    tr2 = bilinear_interpolation(grid_medium, dfv[:, 0], dfv[:, 1])
    tr2 = tr2.reshape(shape)

    tr3 = bilinear_interpolation(grid_big, dfv[:, 0], dfv[:, 1])
    tr3 = tr3.reshape(shape)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(tr1.numpy(), cmap='gray')
    axs[0].set_title('Transformed Grid Small')

    axs[1].imshow(tr2.numpy(), cmap='gray')
    axs[1].set_title('Transformed Grid Medium')

    axs[2].imshow(tr3.numpy(), cmap='gray')
    axs[2].set_title('Transformed Grid Big')

    plt.show()
