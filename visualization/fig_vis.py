import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_vector_field(dfv, 
                           background=None, 
                           stride=20, 
                           scale=1.0, 
                           figsize=(8, 8), 
                           colormap='jet', 
                           show_colorbar=True, 
                           title='Displacement Field'):
    """
    stride : int
        Sampling stride for displaying vectors (larger = sparser arrows).
    scale : float
        Quiver scale factor: smaller = longer arrows, bigger = shorter arrows.
        (matplotlib quiver usage can be a bit unintuitive: a bigger 'scale' shrinks the arrows).
    """
    H, W, _ = dfv.shape
    
    # Create figure & axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show background if provided
    if background is not None:
        ax.imshow(background, cmap='gray' if (background.ndim == 2) else None)
    
    # Generate a coordinate grid
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    
    # Downsample the field for sparser arrows
    x_down = x_coords[::stride, ::stride]
    y_down = y_coords[::stride, ::stride]
    
    # Displacement components
    u = dfv[::stride, ::stride, 1]  # dx
    v = dfv[::stride, ::stride, 0]  # dy
    
    # Compute magnitude for color-coding
    magnitudes = np.sqrt(u**2 + v**2)
    
    # Plot quiver
    quiv = ax.quiver(x_down, y_down, 
                     u, v, 
                     magnitudes,    # color by magnitude
                     angles='xy', 
                     scale_units='xy', 
                     scale=1.0/scale,   # bigger 'scale' => shorter arrows
                     cmap=colormap, 
                     alpha=0.8)
    
    if show_colorbar:
        cbar = plt.colorbar(quiv, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Displacement Magnitude', rotation=90)
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # If you want (0,0) at top-left like image coords
    plt.tight_layout()
    return fig, ax

def create_checkerboard(fixed, moving, num_squares=8, fixed_opacity=0.8):
    """Generate a checkerboard blending of two images with fixed image having less opacity."""
    h, w = fixed.shape
    step_x = w // num_squares
    step_y = h // num_squares
    
    checkerboard = np.zeros_like(fixed, dtype=np.float32)
    for i in range(num_squares):
        for j in range(num_squares):
            x_start = i * step_x
            x_end   = (i+1) * step_x
            y_start = j * step_y
            y_end   = (j+1) * step_y
            if (i + j) % 2 == 0:
                checkerboard[y_start:y_end, x_start:x_end] = fixed[y_start:y_end, x_start:x_end] * fixed_opacity
            else:
                checkerboard[y_start:y_end, x_start:x_end] = moving[y_start:y_end, x_start:x_end]
    return checkerboard

def color_overlay(fixed, registered):
    # Ensure both images are float np arrays in [0,1] range, or scale them accordingly
    fixed_norm = (fixed - fixed.min()) / (fixed.max() - fixed.min())
    reg_norm   = (registered - registered.min()) / (registered.max() - registered.min())

    # Create a 3-channel color image
    color_img = np.zeros((fixed.shape[0], fixed.shape[1], 3), dtype=np.float32)
    # Put fixed in red channel
    color_img[..., 0] = fixed_norm
    # Put registered in green channel
    color_img[..., 1] = reg_norm
    # Blue channel can remain 0 or hold some other information if desired

    return color_img

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

def save_weight_map_as_image(weight_map, filename="weight_map.png"):
    # Move to CPU and convert to numpy
    weight_map_np = weight_map.reshape(1708, 1708).astype(np.float32)
    
    # Since it's a probability map, its max may be small. Rescale to [0, 1]
    vis_map = (weight_map_np - weight_map_np.min()) / (weight_map_np.max() - weight_map_np.min() + 1e-8)
    
    # # Enhance contrast by applying a gamma correction
    # gamma = 1  # You can adjust this value to make the weights more pronounced
    # vis_map = np.power(vis_map, gamma)
    
    # Save using matplotlib
    plt.imsave(filename, vis_map, cmap='gray')
    print(f"Saved weight map image to {filename}")
