import numpy as np
import matplotlib.pyplot as plt

# ------------------- Parameters -------------------
# Grid parameters
num_points = 30  # Number of points along each axis
x_min, x_max = 0, 1
y_min, y_max = 0, 1

# Deformation parameters
deformation_center = np.array([0.5, 0.5])  # Center of deformation
deformation_radius = 0.3  # Radius of the deformation area
deformation_strength = 0.12  # Magnitude of deformation

# ------------------- Define the Grid -------------------
x, y = np.meshgrid(
    np.linspace(x_min, x_max, num_points),
    np.linspace(y_min, y_max, num_points)
)

# Flatten the grid for easier computation
points = np.vstack([x.ravel(), y.ravel()]).T

# ------------------- Define the Localized Diagonal Deformation Vector Field (DFV) -------------------
def localized_diagonal_deformation(p, center, radius, strength):
    """
    Apply a simple localized diagonal deformation: displacement equally in x and y directions
    within a circular region.
    
    Parameters:
        p (ndarray): Array of points with shape (N, 2).
        center (ndarray): Center of the deformation (x, y).
        radius (float): Radius of the deformation area.
        strength (float): Magnitude of the deformation.
        
    Returns:
        u (ndarray): Deformation in the x-direction.
        v (ndarray): Deformation in the y-direction.
    """
    # Compute distance from each point to the center
    distances = np.linalg.norm(p - center, axis=1)
    
    # Initialize deformation vectors
    u = np.zeros_like(distances)
    v = np.zeros_like(distances)
    
    # Apply deformation only to points within the radius
    mask = distances <= radius
    # Diagonal deformation: equal displacement in x and y
    u[mask] = strength*0.7 * (1 - distances[mask] / radius)
    v[mask] = strength * (1 - distances[mask] / radius)
    
    return u, v

# Get deformation vectors
u, v = localized_diagonal_deformation(points, deformation_center, deformation_radius, deformation_strength)

# Reshape deformation vectors to grid shape
u_grid = u.reshape(x.shape)
v_grid = v.reshape(y.shape)

# ------------------- Apply the Deformation to the Grid -------------------
x_def = x + u_grid
y_def = y + v_grid

# ------------------- Plot 1: DFV Arrows on the Entire Grid -------------------
plt.figure(figsize=(6, 6))
plt.quiver(x, y, u_grid, v_grid, color='red', angles='xy', scale_units='xy', scale=1)
plt.gca().set_aspect('equal')  # Maintain aspect ratio
plt.axis('off')  # Remove axes for LaTeX compatibility
plt.tight_layout()
plt.savefig('dfv_arrows_localized.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

# ------------------- Plot 2: Deformed Grid Only -------------------
# Identify grid lines that pass through the deformation area
# We'll consider a grid line affected if any of its points are within the deformation radius

# Function to determine if a grid line is affected
def is_line_affected(line_coords, center, radius):
    distances = np.linalg.norm(line_coords - center, axis=1)
    return np.any(distances <= radius)

# Determine affected rows and columns
affected_rows = []
affected_cols = []

for i in range(num_points):
    # Extract row points
    row = points[i * num_points:(i + 1) * num_points]
    if is_line_affected(row, deformation_center, deformation_radius):
        affected_rows.append(i)
        
    # Extract column points
    col = points[i::num_points]
    if is_line_affected(col, deformation_center, deformation_radius):
        affected_cols.append(i)

# Plotting the deformed grid lines only
plt.figure(figsize=(6, 6))

# Plot Deformed Rows in Green
for i in affected_rows:
    plt.plot(x_def[i, :], y_def[i, :], color='green', linewidth=1, linestyle='-')

# Plot Deformed Columns in Green
for j in affected_cols:
    plt.plot(x_def[:, j], y_def[:, j], color='green', linewidth=1, linestyle='-')

plt.gca().set_aspect('equal')  # Maintain aspect ratio
plt.axis('off')  # Remove axes for LaTeX compatibility
plt.tight_layout()
plt.savefig('dfv_grid_deformed_only.svg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
