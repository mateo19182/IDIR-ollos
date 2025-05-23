import torch


def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad

#----------------------------------------------

def compute_hyper_elastic_loss_2d(
    input_coords, output, batch_size=None, alpha_l=1, alpha_a=1
):
    """Compute the hyper-elastic regularization loss for 2D inputs."""
    
    grad_u = compute_jacobian_matrix_2d(input_coords, output, add_identity=False)
    
    grad_y = compute_jacobian_matrix_2d(input_coords, output, add_identity=True)

    length_loss = torch.linalg.norm(grad_u, dim=(1, 2))
    length_loss = torch.pow(length_loss, 2)
    length_loss = torch.sum(length_loss)
    length_loss = 0.5 * alpha_l * length_loss

    area_loss = torch.det(grad_y) - 1
    area_loss = torch.maximum(area_loss, torch.zeros_like(area_loss)) 
    area_loss = torch.pow(area_loss, 2)
    area_loss = torch.sum(area_loss) 
    area_loss = alpha_a * area_loss

    loss = length_loss + area_loss

    if batch_size is not None:
        loss /= batch_size

    return loss

def compute_jacobian_loss_2d(input_coords, output, batch_size=None):
    """Compute the jacobian regularization loss."""

    jac = compute_jacobian_matrix_2d(input_coords, output)

    loss = torch.det(jac) - 1
    loss = torch.linalg.norm(loss, 1)

    return loss / batch_size

def compute_jacobian_matrix_2d(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix of the output with respect to the input for 2D images."""
    
    jacobian_matrix = torch.zeros(input_coords.shape[0], 2, 2)
    
    for i in range(2): 
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    
    return jacobian_matrix

def compute_bending_energy_2d(input_coords, output, batch_size=None):
    """Compute the bending energy for 2D images."""
    
    jacobian_matrix = compute_jacobian_matrix_2d(input_coords, output, add_identity=False)

    dx_xy = torch.zeros(input_coords.shape[0], 2, 2) 
    dy_xy = torch.zeros(input_coords.shape[0], 2, 2) 

    for i in range(2): 
        dx_xy[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 0])
        dy_xy[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 1])

    dx_xy = torch.square(dx_xy)
    dy_xy = torch.square(dy_xy)

    loss = (
        torch.mean(dx_xy[:, :, 0])
        + torch.mean(dy_xy[:, :, 1])
        + 2 * torch.mean(dx_xy[:, :, 1])
    )

    if batch_size is not None:
        loss /= batch_size

    return loss
