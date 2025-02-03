from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
import numpy as np

from utils import general
from networks import networks
from objectives import ncc
from objectives import ssim
from objectives import regularizers
import math, random
from visualization import fig_vis


class ImplicitRegistrator2d:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, coordinate_tensor=None, output_shape=(1000, 1000), dimension=0, slice_pos=0
    ):
        """Return the image-values for the given input-coordinates."""
        # Use standard coordinate tensor if none is given
        if coordinate_tensor is None:
            coordinate_tensor = general.make_coordinate_tensor_2d(
                output_shape
            )
        

        output = self.network(coordinate_tensor)

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(output, coordinate_tensor)

        transformed_image = self.transform_no_add(coord_temp)
        return (
            transformed_image.cpu()
            .detach()
            .numpy()
            .reshape(output_shape[0], output_shape[1])
            , coord_temp.cpu().detach().numpy()
        )

    def __init__(self, moving_image, fixed_image, **kwargs):
        """Initialize the learning model."""

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in kwargs)

        self.patience = kwargs.get('patience', 100)  # Number of epochs to wait for improvement
        self.min_delta = kwargs.get('min_delta', 1e-4)  # Minimum change in loss to qualify as an improvement
        self.best_loss = float('inf')
        self.counter = 0

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.save_checkpoints = (
            kwargs["save_checkpoints"]
            if "save_checkpoints" in kwargs
            else self.args["save_checkpoints"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )

        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]

        # Use the provided seed from kwargs, if available; otherwise, use the default.
        seed = kwargs.get("seed", self.args.get("seed", 1))
        torch.manual_seed(seed)

        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )
        if self.network_from_file is None:
            if self.network_type == "MLP":
                self.network = networks.MLP(self.layers)
            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            if self.verbose:
                print(
                    "Network contains {} trainable parameters.".format(
                        general.count_parameters(self.network)
                    )
                )
        else:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()

        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()
        elif self.loss_function_arg.lower() == "ssim":
            self.criterion = ssim.SSIM_()
            

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Parse arguments from kwargs
        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]

        # Parse regularization kwargs
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )
        self.og_batch_size = self.batch_size
        self.phases = kwargs["phases"] if "phases" in kwargs else self.args["phases"]
        self.prev_phase = self.phases +1
        self.phase_length = self.epochs / self.phases

        self.sampling = (
            kwargs["sampling"] if "sampling" in kwargs else self.args["sampling"]
        )
        self.weight_mask = None
        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image


        # if self.sampling == "uniform":
        #     self.possible_coordinate_tensor = general.make_uniform_coordinate_tensor(self.mask, self.fixed_image.shape, self.batch_size)
        # else:
        self.possible_coordinate_tensor = general.make_masked_coordinate_tensor_2d(
                self.mask, self.fixed_image.shape
        )
        #print(self.possible_coordinate_tensor.shape) #torch.Size([2912, 2912])
        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()

    def cuda(self):
        """Move the model to the GPU."""

        # Standard variables
        self.network.cuda()

        # Variables specific to this class
        self.moving_image.cuda()
        self.fixed_image.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        self.args["patience"] = 50
        self.args["min_delta"] = 1e-5
        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None

        self.args["method"] = 1

        self.args["lr"] = 0.00001   
        self.args["batch_size"] = 10000 #tensor size   
        self.args["layers"] = [2, 256, 256, 256, 2]
        self.args["velocity_steps"] = 1  
        self.args["sampling"] = "uniform"  #uniform, random
        self.args["phases"] = 1
        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1
        self.args["save_checkpoints"] = False
        self.args["jacobian_regularization"] = False
        self.args["alpha_jacobian"] = 0.05

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (200, 200)

        self.args["network"] = None

        self.args["epochs"] = 2500
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"

        self.args["network_type"] = "MLP"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1

    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0
        
        #batch size schedule
        current_phase = self.phases - max(int(epoch / self.phase_length), 0)
        if current_phase != self.prev_phase:
            self.batch_size = int(self.og_batch_size ** (1/(2**(current_phase-1))))
            self.prev_phase = current_phase
            print(f"Current epoch: {epoch} phase: {current_phase}, batch size: {self.batch_size}")


        #sampling strategy
        if self.sampling == "weighted":
            if self.weight_mask is None:
                self.weight_mask = general.weight_mask(self.mask, self.fixed_image, save=False)
                if self.weight_mask.sum() != 1:
                    print("Weight mask not normalized")
            indices = torch.multinomial(self.weight_mask, self.batch_size, replacement=True)
        elif self.sampling == "random":
            indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], device="cuda"
            )[: self.batch_size]
        elif self.sampling == "percentage":
            weighted_percentage = 0.5
            weighted_batch_size = int(self.batch_size * weighted_percentage)
            random_batch_size = self.batch_size - weighted_batch_size
            
            if self.weight_mask is None:
                self.weight_mask = general.weight_mask(self.mask, self.fixed_image, save=False)
            weighted_indices = torch.multinomial(self.weight_mask, weighted_batch_size, replacement=True)
            
            random_indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], device="cuda"
            )[: random_batch_size]
            
            indices = torch.cat((weighted_indices, random_indices))
        elif self.sampling == "uniform":  # not optimized
            self.possible_coordinate_tensor = general.make_uniform_coordinate_tensor(self.mask, self.fixed_image.shape, self.batch_size)
            indices = torch.arange(self.possible_coordinate_tensor.shape[0], device='cuda')
        if epoch == 0 and True:
            fig_vis.visualize_sampling(indices, self.possible_coordinate_tensor)

        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)
        output = self.network(coordinate_tensor)
        coord_temp = torch.add(output, coordinate_tensor)
        output = coord_temp

        transformed_image = self.transform_no_add(coord_temp)
        fixed_image = general.bilinear_interpolation(
            self.fixed_image,
            coordinate_tensor[:, 0],
            coordinate_tensor[:, 1],
        )

        # Compute the loss
        loss += self.criterion(transformed_image, fixed_image)

        # Store the value of the data loss
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        # Relativation of output
        output_rel = torch.subtract(output, coordinate_tensor)

        if self.jacobian_regularization:
            jacobian_loss = self.alpha_jacobian * regularizers.compute_jacobian_loss_2d(
            coordinate_tensor, output_rel, batch_size=self.batch_size
            )
            loss +=  jacobian_loss
            if epoch % 50 == 0:
                print(f"Jacobian Regularization loss={jacobian_loss.item()}")

        if self.hyper_regularization:
            hyper_loss = self.alpha_hyper * regularizers.compute_hyper_elastic_loss_2d(
            coordinate_tensor, output_rel, batch_size=self.batch_size
            )
            loss +=  hyper_loss
            if epoch % 50 == 0:
                print(f"Hyper Regularization loss={hyper_loss.item()}")

        if self.bending_regularization:
            bending_loss = self.alpha_bending * regularizers.compute_bending_energy_2d(
            coordinate_tensor, output_rel, batch_size=self.batch_size
            )
            loss +=  bending_loss
            if epoch % 50 == 0:
                print(f"Bending Regularization loss={bending_loss.item()}")


        # Perform the backpropagation and update the parameters accordingly

        for param in self.network.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def transform(
        self, transformation, coordinate_tensor=None, moving_image=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordinate_tensor is None:
            coordinate_tensor = self.coordinate_tensor

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordinate_tensor)
        return general.bilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
        )

    def transform_no_add(self, transformation, moving_image=None, reshape=False):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image
        # print('GET MOVING')
        return general.bilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
        )

    def fit(self, epochs=None, red_blue=False):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]

        # Perform training iterations
        for i in tqdm.tqdm(range(epochs), desc="Training", unit="epoch"):
            if (i-1) % 25 == 0:
                tqdm.tqdm.write(f"Epoch {i+1}/{epochs}, Loss: {self.loss_list[i-1]:.6f}")
            if self.save_checkpoints:
                if i%500 == 0:
                    path = os.path.join(self.save_folder, 'epoch-{}.pth'.format(i))
                    torch.save(self.network.state_dict(), path)
            self.training_iteration(i)
            current_loss = self.loss_list[i]
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {i}")
                    break

        with open(os.path.join(self.save_folder,'loss_list.txt'), 'w') as f:
            if(self.counter >= self.patience):
                f.write(f"Early stopping triggered at epoch {i}\n")
            for item in self.loss_list:
                f.write("%s\n" % item)
            

        general.plot_loss_curves(self.loss_list, self.data_loss_list, self.epochs, self.save_folder)
