import torch.nn.functional as F
import torch
from torch import nn
import numpy as np


class Siren(nn.Module):
    """This is a dense neural network with sine activation functions.

    Arguments:
    layers -- ([*int]) amount of nodes in each layer of the network, e.g. [2, 16, 16, 1]
    gpu -- (boolean) use GPU when True, CPU when False
    weight_init -- (boolean) use special weight initialization if True
    omega -- (float) parameter used in the forward function
    """

    def __init__(self, layers, weight_init=True, omega=30):
        """Initialize the network."""

        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega
        # self.args["omega"] = 32
        # self.args["layers"] = [3, 256, 256, 256, 3]

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # We  ight Initialization
            if weight_init:
                with torch.no_grad():
                    if i == 0:
                        # init.kaiming_uniform_(self.layers[-1].weight, a=0, mode='fan_in', nonlinearity='linear')                        
                        self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i]) 
                        # Current Initialization Range: [−0.5,0.5] // SIREN Recommended Initialization Range: [−15.0,15.0] ?
                        # print(self.layers[-1].weight)                             
                    else:                
                        # init.kaiming_uniform_(self.layers[-1].weight, a=0, mode='fan_in', nonlinearity='linear')                        
                        self.layers[-1].weight.uniform_(    #this one ok
                            -np.sqrt(6 / layers[i]) / self.omega,
                            np.sqrt(6 / layers[i]) / self.omega,
                        )

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)


class MLP(nn.Module):
    def __init__(self, layers):
        """Initialize the network."""

        super(MLP, self).__init__()
        self.n_layers = len(layers) - 1

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            # He/Kaiming initialization 
        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward function of the network."""

        # Perform relu on all layers except for the last one
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))

        # Propagate through final layer and return the output
        return self.layers[-1](x)
