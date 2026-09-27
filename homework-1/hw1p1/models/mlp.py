import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:
    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use ReLU activations for the layer.
        """
        self.debug = debug

        # MLP0 has one linear layer followed by ReLU
        self.layers = [Linear(2, 3), ReLU()]

    def forward(self, A0):
        """
        Create your own mytorch.models.MLP0!
        Pass the input through the linear layer followed by the activation layer to get the model output.
        Read the writeup (Hint: MLP0 Section) for further details on MLP0 forward and backward implementation.
        """
        # First pass the input through the linear layer
        Z0 = self.layers[0].forward(A0)

        # Apply ReLU to the output of the linear layer
        A1 = self.layers[1].forward(Z0)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):
        """
        Create your own mytorch.models.MLP0!
        Read the writeup (Hint: MLP0 Section) for further details on MLP0 forward and backward implementation.
        Refer to the pseudocode in the writeup to implement backpropagation through the model.
        """
        # Backpropagate through ReLU first
        dLdZ0 = self.layers[1].backward(dLdA1)

        # Then backpropagate through the linear layer
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:
    def __init__(self, debug=False):
        """
        Initialize 2 linear layers.
        Layer 1 of shape (2,3).
        Layer 2 of shape (3, 2).
        Use ReLU activations for both the layers.
        Implement it on the same lines (in a list) as in the MLP0 class.
        """
        self.debug = debug

        # The layers are arranged as Linear -> ReLU -> Linear -> ReLU
        self.layers = [Linear(2, 3), ReLU(), Linear(3, 2), ReLU()]

    def forward(self, A0):
        """
        Create your own mytorch.models.MLP1!
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        Read the writeup (Hint: MLP1 Section) for further details on MLP1 forward and backward implementation.
        """
        # Pass through the first linear layer and ReLU
        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)

        # Pass the result through the second linear layer and ReLU
        Z1 = self.layers[2].forward(A1)
        A2 = self.layers[3].forward(Z1)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        """
        Create your own mytorch.models.MLP1!
        Read the writeup (Hint: MLP1 Section) for further details on MLP1 forward and backward implementation.
        Refer to the pseudocode in the writeup to implement backpropagation through the model.
        """
        # Backpropagate through the second ReLU and linear layer
        dLdZ1 = self.layers[3].backward(dLdA2)
        dLdA1 = self.layers[2].backward(dLdZ1)

        # Continue backpropagating through the first ReLU and linear layer
        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2).
        Note: Use ReLU activations for ALL the linear layers.
        Implement it on the same lines (in a list) as in the MLP1 class.

        Hint: Refer the diagrammatic view in the writeup for better understanding!
        """
        self.debug = debug

        # The layers follow the pattern Linear -> ReLU for every layer
        self.layers = [
            Linear(2, 4), ReLU(),
            Linear(4, 8), ReLU(),
            Linear(8, 8), ReLU(),
            Linear(8, 4), ReLU(),
            Linear(4, 2), ReLU()
        ]

    def forward(self, A):
        """
        Create your own mytorch.models.MLP4!
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        Read the writeup (Hint: MLP4 Section) for further details on MLP4 forward and backward implementation.
        """
        if self.debug:
            self.A = [A]

        L = len(self.layers)

        # Apply each layer in order during the forward pass
        for i in range(L):
            A = self.layers[i].forward(A)

            if self.debug:
                self.A.append(A)

        return A

    def backward(self, dLdA):
        """
        Create your own mytorch.models.MLP4!
        Read the writeup (Hint: MLP4 Section) for further details on MLP4 forward and backward implementation.
        Refer to the pseudocode in the writeup to implement backpropagation through the model.
        """
        if self.debug:
            self.dLdA = [dLdA]

        L = len(self.layers)

        # Go through the layers in reverse order during backpropagation
        for i in reversed(range(L)):
            dLdA = self.layers[i].backward(dLdA)

            if self.debug:
                self.dLdA = [dLdA] + self.dLdA

        return dLdA

"""
Name : Bhuvana Teja Kanakam
Course : 11-785 Introduction to Deep Learning
Homework : 1 - Part 1

My Understanding : MLP is basically a combination of linear layers 
and activation functions. The output of one layer becomes the input 
to the next layer. In the forward pass, I go through the layers in 
the given order and calculate the output at each step. For MLP0, 
there is one linear layer followed by ReLU. For MLP1, there are two 
linear layers with ReLU after each one. MLP4 follows the same idea 
but has more layers, so I use a loop instead of writing each layer 
separately. During the backward pass, I start from the gradient of 
the final output and move through the layers in reverse order. Each 
layer calculates the gradient that needs to be passed to the previous 
layer. For MLP4, I store the activations during the forward pass and 
the gradients during the backward pass when debug is enabled. This 
helps in checking the intermediate values and gradients during debugging.
"""