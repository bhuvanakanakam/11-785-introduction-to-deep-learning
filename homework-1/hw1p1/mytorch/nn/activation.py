import numpy as np
import scipy

class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        # Identity does not change the input, so the output is the same as Z.
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        # Since A = Z, the derivative of A with respect to Z is 1 everywhere.
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    """
    Sigmoid activation function.
    """

    def forward(self, Z):
        # Sigmoid maps each value to a value between 0 and 1.
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        # The derivative of sigmoid can be written using its output as A * (1 - A).
        dAdZ = self.A * (1 - self.A)

        # Apply the chain rule to get the gradient with respect to Z.
        dLdZ = dLdA * dAdZ
        return dLdZ


class Tanh:
    """
    Tanh activation function.
    """

    def forward(self, Z):
        # Implement tanh using its exponential form instead of np.tanh().
        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        # The derivative of tanh is 1 - tanh(Z)^2, which is 1 - A^2.
        dAdZ = 1 - self.A ** 2

        # Apply the chain rule.
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.
    """

    def forward(self, Z):
        # ReLU keeps positive values and changes negative values to zero.
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        # The derivative is 1 for positive values and 0 for non-positive values.
        dAdZ = np.where(self.A > 0, 1, 0)

        # Apply the chain rule to get the gradient with respect to Z.
        dLdZ = dLdA * dAdZ
        return dLdZ


class GELU:
    """
    GELU (Gaussian Error Linear Unit) activation function.
    """

    def forward(self, Z):
        # Save Z because it is needed again when calculating the derivative.
        self.Z = Z

        # GELU can be written using the error function.
        self.A = 0.5 * Z * (
            1 + scipy.special.erf(Z / np.sqrt(2))
        )
        return self.A

    def backward(self, dLdA):
        # This is the derivative of GELU with respect to its input Z.
        dAdZ = (
            0.5 * (1 + scipy.special.erf(self.Z / np.sqrt(2)))
            + self.Z * np.exp(-(self.Z ** 2) / 2) / np.sqrt(2 * np.pi)
        )

        # Use the chain rule to calculate dL/dZ.
        dLdZ = dLdA * dAdZ
        return dLdZ


class Swish:
    """
    Swish activation function.
    """

    def __init__(self, beta=1.0):
        # Beta is a learnable parameter for Swish.
        self.beta = beta

    def forward(self, Z):
        # Save Z and the sigmoid value because both are used during backpropagation.
        self.Z = Z
        self.sigmoid = 1 / (1 + np.exp(-self.beta * Z))

        # Swish is Z multiplied by sigmoid(beta * Z).
        self.A = Z * self.sigmoid
        return self.A

    def backward(self, dLdA):
        # Derivative of Swish with respect to Z.
        dAdZ = (
            self.sigmoid
            + self.beta * self.Z * self.sigmoid * (1 - self.sigmoid)
        )

        # Gradient of the loss with respect to Z.
        dLdZ = dLdA * dAdZ

        # Since beta is learnable, we also need the gradient of the loss with respect to beta.
        dAdbeta = self.Z ** 2 * self.sigmoid * (1 - self.sigmoid)
        self.dLdbeta = np.sum(dLdA * dAdbeta)

        # The backward function returns the gradient with respect to the input Z.
        return dLdZ


class Softmax:
    """
    Softmax activation function.
    """

    def forward(self, Z):
        # Subtract the maximum value in each row before exponentiating
        # to avoid very large exponential values.
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)

        exp_Z = np.exp(Z_shifted)

        # Normalize each row so that all output values add up to 1.
        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        # N is the number of samples and C is the number of features.
        N = dLdA.shape[0]
        C = dLdA.shape[1]

        # Store the gradients for all samples here.
        dLdZ = np.zeros_like(dLdA)

        # Softmax is different from the other activations because
        # each output depends on all the inputs in the row.
        for i in range(N):
            # For each sample, the softmax derivative is represented
            # by a C x C Jacobian matrix.
            J = np.zeros((C, C))

            for m in range(C):
                for n in range(C):

                    # Diagonal entries of the Jacobian.
                    if m == n:
                        J[m, n] = self.A[i, m] * (1 - self.A[i, n])

                    # Off-diagonal entries of the Jacobian.
                    else:
                        J[m, n] = -self.A[i, m] * self.A[i, n]

            # Multiply dLdA by the Jacobian to get dLdZ for this sample.
            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ

"""
Name : Bhuvana Teja Kanakam
Course : 11-785 Introduction to Deep Learning
Homework : 1 - Part 1

My Understanding : activation functions are used to introduce the non lineratity
in between the linear layers. otherwise, multiple linear layers essentially act 
like one when stacked. during the activation, forward pass : converts the z to a, 
and during the backward pass derivative of the loss wrt to w, x and b is used.

For Sigmoid, Tanh, ReLU, and GELU, the activation is applied element-wise, then 
derivatives -> element wise. output or input can be stored in the fp, to use for 
derivate calculations in the bp.

Swish is slightly different because it has a learnable parameter beta. hence, during
bp, gradient wrt z and beta needs to be calculated. and softmaz, needs a jacobian 
because the value of each output is dependant on all the inputs in that row, so not
element wise.
"""