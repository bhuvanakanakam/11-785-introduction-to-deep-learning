import numpy as np

class Linear:
    def __init__(self, in_features, out_features, debug=False):
        # Initialize weights and bias with zeros
        self.in_features = in_features
        self.out_features = out_features

        self.debug = debug
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

    def forward(self, A):
        # A is the input to the linear layer with shape (N, C0)
        # The output Z will have shape (N, C1)

        self.A = A
        self.N = A.shape[0]  # store the number of samples in the batch

        # Create a column of ones so that the bias can be added to every sample
        self.ones = np.ones((self.N, 1))

        Z = A @ self.W.T + self.ones @ self.b.T

        if self.debug:
            self.Z = Z
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)
        """

        # Calculate the gradients of the loss with respect to W, b and A
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = np.sum(dLdZ, axis=0, keepdims=True).T
        self.dLdA = dLdZ @ self.W

        if self.debug:
            self.dLdZ = dLdZ

        return self.dLdA


"""
Name : Bhuvana Teja Kanakam
Course : 11-785 Introduction to Deep Learning
Homework : 1 - Part 1

My Understanding:
in the fp, the input A gets multiplied by the weight matrix W with the bias,
essentially making this into a vector form -> so a matrix multiplication!

in the bp, the gradient coming from the next layer is used to calculate the
gradients with respect to the weights, bias, and input. the gradient with 
respect to A is returned so that the previous layer can continue the backpropagation.
"""