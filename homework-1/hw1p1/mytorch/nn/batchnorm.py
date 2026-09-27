import numpy as np


class BatchNorm1d:
    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference.
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization.
        :param Z: batch of input data Z (N, num_features).
        :param eval: flag to indicate training or inference mode.
        :return: batch normalized data.
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = np.mean(Z, axis=0, keepdims=True)
        self.V = np.mean((Z - self.M) ** 2, axis=0, keepdims=True)

        if eval == False:
            # training mode
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            # inference mode
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward pass for batch normalization.
        :param dLdBZ: Gradient loss wrt the output of BatchNorm transformation for Z (N, num_features).
        :return: Gradient of loss (L) wrt batch of input batch data Z (N, num_features).
        """
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW

        dLdV = np.sum(
            dLdNZ * (self.Z - self.M) * (-0.5) * (self.V + self.eps) ** (-1.5),
            axis=0,
            keepdims=True
        )

        dNZdM = -1 / np.sqrt(self.V + self.eps)
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True)

        dLdZ = (
            dLdNZ / np.sqrt(self.V + self.eps)
            + dLdV * 2 * (self.Z - self.M) / self.N
            + dLdM / self.N
        )

        return dLdZ


"""
Name : Bhuvana Teja Kanakam
Course : 11-785 Introduction to Deep Learning
Homework : 1 - Part 1

My Understanding : batch norm, normalizes the output of the layer using
the mean and variance of the current mini batch. helps in more stable training.

during the fp, have to calc mean and variance - then normalize the input - 
and then scale + shift using bw bb.

during the bpm the graident needs to pass through the scaling, normalization, 
var and mean calculations. calc grad, combine and compute final.
"""