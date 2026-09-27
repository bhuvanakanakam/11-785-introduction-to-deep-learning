import numpy as np
from .activation import Softmax


class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        # Calculate the squared error for each output
        se = (A - Y) ** 2

        # Sum all the squared errors and divide by the total number of values
        sse = np.sum(se)
        mse = sse / (self.N * self.C)

        return mse

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        # Differentiate the MSE with respect to A
        dLdA = (2 * (self.A - self.Y)) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        Hint: Read the writeup to determine the shapes of all the variables.
        Note: Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C, 1), dtype='f')
        Ones_N = np.ones((1, self.N), dtype='f')

        self.softmax = Softmax()

        # Convert the model output into probabilities
        softmax_output = self.softmax.forward(A)

        # Calculate cross entropy for each sample
        crossentropy = -self.Y * np.log(softmax_output + 1e-12)

        # Sum over the classes and then take the mean over the batch
        sum_crossentropy_loss = np.sum(crossentropy, axis=1)
        mean_crossentropy_loss = np.mean(sum_crossentropy_loss)

        return mean_crossentropy_loss

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        """
        # Gradient of cross entropy with softmax with respect to the model output
        dLdA = (self.softmax.forward(self.A) - self.Y) / self.N

        return dLdA

"""
Name : Bhuvana Teja Kanakam
Course : 11-785 Introduction to Deep Learning
Homework : 1 - Part 1

My Understanding : loss functions tell us how far the desired output is from
the actual target. during the fp, loss is the diff from the predictions to the 
target values. during the bp, graident of the loss wrt prediction, and then
pass it into the network.

for the mseloss, calc the squared diff between the prediction and target, 
and then the mean over all the samples and features.

for crossentropyloss, apply softmax to get probabilities and then 
calculate the cross entropy with the target. during bp, the gradient 
becomes the softmax output minus the target, divided by the batch size.
"""