import numpy as np

class SGD:
    def __init__(self, model, lr=0.1, momentum=0):
        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum

        # momentum : initalize the velocity for weight and biases.
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f") for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f") for i in range(self.L)]

    def step(self):
        for i in range(self.L):
            if self.mu == 0:
                # If no momentum, use standard SGD update.
                self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW
                self.l[i].b = self.l[i].b - self.lr * self.l[i].dLdb

            else:
                # If momentum is used, update velocity terms.
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb

                # Update weights and biases using momentum and learning rate.
                self.l[i].W = self.l[i].W - self.lr * self.v_W[i]
                self.l[i].b = self.l[i].b - self.lr * self.v_b[i]

"""
Name : Bhuvana Teja Kanakam
Course : 11-785 Introduction to Deep Learning
Homework : 1 - Part 1

My Understanding : during the backpropogation, using gradients, to update 
the  weights and biases, sgd is used.

without the momentum - subract the lr multiplised by the gradient.
with the momentum, have velocity for both w and b, the velocity then keeps 
information fro the prev update.velocity is then updated using momentum 
value and current gradient
"""