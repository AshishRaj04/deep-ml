
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        # Initialize weights W, biases w0, and optimizers
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, size=(self.input_shape[0], self.n_units))
        self.w0 = np.zeros(shape=(1, self.n_units))
        self.optimizer_W = copy.deepcopy(optimizer)
        self.optimizer_w0 = copy.deepcopy(optimizer)


    def parameters(self):
        # Return total number of parameters
        a,b = self.W.shape
        m,n = self.w0.shape
        return (a*b) + (m*n)

        

    def forward_pass(self, X, training=True):
        # Compute and return the forward pass
        self.X = X
        # self.trainable = training
        
        return X @ self.W + self.w0
        

    def backward_pass(self, accum_grad):
        # Compute gradients, update weights if trainable, return grad w.r.t. input
        W_old = self.W.copy()
        if self.trainable:
            grad_W = self.X.T @ accum_grad
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)

            self.W = self.optimizer_W.update(self.W, grad_W)
            self.w0 = self.optimizer_w0.update(self.w0, grad_w0)
        
        return accum_grad @ W_old.T


    def output_shape(self):
        # Return output shape tuple
        return (self.n_units,)
