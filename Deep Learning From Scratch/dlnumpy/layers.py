import numpy as np
from typing import *

class Layer(object):
    def __init__(self, 
                 input_size: int, 
                 num_neurons: int,
                 L1w: float = 0, 
                 L1b: float = 0, 
                 L2w: float = 0, 
                 L2b: float = 0):
        self.weights = 0.1 * np.random.randn(input_size, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.L1w = L1w
        self.L1b = L1b
        self.L2w = L2w
        self.L2b = L2b
        
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, dvalues: np.ndarray):
        raise NotImplementedError
        
    def deriv_reg(self):    
        if self.L1w > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.L1w * dL1
        if self.L1b > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.L1b * dL1
        if self.L2w > 0:
            self.dweights += 2 * self.L2w * self.weights
        if self.L2b > 0:
            self.dbiases += 2 * self.L2b * self.biases


class Dense(Layer):
    def __init__(self, 
                 input_size: int, 
                 num_neurons: int, 
                 L1w: float = 0, 
                 L1b: float = 0, 
                 L2w: float = 0, 
                 L2b: float = 0):
        super(Dense, self).__init__(input_size, num_neurons, L1w, L1b, L2w, L2b)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return np.dot(X, self.weights) + self.biases
    
    def backward(self, dvalues: np.ndarray):
        self.dweights = np.dot(self.X.T, dvalues) 
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) # Gradient on values
        self.deriv_reg()
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
    
    
class Dropout(Layer):
    """
    Dropout Layer
    -------------
    PARAMS
    -------------
    1. 'rate' -> the rate at which to exclude neuron output
        a. Note: We invert the rate to achieve this, like in tensorflow. In PyTorch, the dropout rate
                 denotes how many neuron outputs to KEEP
    -------------
    """
    def __init__(self, rate: float):
        self.rate = 1 - rate
        
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.bin_mask = np.random.binomial(1, self.rate, size=X.shape) / self.rate
        return X * self.bin_mask
        
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues * self.bin_mask
        return self.dinputs
    