import numpy as np
from typing import *
from losses import CategoricalCrossentropy
from utils import dims

class Activation(object):
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X: np.ndarray):
        raise NotImplementedError
    
    def backward(self, dvalues: np.ndarray):
        raise NotImplementedError

class Linear(Activation):
    """Returns itself"""
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        return dvalues
        # self.dinputs = dvalues.copy()
        # return self.dinputs

class ReLU(Activation):
    """returns X if X > 0 else 0"""
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues.copy()
        self.dinputs[self.X <= 0] = 0
        return self.dinputs
        
class Softmax(Activation):
    """
    Outputs a probability distribution that sums to 1. Great for one hot encoded targets (Categorical)
    """
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True) #probability dist
        return self.output
    
    def backward(self, dvalues: np.ndarray):
        self.dinputs = np.empty_like(dvalues)
        for index, (out, d) in enumerate(zip(self.output, dvalues)):
            out = out.reshape(-1, 1) #flatten
            jacobian_matrix = np.diagflat(out) - np.dot(out, out.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, d)
        return self.dinputs

class Sigmoid(Activation):
    """
    Sigmoid squashes values between 0 and 1
    Formula: 1 / (1 + np.exp(-val))
    """
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues * (1 - self.output) * self.output
        return self.dinputs


class SoftmaxCrossEntropy():
    """
    Softmax Activation - Categorical Cross Entropy Loss Combination 
    > 5.5x Faster when Combined
    > Efficiency Stats:
            Combined: 8.5 µs ± 71.5 ns per loop
            Seperate: 44.1 µs ± 641 ns per loop
    """
    def __init__(self):
        self.activation = Softmax()
        self.loss_fn = CategoricalCrossentropy()
    
    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        return self.forward(X, y)
        
    def forward(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        out = self.activation(X)
        return out, self.loss_fn(out, y)

    def backward(self, dvalues: np.ndarray, y: np.ndarray) -> np.ndarray:
        idxs = np.arange(len(dvalues))
        if dims(y) == 2:
            y = np.argmax(y, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[idxs, y] -= 1 # gradient
        self.dinputs = self.dinputs / len(dvalues) #normalize
        return self.dinputs