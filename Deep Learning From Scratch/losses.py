import numpy as np
from typing import *
from utils import dims

class Loss(object):
    def __call__(self, yhat: np.ndarray, y: np.ndarray):
        return self.calculate(yhat, y)
    
    def calculate(self, yhat: np.ndarray, y: np.ndarray):
        losses = self.forward(yhat, y)
        return np.mean(losses) #avg loss for the batch
        
    def forward(self, yhat: np.ndarray, y: np.ndarray):
        raise NotImplementedError
        
    def regularize(self, layer):
        reg_loss = 0
        #-----L1-----
        if layer.L1w > 0:
            reg_loss += layer.L1w * np.sum(np.abs(layer.weights))

        if layer.L1b > 0:
            reg_loss += layer.L1b * np.sum(np.abs(layer.biases))
        #-----L2-----
        if layer.L2w > 0:
            reg_loss += layer.L2w * np.sum(layer.weights ** 2)
    
        if layer.L2b > 0:
            reg_loss += layer.L2b * np.sum(layer.weights ** 2)
        
        return reg_loss


class CategoricalCrossentropy(Loss):
    """Calculates negative log liklihood (nll)"""
    def forward(self, yhat, y):
        idxs = np.arange(len(yhat))
        yhat = np.clip(yhat, 1e-7, 1-1e-7) #clip so no zero values
        y_dims = dims(y)
        if y_dims == 1:
            confidence = yhat[idxs, y]
        elif y_dims == 2:
            confidence = np.sum(yhat * y, axis=1)
        return -np.log(confidence) #nll
    
    def backward(self, dvalues, y):     
        """
        Normalization Rationale:
            1. During optimization, the gradients related to each weight will be summed.
               This means that the more samples we have, the larger the sum will become.
               To avoid this becoming an issue, we divide the gradients by the number of samples.
               This way, the optimizer is ignorant of the number of samples.
        """
        if dims(y) == 1: #one hot if labels are sparse
            y = np.eye(len(dvalues[0]))[y]
        self.dinputs = -y / dvalues #Calculate gradient
        self.dinputs = self.dinputs / len(dvalues) #normalize 
        return self.dinputs

