import numpy as np
from typing import *
from .utils import dims

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

class MeanSquaredError(Loss):
    """
    Mean Squared Error
    -----------------------
    Calculation:
        1. Forward
            MEAN( y[0] - yhat[0]) ** 2 + (y[1] - yhat[1]) ** 2 + ... + (y[n] - yhat[n]) ** 2 )
        2. Backward:
            partial_deriv is:
                (-2(y - yhat))/num_outputs (to normalize)
    -----------------------
    Note:
        axis=-1 tells numpy to calculate mean across outputs, for each sample separately
    """
    def forward(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.mean((y - yhat)**2, axis=-1)

    def backward(self, dvalues: np.ndarray, y: np.ndarray):
        num_samples, num_outputs = len(dvalues), len(dvalues[0])
        self.dinputs = -2 * (y - dvalues) / num_outputs #gradient
        self.dinputs = self.dinputs / num_samples #normalize
        return self.dinputs

class MeanAboluteError(Loss):
    """
    Mean Abolute Error
    -----------------------
    Calculation:
        1. Forward
            MEAN( abs(y[0] - yhat[0]) + ... + abs(y[n] - yhat[n]) )
        2. Backward:
            a. Deriv for abs = 1 if val > 0 else -1
    -----------------------
    Note:
        axis=-1 tells numpy to calculate mean across outputs, for each sample separately
    """
    def forward(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(y - yhat), axis=-1)

    def backward(self, dvalues: np.ndarray, y: np.ndarray):
        num_samples, num_outputs = len(dvalues), len(dvalues[0])
        self.dinputs = np.sign(y - dvalues) / num_outputs
        self.dinputs = self.dinputs / num_samples
        return self.dinputs

class CategoricalCrossentropy(Loss):
    """
    Calculates negative log liklihood (nll)
    ----------------
    Use For multiclass classification
    ----------------
    Targets should be sparse, like so [1 0 0 0 2 0 0 1 0 0 2]
    ----------------
    """
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

class BinaryCrossEntropy(Loss):
    """
    Use For Binary Classification
    ----------------
    Sum the Log Liklihoods of correct/incorrect
    np.mean( -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) , axis=-1)
    ----------------
    Note, we need to reshape targets so that they are no longer sparse
        [0, 0, 0, 1] -> reshape(-1,1) -> [[0], [0], [0], [1]]
    
    """
    def forward(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
        loss = -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        loss = np.mean(loss, axis=-1)
        return loss
    

    def backward(self, dvalues: np.ndarray, y: np.ndarray) -> np.ndarray:
        num_samples, num_outputs = len(dvalues), len(dvalues[0])
        dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y / dvalues - (1 - y) / (1 - dvalues)) / num_outputs
        self.dinputs = self.dinputs / num_samples
        return self.dinputs


