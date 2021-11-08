import numpy as np
from typing import *
from .utils import dims

class Accuracy(object):
    def __call__(self, yhat: np.ndarray, y: np.ndarray) -> float:
        return self.calculate(yhat, y)
    
    def calculate(self, yhat: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.compare(yhat, y))
    
    def compare(self, yhat: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError
    
class Regression(Accuracy):
    def __init__(self, precision: float = None):
        self.precision = precision
        
    def compare(self, yhat: np.ndarray, y: np.ndarray) -> float:
        self.precision = np.std(y) / 250 if not self.precision else self.precision
        return np.abs(yhat - y) < self.precision

class Categorical(Accuracy):
    def __init__(self, binary: bool):
        self.binary = binary
        
    def compare(self, yhat: np.ndarray, y: np.ndarray) -> float:
        if not self.binary and dims(yhat) == 2:
            return (y == np.argmax(yhat, axis=1))
        return (y == yhat)
    