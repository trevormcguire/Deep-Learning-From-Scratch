import numpy as np
from typing import *

def one_hot_accuracy(yhat: np.ndarray, y: np.ndarray) -> float:
    if dims(yhat) == 2:
        return np.mean(y == np.argmax(yhat, axis=1))
    return np.mean(y == yhat)

def dims(arr: np.ndarray) -> int:
    return len(arr.shape)

