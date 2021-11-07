import numpy as np
from typing import *

def one_hot_accuracy(yhat: np.ndarray, y: np.ndarray) -> float:
    if dims(yhat) == 2:
        return np.mean(y == np.argmax(yhat, axis=1))
    return np.mean(y == yhat)

def dims(arr: np.ndarray) -> int:
    return len(arr.shape)

def build_pred_method(final_layer: Callable) -> Callable:
    """
    Returns a Callable to transform raw logits to desired output, depending on activation type
    """
    pred_methods = {
                "Softmax": lambda outputs: np.argmax(outputs, axis=1),
                "Sigmoid": lambda outputs: (outputs > 0.5) * 1,
                "Linear": lambda outputs: outputs,
                "ReLU": lambda outputs: outputs
            }
    name = final_layer.__class__.__name__
    if name in pred_methods:
        return pred_methods[name]
    return lambda outputs: outputs #default, return no transformation function

