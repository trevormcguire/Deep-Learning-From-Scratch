import numpy as np
import matplotlib.pyplot as plt
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

def plot_history(history: Dict[str, np.ndarray], size: Tuple[int] = (12,8)):
    for field in ["loss", "accuracy"]:
        assert field in history, f"{field} must be a key in history"
    fig, ax = plt.subplots(figsize=size)
    ax.plot(history["loss"], c="red", label="Loss")
    ax2 = ax.twinx()
    ax2.plot(history["accuracy"], c="blue", label="Accuracy")
    plt.title("Training Loss and Accuracy")

    handles,labels = [],[]
    for ax in fig.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)

    plt.legend(handles, labels)
    plt.show()