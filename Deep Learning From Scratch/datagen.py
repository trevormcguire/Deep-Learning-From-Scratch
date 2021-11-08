import numpy as np
from typing import *


def spiral(points: int, classes: int) -> Tuple:
    """
    Copyright (c) 2015 Andrej Karpathy
    License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
    Source: https://cs231n.github.io/neural-networks-case-study/
    """
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

def sine(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.arange(n_samples).reshape(-1,1) / n_samples
    y = np.sin(2 * np.pi * X).reshape(-1,1)
    return X, y

def cosine(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.arange(n_samples).reshape(-1,1) / n_samples
    y = np.cos(2 * np.pi * X).reshape(-1,1)
    return X, y

def cosinewave(frequency: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(-frequency*np.pi, frequency*np.pi, 0.01)
    y = np.cos(x)
    return x, y

def sinewave(frequency: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(-frequency*np.pi, frequency*np.pi, 0.01)
    y = np.sin(x)
    return x, y
