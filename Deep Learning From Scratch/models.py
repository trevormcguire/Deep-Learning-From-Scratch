import numpy as np
from typing import *

class Model(object):
    def __init__(self):
        self.layers = []
        self.num_layers = 0
        self.trainable = []
        
    def add(self, layer):
        self.layers.append(layer)
        self.num_layers += 1
        if hasattr(layer, "weights"):
            self.trainable.append(layer)
        
    def set(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        for idx in range(self.num_layers):
            out = self.layers[idx](X) if idx == 0 else self.layers[idx](out)
        return out
    
    def backward(self, yhat: np.ndarray, y: np.ndarray):
        dinput = self.loss.backward(yhat, y)
        for layer in reversed(self.layers):
            dinput = layer.backward(dinput)
            
    def optimize(self):
        for layer in self.trainable:
            self.optimizer.step(layer)
        self.optimizer.update_lr()
        
    def calc_loss(self, yhat: np.ndarray, y: np.ndarray):
        loss = self.loss(yhat, y)
        for layer in self.trainable:
            loss += self.loss.regularize(layer)
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, num_epochs: int, print_every: int = 1):
        history = {"loss": [], "accuracy": []}
        for epoch in range(1, num_epochs+1):
            out = self.forward(X)
            loss = self.calc_loss(out, y)
            history["loss"].append(loss)
            
            self.backward(out, y)
            self.optimize()
        
        return history