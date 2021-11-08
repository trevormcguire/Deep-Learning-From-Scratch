import numpy as np
from typing import *
from .utils import build_pred_method
import pickle

class Model(object):
    def __init__(self):
        self.layers = []
        self.num_layers = 0
        self.trainable = []
        self.pred_method = None
        
    def add(self, layer):
        self.layers.append(layer)
        self.num_layers += 1
        if hasattr(layer, "weights"):
            self.trainable.append(layer)
        self.pred_method = build_pred_method(self.layers[-1])
        
    def set(self, loss, optimizer, accuracy_fn): #params that follow the * are keyword arguments
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy_fn = accuracy_fn
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        for idx in range(self.num_layers):
            out = self.layers[idx](X) if idx == 0 else self.layers[idx](out)
        return out
    
    def backward(self, output: np.ndarray, y: np.ndarray):
        dinput = self.loss.backward(output, y)
        for layer in reversed(self.layers):
            dinput = layer.backward(dinput)
            
    def optimize(self):
        for layer in self.trainable:
            self.optimizer.step(layer)
        self.optimizer.update_lr()
        
    def calc_loss(self, output: np.ndarray, y: np.ndarray):
        loss = self.loss(output, y)
        for layer in self.trainable:
            loss += self.loss.regularize(layer)
        return loss
    
    
    def train(self, X: np.ndarray, y: np.ndarray, num_epochs: int, print_every: int = 100):
        history = {"loss": [], "accuracy": []}
        for epoch in range(1, num_epochs+1):
            out = self.forward(X)
            loss = self.calc_loss(out, y)
            history["loss"].append(loss)
            
            yhat = self.pred_method(out)
            acc = self.accuracy_fn(yhat, y)
            history["accuracy"].append(acc)
            
            self.backward(out, y)
            self.optimize()
            if print_every and not epoch % print_every:
                print(f"Epoch: {epoch}: Loss: {loss} || Accuracy: {acc}")

        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)
        return self.pred_method(out) #yhat

    def get_params(self) -> List:
        params = []
        for layer in self.trainable:
            params.append(layer.get_params())
        return params

    def set_params(self, params: List):
        for (weights, biases), layer in zip(params, self.trainable):
            layer.set_params(weights, biases)
 
    def save_params(self, path: str):
        params = self.get_params()
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, path: str):
        with open(path, 'rb') as f:
            params = pickle.load(f)
        self.set_params(params)
        
        