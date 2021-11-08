import numpy as np
from typing import *

class Optimizer(object):
    def __init__(self, lr: float, decay: float = None, momentum: float = None):
        self.lr = lr
        self.start_lr = lr
        self.decay = decay
        self.n_steps = 0
        self.momentum = momentum
    
    def __call__(self, layer):
        self.step(layer)
        
    def step(self, layer):
        raise NotImplementedError
        
    def update_lr(self):
        self.n_steps += 1
        if self.decay:          
            self.lr = self.start_lr * (1. / (1 + self.decay * self.n_steps))

class SGD(Optimizer):
    """
    -------------------
    Stoachastic Gradient Descent 
    -------------------
    PARAMS
        1. lr -> initial learning rate
        2. Decay -> how much to decay the initial learning per step
        3. Momentum -> Rolling average of gradient changes over a number of steps
    -------------------
    """
    def __init__(self, lr: float, decay: float = None, momentum: float = None):
        super(SGD, self).__init__(lr, decay, momentum)
    
    def step(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'): #create momentum arrs if they dont exist
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            weight_updates = self.momentum * layer.weight_momentums - self.lr * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - self.lr * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else: #vanilla
            weight_updates = -self.lr * layer.dweights 
            bias_updates = -self.lr * layer.dbiases
            
        layer.weights += weight_updates
        layer.biases += bias_updates


class AdaGrad(Optimizer):
    """
    -------------------
    Adaptive Gradients
    -------------------
    1. Has a per-paramter learning rate, rather than global.
    2. The cache holds history of squared gradients
    3. Epsilon prevents division by 0
    --------------
    PARAMS
        1. epislon -> keep small (1e-7, for example). Meant to remediate division by 0
    --------------
    Note: Can cause learning rate to stall out due the division operator as values become smaller
    """
    def __init__(self, lr: float, decay: float = None, epsilon: float = 1e-7):
        super(AdaGrad, self).__init__(lr, decay)
        self.eps = epsilon
        
    def add_weight_cache(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
    
    def update_params(self, layer):
        layer.weights += -self.lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.eps)
        layer.biases += -self.lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.eps)
        
    def step(self, layer):
        self.add_weight_cache(layer) 
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        self.update_params(layer)
    
        
        
class RMSProp(AdaGrad):
    """
    -------------------
    Root Mean Square Propagation
    -------------------
    1. Same as AdaGrad except the calculation of cache

    2. Cache Calculation:
            >>> cache = rho * cache + (1 - rho) * gradient ** 2
            >>> #rho is the cache memory decay rate.
    
    3. Small gradient updates are enough to keep it going, so initial learning rate needs to be small (0.001)
    --------------
    PARAMS
        1. rho -> cache memory decay rate
    --------------
    """
    def __init__(self, lr: float, decay: float = None, epsilon: float = 1e-7, rho: float = 0.9):
        super(RMSProp, self).__init__(lr, decay, epsilon)
        self.rho = rho
        
    def step(self, layer):
        self.add_weight_cache(layer)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        self.update_params(layer)

        
class Adam(AdaGrad):
    """
    -------------------
    Adaptive Momentum
    -------------------
    1. Adds a bias correction mechanism applied to the cache and momentum
            > Bias correction compensates for initial zeroed values before they warm up with initial steps
            > Cache and Momentum are divided by 1-beta**step.
            > As steps increases beta**step decreases and approaches 0
    --------------
    PARAMS
        1. beta1 -> divides the momentum for bias correction
        2. beta2 -> divides the cache for bias correction
    --------------
    """
    def __init__(self, 
                 lr: float, 
                 decay: float = None, 
                 epsilon: float = 1e-7, 
                 beta1: float = 0.9, 
                 beta2: float = 0.999):
        super(Adam, self).__init__(lr, decay, epsilon)
        self.beta1 = beta1
        self.beta2 = beta2

        
    def add_weight_cache(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)

    def step(self, layer):
        self.add_weight_cache(layer)
        exp_step = self.n_steps + 1
        
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (exp_step))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (exp_step))
        
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (exp_step))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (exp_step))
        
        layer.weights += -self.lr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.eps)
        layer.biases += -self.lr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.eps)

        

