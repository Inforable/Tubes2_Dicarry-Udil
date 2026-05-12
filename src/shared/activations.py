import numpy as np

def relu(x):
    """
    Rectified Linear Unit activation function.
    f(x) = max(0, x)
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax activation function.
    S(x_i) = exp(x_i) / sum(exp(x_j))
    
    Handles multi-dimensional arrays (N, C) by applying softmax across the last axis.
    """
    # Subtracting the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    """
    Sigmoid activation function.
    f(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """
    Hyperbolic tangent activation function.
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return np.tanh(x)
