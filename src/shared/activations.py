import numpy as np

def relu(x):
    # f(x) = max(0, x)
    return np.maximum(0, x)

def softmax(x):
    # S(x_i) = exp(x_i) / sum(exp(x_j))
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    # f(x) = 1 / (1 + exp(-x))
    return 1 / (1 + np.exp(-x))

def tanh(x):
    # f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    return np.tanh(x)
