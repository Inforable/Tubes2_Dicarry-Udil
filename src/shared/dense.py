import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.activations import relu, softmax, sigmoid, tanh

class Dense:
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Weights and bias to be loaded from Keras
        self.weights = None # Kernel: (input_dim, output_dim)
        self.bias = None    # Bias: (output_dim,)

    def load_weights(self, weights):
        self.weights = weights[0]
        self.bias = weights[1]

    def _apply_activation(self, z):
        if self.activation_name == 'relu':
            return relu(z)
        elif self.activation_name == 'softmax':
            return softmax(z)
        elif self.activation_name == 'sigmoid':
            return sigmoid(z)
        elif self.activation_name == 'tanh':
            return tanh(z)
        return z

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self._apply_activation(z)
