import numpy as np
import sys
import os

# Add src to path to import activations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.activations import relu, softmax, sigmoid, tanh

class Dense:
    def __init__(self, input_dim, output_dim, activation=None):
        """
        From-scratch Dense (Fully Connected) layer.
        
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of neurons in the layer.
            activation (str): Activation function to use ('relu', 'softmax', 'sigmoid', 'tanh', or None).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Weights and bias to be loaded from Keras
        self.weights = None # Kernel: (input_dim, output_dim)
        self.bias = None    # Bias: (output_dim,)

    def load_weights(self, weights):
        """
        Load weights from Keras Dense layer.
        Keras weights order: [kernel, bias]
        """
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
        """
        Forward pass.
        Args:
            x (np.ndarray): Input of shape (batch_size, ..., input_dim)
        Returns:
            np.ndarray: Output of shape (batch_size, ..., output_dim)
        """
        # z = x * W + b
        z = np.dot(x, self.weights) + self.bias
        return self._apply_activation(z)
