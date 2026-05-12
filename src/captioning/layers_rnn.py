import numpy as np
import sys
import os

# Add src to path to import activations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.activations import tanh, softmax

class EmbeddingLayer:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = None # Shape: (vocab_size, embed_dim)

    def load_weights(self, weights):
        """Load weights from a Keras model (list of one numpy array)."""
        self.weights = weights[0]

    def forward(self, x):
        """
        Forward pass for embedding layer.
        Args:
            x (np.ndarray): Input indices of shape (batch_size, seq_len)
        Returns:
            np.ndarray: Embedded sequences of shape (batch_size, seq_len, embed_dim)
        """
        return self.weights[x]

class SimpleRNNCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Weights to be loaded from Keras
        self.W_x = None # Kernel (input to hidden)
        self.W_h = None # Recurrent kernel (hidden to hidden)
        self.b = None   # Bias

    def load_weights(self, weights):
        """
        Load weights from Keras SimpleRNN.
        Keras weights order: [kernel, recurrent_kernel, bias]
        """
        self.W_x = weights[0]
        self.W_h = weights[1]
        self.b = weights[2]

    def forward(self, x, h_prev):
        """
        Single timestep forward pass.
        Args:
            x (np.ndarray): Input at current timestep (batch_size, input_dim)
            h_prev (np.ndarray): Hidden state from previous timestep (batch_size, hidden_dim)
        Returns:
            np.ndarray: New hidden state (batch_size, hidden_dim)
        """
        # h_t = tanh(x_t * W_x + h_{t-1} * W_h + b)
        h_next = tanh(np.dot(x, self.W_x) + np.dot(h_prev, self.W_h) + self.b)
        return h_next

class RNNLayer:
    """Wrapper to handle a sequence through SimpleRNNCell."""
    def __init__(self, cell):
        self.cell = cell

    def forward(self, x, h_initial=None):
        """
        Forward pass for a sequence.
        Args:
            x (np.ndarray): Input sequence (batch_size, seq_len, input_dim)
            h_initial (np.ndarray): Initial hidden state (batch_size, hidden_dim)
        Returns:
            np.ndarray: All hidden states (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        hidden_dim = self.cell.hidden_dim
        
        if h_initial is None:
            h_initial = np.zeros((batch_size, hidden_dim))
            
        h_states = np.zeros((batch_size, seq_len, hidden_dim))
        h_curr = h_initial
        
        for t in range(seq_len):
            h_curr = self.cell.forward(x[:, t, :], h_curr)
            h_states[:, t, :] = h_curr
            
        return h_states
