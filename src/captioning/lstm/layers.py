import numpy as np
import sys
import os

# Add src to path to import activations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from shared.activations import tanh, sigmoid, softmax

class EmbeddingLayer:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = None  # Shape: (vocab_size, embed_dim)

    def load_weights(self, weights):
        """Load weights from a Keras model (list of one numpy array)."""
        self.weights = weights[0]

    def forward(self, x):
        """
        Forward pass untuk embedding layer.
        
        Args:
            x (np.ndarray): Token indices dengan shape (batch_size, seq_len)
        
        Returns:
            np.ndarray: Embedded vectors dengan shape (batch_size, seq_len, embed_dim)
        """
        return self.weights[x]


class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Keras menyimpan semua 4 gates dalam satu matrix: [i_gate, f_gate, cell, o_gate]
        self.W_x = None  # Kernel shape: (input_dim, 4*hidden_dim)
        self.W_h = None  # Recurrent kernel shape: (hidden_dim, 4*hidden_dim)
        self.b = None    # Bias shape: (4*hidden_dim,)

    def load_weights(self, weights):
        """
        Load weights from Keras LSTM layer. 
        Keras weights order: [kernel, recurrent_kernel, bias]
        """
        self.W_x = weights[0]  # (input_dim, 4*hidden_dim)
        self.W_h = weights[1]  # (hidden_dim, 4*hidden_dim)
        self.b = weights[2]    # (4*hidden_dim,)

    def forward(self, x, h_prev, c_prev):
        """
        Single timestep forward pass LSTM.
        Args:
            x (np.ndarray): Input pada timestep saat ini, shape (batch_size, input_dim)
            h_prev (np.ndarray): Hidden state dari timestep sebelumnya, shape (batch_size, hidden_dim)
            c_prev (np.ndarray): Cell state (memory) dari timestep sebelumnya, shape (batch_size, hidden_dim)
        
        Returns:
            tuple: (h_new, c_new)
        """
        batch_size = x.shape[0]
        
        # Hitung pre-activation untuk 4 gates
        z = np.dot(x, self.W_x) + np.dot(h_prev, self.W_h) + self.b
        
        # Split z menjadi 4 komponen (untuk setiap gate)
        i = z[:, :self.hidden_dim] # Input gate
        f = z[:, self.hidden_dim:2*self.hidden_dim] # Forget gate
        g = z[:, 2*self.hidden_dim:3*self.hidden_dim] # Cell candidate
        o = z[:, 3*self.hidden_dim:] # Output gate
        
        # Apply activation functions untuk setiap gate
        i_t = sigmoid(i)
        f_t = sigmoid(f)
        g_t = tanh(g)
        o_t = sigmoid(o)
        
        # Update cell state (memory)
        c_new = f_t * c_prev + i_t * g_t
        
        # Hitung hidden state (output)
        h_new = o_t * tanh(c_new)
        
        return h_new, c_new


class LSTMLayer:
    """Wrapper to handle a sequence through LSTMCell."""
    def __init__(self, cell):
        self.cell = cell

    def forward(self, x, h_initial=None, c_initial=None):
        """
        Forward pass for entire sequence.
        Args:
            x (np.ndarray): Input sequence, shape (batch_size, seq_len, input_dim)
            h_initial (np.ndarray): Initial hidden state, shape (batch_size, hidden_dim)
            c_initial (np.ndarray): Initial cell state, shape (batch_size, hidden_dim)    
        Returns:
            np.ndarray: Semua hidden states, shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        hidden_dim = self.cell.hidden_dim
        
        if h_initial is None:
            h_initial = np.zeros((batch_size, hidden_dim))
        if c_initial is None:
            c_initial = np.zeros((batch_size, hidden_dim))
        
        h_states = np.zeros((batch_size, seq_len, hidden_dim))
        
        h_curr = h_initial
        c_curr = c_initial
        
        for t in range(seq_len):
            # Forward pass pada timestep t
            h_curr, c_curr = self.cell.forward(x[:, t, :], h_curr, c_curr)
            
            # Simpan hidden state
            h_states[:, t, :] = h_curr
        
        return h_states
