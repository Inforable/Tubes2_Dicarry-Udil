import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from shared.activations import tanh, sigmoid, softmax

class EmbeddingLayer:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = None

    def load_weights(self, weights):
        self.weights = weights[0]

    def forward(self, x):
        return self.weights[x]


class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_x = None
        self.W_h = None
        self.b = None

    def load_weights(self, weights):
        self.W_x = weights[0]
        self.W_h = weights[1]
        self.b = weights[2]

    def forward(self, x, h_prev, c_prev):
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

class StackedLSTMCell:
    def __init__(self, input_dim, hidden_dim, num_layers):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = []

        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.cells.append(LSTMCell(in_d, hidden_dim))
    
    def load_weights(self, keras_model):
        for i in range(self.num_layers):
            layer = keras_model.get_layer(f'lstm_{i}')
            self.cells[i].load_weights(layer.get_weights())
    
    def forward(self, x, h_prev_list, c_prev_list):
        h_next_list = []
        c_next_list = []
        curr_input = x
        
        for i in range(self.num_layers):
            h_next, c_next = self.cells[i].forward(curr_input, h_prev_list[i], c_prev_list[i])
            curr_input = h_next
            h_next_list.append(h_next)
            c_next_list.append(c_next)
        
        return h_next_list, c_next_list

class LSTMLayer:
    def __init__(self, stacked_cell):
        self.stacked_cell = stacked_cell
    
    def forward(self, x, h_initial_list=None, c_initial_list=None):
        batch_size, seq_len, _ = x.shape
        num_layers = self.stacked_cell.num_layers
        hidden_dim = self.stacked_cell.hidden_dim

        if h_initial_list is None:
            h_initial_list = [np.zeros((batch_size, hidden_dim)) for _ in range(num_layers)]
        if c_initial_list is None:
            c_initial_list = [np.zeros((batch_size, hidden_dim)) for _ in range(num_layers)]

        h_states = np.zeros((batch_size, seq_len, hidden_dim))
        
        h_curr_list = h_initial_list
        c_curr_list = c_initial_list

        for t in range(seq_len):
            h_curr_list, c_curr_list = self.stacked_cell.forward(x[:, t, :], h_curr_list, c_curr_list)
            h_states[:, t, :] = h_curr_list[-1]  # Output dari layer terakhir
        
        return h_states, h_curr_list, c_curr_list
    