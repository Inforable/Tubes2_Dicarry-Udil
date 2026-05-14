import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from shared.activations import tanh

class EmbeddingLayer:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = None 

    def load_weights(self, weights):
        self.weights = weights[0]

    def forward(self, x):
        return self.weights[x]

class SimpleRNNCell:
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

    def forward(self, x, h_prev):
        h_next = tanh(np.dot(x, self.W_x) + np.dot(h_prev, self.W_h) + self.b)
        return h_next

class StackedRNN:
    def __init__(self, input_dim, hidden_dim, num_layers):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.cells.append(SimpleRNNCell(in_d, hidden_dim))

    def load_weights(self, keras_model):
        for i in range(self.num_layers):
            layer = keras_model.get_layer(f'simple_rnn_{i}')
            self.cells[i].load_weights(layer.get_weights())

    def forward(self, x, h_prev_list):
        h_next_list = []
        curr_input = x
        for i in range(self.num_layers):
            h_next = self.cells[i].forward(curr_input, h_prev_list[i])
            curr_input = h_next
            h_next_list.append(h_next)
        return h_next_list