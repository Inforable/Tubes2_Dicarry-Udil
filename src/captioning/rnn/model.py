import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from captioning.rnn.layers import EmbeddingLayer, StackedRNN
from shared.dense import Dense

class RNNCaptioner:
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_length, num_layers=1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_layers = num_layers
        
        self.image_projection = Dense(2048, embed_dim, activation='relu')
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.stacked_rnn = StackedRNN(embed_dim, hidden_dim, num_layers)
        self.output_layer = Dense(hidden_dim, vocab_size, activation='softmax')

    def load_weights_from_keras(self, keras_model):
        self.embedding.load_weights(keras_model.get_layer('embedding').get_weights())
        self.image_projection.load_weights(keras_model.get_layer('image_projection').get_weights())
        self.stacked_rnn.load_weights(keras_model)
        self.output_layer.load_weights(keras_model.get_layer('dense_output').get_weights())

    def generate_caption(self, image_features, tokenizer):
        img_embed = self.image_projection.forward(image_features) # (1, embed_dim)
        
        # Initialize hidden states with zeros (h0)
        h_list = [np.zeros((1, self.hidden_dim)) for _ in range(self.num_layers)]
        
        # Pre-inject: Feature vector CNN di-project ke embed_dim, lalu masuk sebagai x_{-1}
        h_list = self.stacked_rnn.forward(img_embed, h_list)
        
        result_caption = []
        curr_token = tokenizer.word_index[tokenizer.start_token]
        
        # Greedy decoding step
        for _ in range(self.max_length):
            x = self.embedding.forward(np.array([[curr_token]])) # (1, 1, embed_dim)
            h_list = self.stacked_rnn.forward(x[:, 0, :], h_list)
            
            preds = self.output_layer.forward(h_list[-1]) 
            curr_token = np.argmax(preds[0])
            
            word = tokenizer.index_word.get(curr_token, '')
            if word == tokenizer.end_token or word == tokenizer.pad_token:
                break
            
            result_caption.append(word)
            
        return ' '.join(result_caption)