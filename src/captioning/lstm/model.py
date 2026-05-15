import numpy as np
import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from captioning.lstm.layers import EmbeddingLayer, LSTMCell, LSTMLayer
from shared.dense import Dense
from captioning.common import inference as inference_utils
from captioning.common import model_config as model_config


class LSTMCaptioner:
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_length, num_layers=1, cnn_feature_dim=2048):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_layers = num_layers
        self.cnn_feature_dim = cnn_feature_dim

        # Layers
        self.image_projection = Dense(cnn_feature_dim, embed_dim, activation=None)
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        
        # Create list of LSTM cells (one per layer)
        self.lstm_cells = []
        for i in range(num_layers):
            input_dim = embed_dim if i == 0 else hidden_dim
            self.lstm_cells.append(LSTMCell(input_dim, hidden_dim))
        
        self.output_layer = Dense(hidden_dim, vocab_size, activation='softmax')

    def load_weights_from_keras(self, keras_model):
        """Load weights from a trained Keras model."""
        self.embedding.load_weights(keras_model.get_layer('embedding').get_weights())
        self.image_projection.load_weights(keras_model.get_layer('image_projection').get_weights())
        
        # Load all LSTM layers
        lstm_layers_to_load = [layer for layer in keras_model.layers if layer.name.startswith('lstm_')]
        if len(lstm_layers_to_load) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} LSTM layers in model, but found {len(lstm_layers_to_load)}"
            )
        
        for i, (lstm_cell, keras_lstm_layer) in enumerate(zip(self.lstm_cells, lstm_layers_to_load)):
            lstm_cell.load_weights(keras_lstm_layer.get_weights())
        
        self.output_layer.load_weights(keras_model.get_layer('dense_output').get_weights())

    def generate_caption_greedy(self, image_features, tokenizer, max_length=None):
        """
        Generate caption untuk single image menggunakan greedy decoding.        
        Args:
            image_features (np.ndarray): CNN features untuk satu gambar, shape (1, cnn_feature_dim)
            tokenizer: Fitted tokenizer dengan word_index dan index_word
            max_length (int): Maksimum caption length (jika None, gunakan self.max_length)
        Returns:
            str: Generated caption
        """
        if max_length is None:
            max_length = self.max_length
        
        # Project CNN features ke embedding dimension
        img_embed = self.image_projection.forward(image_features)
        
        # Initialize hidden and cell states for all layers
        h_states = [np.zeros((1, self.hidden_dim)) for _ in range(self.num_layers)]
        c_states = [np.zeros((1, self.hidden_dim)) for _ in range(self.num_layers)]
        
        # Pre-inject phase: pass CNN feature through all LSTM layers
        x = img_embed
        for layer_idx in range(self.num_layers):
            h_states[layer_idx], c_states[layer_idx] = self.lstm_cells[layer_idx].forward(
                x, h_states[layer_idx], c_states[layer_idx]
            )
            x = h_states[layer_idx]
        
        # Iteratively generate tokens
        result_caption = []
        curr_token = tokenizer.word_index[tokenizer.start_token]
        
        for step in range(max_length):
            # Embed current token
            x = self.embedding.forward(np.array([[curr_token]]))
            
            # Pass through all LSTM layers
            for layer_idx in range(self.num_layers):
                x_input = x[:, 0, :] if x.ndim == 3 else x
                h_states[layer_idx], c_states[layer_idx] = self.lstm_cells[layer_idx].forward(
                    x_input, h_states[layer_idx], c_states[layer_idx]
                )
                x = h_states[layer_idx]
            
            # Output prediction from last hidden state
            preds = self.output_layer.forward(h_states[-1])
            
            # Greedy sampling
            curr_token = np.argmax(preds[0])
            
            # Convert token index to word
            word = tokenizer.index_word.get(curr_token, '')
            
            # Check stopping conditions
            if word == tokenizer.end_token or word == tokenizer.pad_token:
                break
            
            if word and word != tokenizer.start_token:
                result_caption.append(word)
        
        return ' '.join(result_caption)

    def generate_caption_batch(self, image_features_batch, tokenizer, max_length=None):
        return inference_utils.generate_caption_batch(self, image_features_batch, tokenizer, max_length=max_length)

    def save_model_config(self, save_path):
        return model_config.save_config(self, save_path)

    @classmethod
    def load_model_config(cls, config_path):
        cfg = model_config.load_config(config_path)
        return cls(**cfg)
