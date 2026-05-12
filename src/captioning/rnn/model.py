import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from captioning.rnn.layers import EmbeddingLayer, SimpleRNNCell, RNNLayer
from shared.dense import Dense

class RNNCaptioner:
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_length):
        """
        Full RNN-based Image Captioning model from scratch.
        
        Architecture:
        1. Image Feature -> Dense Projection -> embed_dim (x_{-1})
        2. Caption tokens -> Embedding -> embed_dim
        3. Sequence [image_embed, caption_embeds] -> RNN -> hidden_states
        4. Hidden States -> Dense Output -> vocab_size (softmax)
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Layers
        self.image_projection = Dense(2048, embed_dim) # Assuming InceptionV3 features (2048)
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.rnn_cell = SimpleRNNCell(embed_dim, hidden_dim)
        self.rnn_layer = RNNLayer(self.rnn_cell)
        self.output_layer = Dense(hidden_dim, vocab_size, activation='softmax')

    def load_weights_from_keras(self, keras_model):
        """
        Load weights from a trained Keras model.
        Expects a model with layers in a specific order or reachable by name.
        """
        # This implementation assumes standard naming/order for simplicity
        # In practice, Member B will map these specifically from their .h5 files
        self.embedding.load_weights(keras_model.get_layer('embedding').get_weights())
        self.image_projection.load_weights(keras_model.get_layer('image_projection').get_weights())
        self.rnn_cell.load_weights(keras_model.get_layer('simple_rnn').get_weights())
        self.output_layer.load_weights(keras_model.get_layer('dense_output').get_weights())

    def generate_caption(self, image_features, tokenizer):
        """
        Greedy decoding to generate a caption for a single image.
        Args:
            image_features (np.ndarray): (1, 2048)
            tokenizer (Tokenizer): The fitted tokenizer instance
        Returns:
            str: Generated caption
        """
        # 1. Project image features to embedding space
        img_embed = self.image_projection.forward(image_features) # (1, embed_dim)
        
        # 2. Initial state
        h = np.zeros((1, self.hidden_dim))
        
        # 3. First step: Image injection (pre-inject method)
        h = self.rnn_cell.forward(img_embed, h)
        
        # 4. Iteratively generate tokens
        result_caption = []
        curr_token = tokenizer.word_index[tokenizer.start_token]
        
        for _ in range(self.max_length):
            # Embed current token
            x = self.embedding.forward(np.array([[curr_token]])) # (1, 1, embed_dim)
            
            # RNN step
            h = self.rnn_cell.forward(x[:, 0, :], h)
            
            # Output prediction
            preds = self.output_layer.forward(h) # (1, vocab_size)
            curr_token = np.argmax(preds[0])
            
            word = tokenizer.index_word.get(curr_token, '')
            if word == tokenizer.end_token or word == tokenizer.pad_token:
                break
            
            result_caption.append(word)
            
        return ' '.join(result_caption)
