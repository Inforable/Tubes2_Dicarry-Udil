import numpy as np
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from captioning.lstm.layers import EmbeddingLayer, StackedLSTMCell
from shared.dense import Dense
from captioning.common import model_config as model_config

class LSTMCaptioner:
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_length, num_layers=1, cnn_feature_dim=2048):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_layers = num_layers
        self.cnn_feature_dim = cnn_feature_dim

        self.image_projection = Dense(cnn_feature_dim, embed_dim, activation='relu')
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.stacked_lstm = StackedLSTMCell(embed_dim, hidden_dim, num_layers)
        self.output_layer = Dense(hidden_dim, vocab_size, activation='softmax')

    def load_weights_from_keras(self, keras_model):
        self.embedding.load_weights(keras_model.get_layer('embedding').get_weights())
        self.image_projection.load_weights(keras_model.get_layer('image_projection').get_weights())
        self.stacked_lstm.load_weights(keras_model)
        self.output_layer.load_weights(keras_model.get_layer('dense_output').get_weights())

    def generate_caption_greedy(self, image_features, tokenizer, max_length=None):
        if max_length is None:
            max_length = self.max_length
        
        img_embed = self.image_projection.forward(image_features)
        
        h_states = [np.zeros((1, self.hidden_dim)) for _ in range(self.num_layers)]
        c_states = [np.zeros((1, self.hidden_dim)) for _ in range(self.num_layers)]
        
        h_states, c_states = self.stacked_lstm.forward(img_embed, h_states, c_states)
        
        result_caption = []
        curr_token = tokenizer.word_index[tokenizer.start_token]
        
        for step in range(max_length):
            x = self.embedding.forward(np.array([[curr_token]]))
            x_input = x[:, 0, :] if x.ndim == 3 else x
            
            h_states, c_states = self.stacked_lstm.forward(x_input, h_states, c_states)
            
            preds = self.output_layer.forward(h_states[-1])
            curr_token = np.argmax(preds[0])
            
            word = tokenizer.index_word.get(curr_token, '')
            if word == tokenizer.end_token or word == tokenizer.pad_token:
                break
            
            if word and word != tokenizer.start_token:
                result_caption.append(word)
        
        return ' '.join(result_caption)

    def generate_caption_beam(self, image_features, tokenizer, beam_width=3, max_length=None):
        if max_length is None:
            max_length = self.max_length
        
        img_embed = self.image_projection.forward(image_features)
        
        h_states_init = [np.zeros((1, self.hidden_dim)) for _ in range(self.num_layers)]
        c_states_init = [np.zeros((1, self.hidden_dim)) for _ in range(self.num_layers)]
        
        h_states_init, c_states_init = self.stacked_lstm.forward(img_embed, h_states_init, c_states_init)
        
        start_token = tokenizer.word_index[tokenizer.start_token]
        beam = [(0.0, [start_token], (h_states_init, c_states_init), False)]
        final_sequences = []
        
        for step in range(max_length):
            candidates = []
            
            for log_prob, tokens, state, _ in beam:
                if len(tokens) > 0:
                    curr_token = tokens[-1]
                    x_emb = self.embedding.forward(np.array([[curr_token]]))
                    x_input = x_emb[:, 0, :] if x_emb.ndim == 3 else x_emb
                    
                    h_states, c_states = state
                    
                    h_states_cp = [h.copy() for h in h_states]
                    c_states_cp = [c.copy() for c in c_states]
                    
                    h_states_new, c_states_new = self.stacked_lstm.forward(x_input, h_states_cp, c_states_cp)
                        
                    preds = self.output_layer.forward(h_states_new[-1])[0]
                    log_preds = np.log(preds + 1e-10)
                    top_k_indices = np.argsort(log_preds)[-beam_width:][::-1]
                    
                    for next_token_idx in top_k_indices:
                        next_token = int(next_token_idx)
                        new_log_prob = log_prob + log_preds[next_token]
                        new_tokens = tokens + [next_token]
                        
                        word = tokenizer.index_word.get(next_token, '')
                        is_end = (word == tokenizer.end_token or word == tokenizer.pad_token)
                        
                        candidates.append((new_log_prob, new_tokens, (h_states_new, c_states_new), is_end))
            
            candidates.sort(reverse=True, key=lambda x: x[0])
            completed = [c for c in candidates if c[3]]
            ongoing = [c for c in candidates if not c[3]]
            
            final_sequences.extend(completed)
            beam = ongoing[:beam_width]
            if len(beam) == 0:
                break
                
        final_sequences.extend(beam)
        if final_sequences:
            _, best_tokens, _, _ = max(final_sequences, key=lambda x: x[0])
        else:
            best_tokens = [start_token]
        
        result_caption = []
        for token in best_tokens[1:]:
            word = tokenizer.index_word.get(token, '')
            if word == tokenizer.end_token or word == tokenizer.pad_token:
                break
            if word and word != tokenizer.start_token:
                result_caption.append(word)
                
        return ' '.join(result_caption)
    
    def save_model_config(self, save_path):
        return model_config.save_config(self, save_path)

    @classmethod
    def load_model_config(cls, config_path):
        cfg = model_config.load_config(config_path)
        return cls(**cfg)
