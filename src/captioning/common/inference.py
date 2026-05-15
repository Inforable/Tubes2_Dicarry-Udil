import numpy as np

def generate_caption_greedy(captioner, image_features, tokenizer, max_length=None):
    if hasattr(captioner, 'generate_caption_greedy'):
        return captioner.generate_caption_greedy(image_features, tokenizer, max_length=max_length)

    if hasattr(captioner, 'generate_caption'):
        return captioner.generate_caption(image_features, tokenizer)

    raise AttributeError('Captioner does not implement a compatible generate method')


def generate_caption_batch(captioner, image_features_batch, tokenizer, max_length=None):
    batch_size = image_features_batch.shape[0]
    captions = []
    for i in range(batch_size):
        img_feat = image_features_batch[i:i+1, :]
        cap = generate_caption_greedy(captioner, img_feat, tokenizer, max_length=max_length)
        captions.append(cap)
    return captions


def generate_caption_beam(captioner, image_features, tokenizer, beam_width=3, max_length=None):
    if max_length is None:
        max_length = captioner.max_length
    
    img_embed = captioner.image_projection.forward(image_features)
    
    # Initialize hidden and cell states based on architecture
    is_lstm = hasattr(captioner, 'lstm_cells')  # True for LSTMCaptioner
    
    if is_lstm:
        # LSTM: h_states dan c_states lists
        h_states_init = [np.zeros((1, captioner.hidden_dim)) for _ in range(captioner.num_layers)]
        c_states_init = [np.zeros((1, captioner.hidden_dim)) for _ in range(captioner.num_layers)]
        
        # Pre-inject phase
        x = img_embed
        for layer_idx in range(captioner.num_layers):
            h_states_init[layer_idx], c_states_init[layer_idx] = captioner.lstm_cells[layer_idx].forward(
                x, h_states_init[layer_idx], c_states_init[layer_idx]
            )
            x = h_states_init[layer_idx]
        
        initial_state = (h_states_init, c_states_init)
    else:
        # RNN: h_list only
        h_list_init = [np.zeros((1, captioner.hidden_dim)) for _ in range(captioner.num_layers)]
        h_list_init = captioner.stacked_rnn.forward(img_embed, h_list_init)
        initial_state = h_list_init
    
    # Initialize beam: (log_prob, [tokens], state, completed)
    start_token = tokenizer.word_index[tokenizer.start_token]
    beam = [(0.0, [start_token], initial_state, False)]
    final_sequences = []
    
    for step in range(max_length):
        candidates = []
        
        for log_prob, tokens, state, _ in beam:
            if len(tokens) > 0:
                curr_token = tokens[-1]
                
                # Embed current token
                x = captioner.embedding.forward(np.array([[curr_token]]))
                
                # Forward through model based on architecture
                if is_lstm:
                    h_states, c_states = state
                    h_states_new = h_states.copy()
                    c_states_new = c_states.copy()
                    
                    x_forward = x
                    for layer_idx in range(captioner.num_layers):
                        x_input = x_forward[:, 0, :] if x_forward.ndim == 3 else x_forward
                        h_states_new[layer_idx], c_states_new[layer_idx] = captioner.lstm_cells[layer_idx].forward(
                            x_input, h_states_new[layer_idx], c_states_new[layer_idx]
                        )
                        x_forward = h_states_new[layer_idx]
                    
                    preds = captioner.output_layer.forward(h_states_new[-1])[0]
                    state_new = (h_states_new, c_states_new)
                else:
                    h_list = state
                    x_input = x[:, 0, :] if x.ndim == 3 else x
                    h_list_new = captioner.stacked_rnn.forward(x_input, h_list.copy())
                    preds = captioner.output_layer.forward(h_list_new[-1])[0]
                    state_new = h_list_new
                
                # Get log probabilities
                log_preds = np.log(preds + 1e-10)
                
                # Get top-k next tokens
                top_k_indices = np.argsort(log_preds)[-beam_width:][::-1]
                
                for next_token_idx in top_k_indices:
                    next_token = int(next_token_idx)
                    new_log_prob = log_prob + log_preds[next_token]
                    new_tokens = tokens + [next_token]
                    
                    # Check if this is end token
                    word = tokenizer.index_word.get(next_token, '')
                    is_end = (word == tokenizer.end_token or word == tokenizer.pad_token)
                    
                    candidates.append((new_log_prob, new_tokens, state_new, is_end))
        
        # Sort by log probability and keep top-k
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Separate completed and ongoing sequences
        completed = [c for c in candidates if c[3]]
        ongoing = [c for c in candidates if not c[3]]
        
        # Add completed sequences to final list
        final_sequences.extend(completed)
        
        # Keep top beam_width ongoing sequences
        beam = ongoing[:beam_width]
        
        # If no more ongoing sequences, we're done
        if len(beam) == 0:
            break
    
    # Add remaining ongoing sequences to final list
    final_sequences.extend(beam)
    
    # Get best sequence overall
    if final_sequences:
        best_log_prob, best_tokens, _, _ = max(final_sequences, key=lambda x: x[0])
    else:
        best_tokens = [start_token]
    
    # Convert token sequence to caption
    result_caption = []
    for token in best_tokens[1:]:  # Skip start token
        word = tokenizer.index_word.get(token, '')
        
        if word == tokenizer.end_token or word == tokenizer.pad_token:
            break
        
        if word and word != tokenizer.start_token:
            result_caption.append(word)
    
    return ' '.join(result_caption)
