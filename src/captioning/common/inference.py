import numpy as np

def generate_caption_batch(captioner, image_features_batch, tokenizer, max_length=None, mode='greedy', beam_width=3):
    batch_size = image_features_batch.shape[0]
    captions = []
    
    for i in range(batch_size):
        img_feat = image_features_batch[i:i+1, :]
        if mode == 'greedy':
            cap = captioner.generate_caption_greedy(img_feat, tokenizer, max_length=max_length)
        elif mode == 'beam':
            cap = captioner.generate_caption_beam(img_feat, tokenizer, beam_width=beam_width, max_length=max_length)
            
        captions.append(cap)
        
    return captions