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
