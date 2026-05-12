import json

def save_config(obj, save_path):
    """Save minimal model configuration."""
    config = {
        'vocab_size': getattr(obj, 'vocab_size', None),
        'embed_dim': getattr(obj, 'embed_dim', None),
        'hidden_dim': getattr(obj, 'hidden_dim', None),
        'max_length': getattr(obj, 'max_length', None),
        'cnn_feature_dim': getattr(obj, 'cnn_feature_dim', None)
    }
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(path):
    """Load config JSON and return dict."""
    with open(path, 'r') as f:
        return json.load(f)
