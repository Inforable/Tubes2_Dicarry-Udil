import os
import json
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import sys

# Add src to path to import cnn utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from cnn.utils import extract_and_save_features

def main():
    # 1. Setup paths
    images_dir = "data/flickr8k/images"
    save_dir = "data/flickr8k"
    features_path = os.path.join(save_dir, "features.npy")
    index_path = os.path.join(save_dir, "features_index.json")
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} not found.")
        return

    # 2. Initialize Pretrained Model (InceptionV3 recommended)
    # include_top=False removes the final classification layer
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    model.trainable = False  # Freeze weights
    
    # 3. Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    
    # 4. Create and save index mapping (filename -> index)
    # This is crucial so we know which row in features.npy belongs to which image
    index_mapping = {filename: i for i, filename in enumerate(image_files)}
    with open(index_path, 'w') as f:
        json.dump(index_mapping, f)
    print(f"Index mapping saved to {index_path}")
    
    # 5. Extract and save features
    # InceptionV3 expects 299x299 input
    target_size = (299, 299)
    extract_and_save_features(
        model=model,
        image_paths=image_paths,
        target_size=target_size,
        save_path=features_path,
        batch_size=32
    )

if __name__ == "__main__":
    main()
