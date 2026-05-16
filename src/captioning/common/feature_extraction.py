import os
import json
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from cnn.utils import extract_and_save_features

def resolve_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def main():
    project_root = resolve_project_root()
    images_dir = os.path.join(project_root, "data", "flickr8k", "images")
    save_dir = os.path.join(project_root, "data", "flickr8k")
    features_path = os.path.join(save_dir, "features.npy")
    index_path = os.path.join(save_dir, "features_index.json")

    print(f"Resolved project root: {project_root}")
    print(f"Resolved images_dir: {images_dir}")
    print(f"Resolved save_dir: {save_dir}")
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} not found.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Initialize Pretrained Model
    base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    model.trainable = False
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        print(f"Error: No image files found in {images_dir}")
        return
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    
    # Create and save index mapping
    index_mapping = {filename: i for i, filename in enumerate(image_files)}
    with open(index_path, 'w') as f:
        json.dump(index_mapping, f)
    print(f"Index mapping saved to {index_path}")
    
    # Extract and save features
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