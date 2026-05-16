import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model

def load_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32')
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def load_batch(image_paths, target_size=(224, 224)):
    batch_images = []
    for path in image_paths:
        batch_images.append(load_image(path, target_size))
    return np.array(batch_images)

def extract_and_save_features(model, image_paths, target_size, save_path, batch_size=32):
    features = []
    num_images = len(image_paths)
    
    print(f"Extracting features for {num_images} images...")
    
    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_input = load_batch(batch_paths, target_size)
        
        # Extract features using the model
        batch_features = model.predict(batch_input, verbose=0)
        
        # If the output is (N, 1, 1, D), flatten it to (N, D)
        if len(batch_features.shape) > 2:
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
            
        features.append(batch_features)
        
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= num_images:
            print(f"Processed {min(i + batch_size, num_images)}/{num_images} images")
            
    final_features = np.concatenate(features, axis=0)
    np.save(save_path, final_features)
    print(f"Features successfully saved to {save_path}")
    return final_features
