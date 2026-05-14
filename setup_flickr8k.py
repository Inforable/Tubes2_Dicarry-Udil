import os
import shutil
import zipfile

def setup_flickr8k_dataset(zip_path="data/flickr8k/archive.zip", extract_to="data/flickr8k"):
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found. Please download it from Kaggle and place it there.")
        return

    print("Extracting Flickr8k dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Standardize image folder name to lowercase "images"
    old_images_dir = os.path.join(extract_to, "Images")
    new_images_dir = os.path.join(extract_to, "images")

    if os.path.exists(old_images_dir):
        is_same_file = False
        if os.path.exists(new_images_dir):
            try:
                is_same_file = os.path.samefile(old_images_dir, new_images_dir)
            except OSError:
                pass
                
        if is_same_file:
            temp_dir = os.path.join(extract_to, "temp_images_dir_for_rename")
            os.rename(old_images_dir, temp_dir)
            os.rename(temp_dir, new_images_dir)
        elif os.path.exists(new_images_dir):
            # If "images" already exists (e.g. from previous run), merge/replace
            for item in os.listdir(old_images_dir):
                shutil.move(os.path.join(old_images_dir, item), os.path.join(new_images_dir, item))
            os.rmdir(old_images_dir)
        else:
            os.rename(old_images_dir, new_images_dir)

    print("\nSuccess! Flickr8k dataset is ready in data/flickr8k/")
    print(f"Images:   {new_images_dir}")
    print(f"Captions: {os.path.join(extract_to, 'captions.txt')}")

if __name__ == "__main__":
    setup_flickr8k_dataset()
