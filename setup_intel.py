import os
import shutil
import random
import zipfile

def setup_intel_dataset(zip_path="data/intel/archive.zip", extract_to="data/intel"):
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found. Please download it from Kaggle and place it there.")
        return

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    train_dir = os.path.join(extract_to, "train")
    val_dir = os.path.join(extract_to, "val")
    test_dir = os.path.join(extract_to, "test")

    # Paths inside the extracted zip
    seg_train_root = os.path.join(extract_to, "seg_train", "seg_train")
    seg_test_root = os.path.join(extract_to, "seg_test", "seg_test")
    seg_pred_root = os.path.join(extract_to, "seg_pred")

    print("Organizing folders...")
    # Move and rename
    if os.path.exists(seg_train_root):
        os.rename(seg_train_root, train_dir)
    if os.path.exists(seg_test_root):
        os.rename(seg_test_root, test_dir)

    # Cleanup extra nested folders
    for folder in ["seg_train", "seg_test", "seg_pred"]:
        path = os.path.join(extract_to, folder)
        if os.path.exists(path):
            shutil.rmtree(path)

    # Create val split
    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    print("Creating 80/20 train/val split...")
    for c in classes:
        os.makedirs(os.path.join(val_dir, c), exist_ok=True)
        class_train_dir = os.path.join(train_dir, c)
        class_val_dir = os.path.join(val_dir, c)
        
        images = os.listdir(class_train_dir)
        random.seed(42)  # For reproducibility
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.2)
        val_images = images[:split_idx]
        
        for img in val_images:
            shutil.move(os.path.join(class_train_dir, img), os.path.join(class_val_dir, img))

    print("\nSuccess! Intel dataset is ready in data/intel/")
    print(f"Train: {train_dir}")
    print(f"Val:   {val_dir}")
    print(f"Test:  {test_dir}")

if __name__ == "__main__":
    setup_intel_dataset()
