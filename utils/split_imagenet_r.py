import os
import shutil
import random
import argparse

def split_imagenet_r(data_dir, train_ratio=0.8, seed=42):
    """
    Splits ImageNet-R dataset into train and test folders.
    Expected input structure:
    data_dir/
        n01443537/
        n01484850/
        ...
    
    Output structure:
    data_dir/
        train/
            n01443537/
            ...
        test/
            n01443537/
            ...
    """
    random.seed(seed)
    
    # Identify all category folders (e.g., nXXXXXXXX)
    # Skip 'train' and 'test' if they already exist
    categories = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and d not in ['train', 'test']]
    
    if not categories:
        print(f"No category folders found in {data_dir}. Please ensure the images are inside category-named folders.")
        return

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Starting split: {len(categories)} categories found.")
    
    for cat in categories:
        cat_src_path = os.path.join(data_dir, cat)
        images = [f for f in os.listdir(cat_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create category folders in train and test
        os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cat), exist_ok=True)
        
        # Move files
        for img in train_images:
            shutil.move(os.path.join(cat_src_path, img), os.path.join(train_dir, cat, img))
        for img in test_images:
            shutil.move(os.path.join(cat_src_path, img), os.path.join(test_dir, cat, img))
            
        # Remove empty source category folder
        if not os.listdir(cat_src_path):
            os.rmdir(cat_src_path)
        else:
            print(f"Warning: {cat_src_path} is not empty after moving images.")

    print("Split complete!")
    print(f"Train path: {train_dir}")
    print(f"Test path: {test_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split ImageNet-R into train and test sets.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the imagenet-r directory containing category folders.")
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio of training data (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    split_imagenet_r(args.data_dir, args.ratio, args.seed)
