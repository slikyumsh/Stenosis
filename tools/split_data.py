import shutil
import random
import os

# Define paths
train_dir = '../data/train'
val_dir = '../data/val'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split dataset into 80-20
def split_dataset(input_dir, train_dir, val_dir, split_ratio=0.8):
    # Collect all BMP image and XML annotation pairs
    files = [f for f in os.listdir(input_dir) if f.endswith('.bmp')]
    paired_files = [(f, f.replace('.bmp', '.xml')) for f in files]
    
    # Shuffle and split
    random.shuffle(paired_files)
    split_index = int(len(paired_files) * split_ratio)
    train_files = paired_files[:split_index]
    val_files = paired_files[split_index:]
    
    # Copy files to train and val directories
    for img_file, xml_file in train_files:
        shutil.copy(os.path.join(input_dir, img_file), train_dir)
        shutil.copy(os.path.join(input_dir, xml_file), train_dir)
    for img_file, xml_file in val_files:
        shutil.copy(os.path.join(input_dir, img_file), val_dir)
        shutil.copy(os.path.join(input_dir, xml_file), val_dir)
    
    print(f'Training set: {len(train_files)} samples')
    print(f'Validation set: {len(val_files)} samples')

split_dataset('resized_images', train_dir, val_dir)
