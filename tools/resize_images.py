import os
from PIL import Image

# Define paths
input_dir = '../dataset'  # Replace with the path to your dataset
output_dir = '../resized_images'
os.makedirs(output_dir, exist_ok=True)

# Resize all BMP images to 800x800 and save in output_dir
def resize_images(input_dir, output_dir, size=(800, 800)):
    for filename in os.listdir(input_dir):
        if filename.endswith('.bmp'):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size)
            img_resized.save(os.path.join(output_dir, filename))
            print(f'Resized and saved: {filename}')

resize_images(input_dir, output_dir)
