import os
import random
import xml.etree.ElementTree as ET
from PIL import Image
import math

# Define paths
input_dir = '../dataset'
output_dir = '../data'
train_ratio = 0.8
image_size = (800, 800)

# Create output directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def resize_image(image_path, output_path):
    """Resize image to the specified size and save it."""
    with Image.open(image_path) as img:
        img_resized = img.resize(image_size)
        img_resized.save(output_path)

def rotate_and_save(image_path, output_path, angle):
    """Rotate image by a given angle and save it."""
    with Image.open(image_path) as img:
        img_rotated = img.rotate(angle, expand=True)
        img_rotated = img_rotated.resize(image_size)  # Resize after rotation
        img_rotated.save(output_path)

def rotate_annotation(xml_path, angle, original_width, original_height):
    """Rotate XML annotation by a given angle."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    new_width, new_height = image_size
    rotated_annotations = []

    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # Calculate the center of the bounding box
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        # Translate the bounding box center to the origin
        x_center -= original_width / 2
        y_center -= original_height / 2

        # Apply rotation
        x_rotated = x_center * cos_angle - y_center * sin_angle
        y_rotated = x_center * sin_angle + y_center * cos_angle

        # Translate back to the new image center
        x_rotated += new_width / 2
        y_rotated += new_height / 2

        # Calculate new bounding box coordinates
        new_xmin = max(0, int(x_rotated - (xmax - xmin) / 2))
        new_ymin = max(0, int(y_rotated - (ymax - ymin) / 2))
        new_xmax = min(new_width, int(x_rotated + (xmax - xmin) / 2))
        new_ymax = min(new_height, int(y_rotated + (ymax - ymin) / 2))

        rotated_annotations.append((new_xmin, new_ymin, new_xmax, new_ymax))

    return rotated_annotations

def save_rotated_annotation(rotated_annotations, txt_path):
    """Save rotated annotations in YOLO format."""
    new_width, new_height = image_size
    with open(txt_path, 'w') as txt_file:
        for (xmin, ymin, xmax, ymax) in rotated_annotations:
            class_id = 0  # All classes are labeled as 0 in this task
            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / new_width
            y_center = ((ymin + ymax) / 2) / new_height
            bbox_width = (xmax - xmin) / new_width
            bbox_height = (ymax - ymin) / new_height
            # Write to .txt file
            txt_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

def convert_annotation(xml_path, txt_path):
    """Convert XML annotation to YOLO format and save as .txt file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    with open(txt_path, 'w') as txt_file:
        for obj in root.findall('object'):
            class_id = 0  # All classes are labeled as 0 in this task
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            
            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            # Write to .txt file
            txt_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

# Collect all files
images = [f for f in os.listdir(input_dir) if f.endswith('.bmp')]
annotations = [f for f in os.listdir(input_dir) if f.endswith('.xml')]

# Pair images with their annotations
image_annotation_pairs = [(img, img.replace('.bmp', '.xml')) for img in images if img.replace('.bmp', '.xml') in annotations]

# Shuffle and split data
random.shuffle(image_annotation_pairs)
split_index = int(len(image_annotation_pairs) * train_ratio)
train_pairs = image_annotation_pairs[:split_index]
val_pairs = image_annotation_pairs[split_index:]

# Process train and val data
for pairs, folder in [(train_pairs, train_dir), (val_pairs, val_dir)]:
    for image_file, xml_file in pairs:
        image_path = os.path.join(input_dir, image_file)
        xml_path = os.path.join(input_dir, xml_file)

        output_image_path = os.path.join(folder, image_file)
        output_txt_path = os.path.join(folder, image_file.replace('.bmp', '.txt'))
        
        # Save original resized image and annotations
        resize_image(image_path, output_image_path)
        convert_annotation(xml_path, output_txt_path)
        
        # Rotate image by two different random angles and save
        for i in range(2):
            angle = random.uniform(-30, 30)
            rotated_image_path = os.path.join(folder, f"{image_file.replace('.bmp', '')}_rotated_{i}.bmp")
            rotate_and_save(image_path, rotated_image_path, angle)

            # Rotate annotation and save it
            rotated_annotations = rotate_annotation(xml_path, angle, image_size[0], image_size[1])
            rotated_txt_path = os.path.join(folder, f"{image_file.replace('.bmp', '')}_rotated_{i}.txt")
            save_rotated_annotation(rotated_annotations, rotated_txt_path)

print("Processing complete. Images and annotations are organized in 'data/train' and 'data/val'.")