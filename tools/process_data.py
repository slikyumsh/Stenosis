import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np

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

def preprocess_image(image_path, output_path):
    """Apply preprocessing steps to the image and save it."""
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Resize the image
    img = cv2.resize(img, image_size)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise filtering (median filter)
    gray = cv2.medianBlur(gray, 3)
    
    # Contrast enhancement (histogram equalization)
    gray = cv2.equalizeHist(gray)
    
    # Save the processed image
    cv2.imwrite(output_path, gray)

def convert_annotation(xml_path, txt_path):
    """Convert XML annotation to YOLO format and save as .txt file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Original image dimensions from XML
    orig_width = int(root.find('size/width').text)
    orig_height = int(root.find('size/height').text)
    
    # New dimensions after resizing
    width = image_size[0]
    height = image_size[1]
    
    # Scaling factors
    x_scale = width / orig_width
    y_scale = height / orig_height
    
    with open(txt_path, 'w') as txt_file:
        for obj in root.findall('object'):
            class_id = 0  # Assuming all classes are labeled as 0
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            
            # Scale bounding boxes
            xmin = int(xmin * x_scale)
            xmax = int(xmax * x_scale)
            ymin = int(ymin * y_scale)
            ymax = int(ymax * y_scale)
            
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
        
        preprocess_image(image_path, output_image_path)
        convert_annotation(xml_path, output_txt_path)

print("Processing complete. Images and annotations are organized in 'data/train' and 'data/val'.")
