import os
import random
import xml.etree.ElementTree as ET
import math
import numpy as np
import cv2  # OpenCV for image processing

try:
    # Define paths
    input_dir = 'C:/Users/edimv/Desktop/stenosis/dataset'
    output_dir = 'C:/Users/edimv/Desktop/stenosis/data'
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1  # Alternatively, test_ratio = 1 - train_ratio - val_ratio

    image_size = (800, 800)

    # Create output directories
    train_dir = output_dir + '/train'
    val_dir = output_dir + '/val'
    test_dir = output_dir + '/test'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Output directories created at: {output_dir}")

    def preprocess_image(img):
        """Apply preprocessing steps to an image."""
        # Standardization
        mean = np.mean(img)
        std = np.std(img)
        img_standardized = (img - mean) / std

        # Rescale to 0-255
        img_standardized = np.clip(
            ((img_standardized - img_standardized.min()) / (img_standardized.max() - img_standardized.min())) * 255,
            0, 255).astype(np.uint8)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_standardized)

        # Gamma adjustment
        gamma = 0.8  # Adjust gamma value as needed
        invGamma = 1.0 / gamma
        img_gamma = np.power(img_clahe / 255.0, invGamma) * 255.0
        img_gamma = np.clip(img_gamma, 0, 255).astype(np.uint8)

        return img_gamma

    def resize_and_preprocess_image(image_path, output_path):
        """Resize image to the specified size, apply preprocessing, and save it."""
        # Read the image in gray-scale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return

        # Resize the image
        img_resized = cv2.resize(img, image_size)

        # Apply preprocessing
        img_preprocessed = preprocess_image(img_resized)

        # Save the preprocessed image
        cv2.imwrite(output_path, img_preprocessed)

    def rotate_and_save(image_path, output_path, angle):
        """Rotate image by a given angle, apply preprocessing, and save it."""
        # Read the image in gray-scale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {image_path}")
            return

        # Get image size
        (h, w) = img.shape[:2]

        # Compute the center of the image
        center = (w // 2, h // 2)

        # Rotate the image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Resize the image
        img_resized = cv2.resize(img_rotated, image_size)

        # Apply preprocessing
        img_preprocessed = preprocess_image(img_resized)

        # Save the preprocessed image
        cv2.imwrite(output_path, img_preprocessed)

    def rotate_annotation(xml_path, angle, original_width, original_height):
        """Rotate XML annotation by a given angle."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Failed to parse XML file: {xml_path}, Error: {e}")
            return []

        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        # The image will be resized to new_width and new_height
        new_width, new_height = image_size
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        rotated_annotations = []

        for obj in root.findall('object'):
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # Adjust bounding box coordinates according to the new image size
            xmin = xmin * scale_x
            ymin = ymin * scale_y
            xmax = xmax * scale_x
            ymax = ymax * scale_y

            # Calculate the center of the bounding box
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # Translate the bounding box center to the origin
            x_center -= new_width / 2
            y_center -= new_height / 2

            # Apply rotation
            x_rotated = x_center * cos_angle - y_center * sin_angle
            y_rotated = x_center * sin_angle + y_center * cos_angle

            # Translate back to the new image center
            x_rotated += new_width / 2
            y_rotated += new_height / 2

            # Since the box dimensions remain the same, but their position changes
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            # Calculate new bounding box coordinates
            new_xmin = max(0, int(x_rotated - bbox_width / 2))
            new_ymin = max(0, int(y_rotated - bbox_height / 2))
            new_xmax = min(new_width, int(x_rotated + bbox_width / 2))
            new_ymax = min(new_height, int(y_rotated + bbox_height / 2))

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
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Failed to parse XML file: {xml_path}, Error: {e}")
            return

        original_width = int(root.find('size/width').text)
        original_height = int(root.find('size/height').text)

        new_width, new_height = image_size

        with open(txt_path, 'w') as txt_file:
            for obj in root.findall('object'):
                class_id = 0  # All classes are labeled as 0 in this task
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)

                # Adjust bounding box coordinates according to the new image size
                xmin = xmin * new_width / original_width
                ymin = ymin * new_height / original_height
                xmax = xmax * new_width / original_width
                ymax = ymax * new_height / original_height

                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / new_width
                y_center = ((ymin + ymax) / 2) / new_height
                bbox_width = (xmax - xmin) / new_width
                bbox_height = (ymax - ymin) / new_height

                # Write to .txt file
                txt_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

    # Collect all files
    images = [f for f in os.listdir(input_dir) if f.endswith('.bmp')]
    annotations = [f for f in os.listdir(input_dir) if f.endswith('.xml')]

    print(f"Found {len(images)} images and {len(annotations)} annotations in '{input_dir}'.")

    # Pair images with their annotations
    image_annotation_pairs = [
        (img, img.replace('.bmp', '.xml'))
        for img in images if img.replace('.bmp', '.xml') in annotations
    ]

    print(f"Total image-annotation pairs: {len(image_annotation_pairs)}")

    # Check if any pairs are found
    if not image_annotation_pairs:
        print("No image-annotation pairs found. Please check your input directory and file extensions.")
        exit()

    # Shuffle data
    random.shuffle(image_annotation_pairs)

    # Split data into train, val, test
    total_pairs = len(image_annotation_pairs)
    train_end = int(total_pairs * train_ratio)
    val_end = train_end + int(total_pairs * val_ratio)

    train_pairs = image_annotation_pairs[:train_end]
    val_pairs = image_annotation_pairs[train_end:val_end]
    test_pairs = image_annotation_pairs[val_end:]

    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    # Process train, val, and test data
    for pairs, folder in [(train_pairs, train_dir), (val_pairs, val_dir), (test_pairs, test_dir)]:
        print(f"Processing folder: {folder}")
        for image_file, xml_file in pairs:
            image_path = os.path.join(input_dir, image_file)
            xml_path = os.path.join(input_dir, xml_file)

            output_image_path = os.path.join(folder, image_file)
            output_txt_path = os.path.join(folder, image_file.replace('.bmp', '.txt'))

            # Save original resized image and annotations
            resize_and_preprocess_image(image_path, output_image_path)
            convert_annotation(xml_path, output_txt_path)

            # For train and val data, perform augmentation
            if folder != test_dir:
                # Rotate image by two different random angles and save
                for i in range(2):
                    angle = random.uniform(-20, 20)
                    rotated_image_path = os.path.join(folder, f"{image_file.replace('.bmp', '')}_rotated_{i}.bmp")
                    rotate_and_save(image_path, rotated_image_path, angle)

                    # Rotate annotation and save it
                    original_width, original_height = image_size  # Since images are resized before rotation
                    rotated_annotations = rotate_annotation(xml_path, angle, original_width, original_height)
                    rotated_txt_path = os.path.join(folder, f"{image_file.replace('.bmp', '')}_rotated_{i}.txt")
                    save_rotated_annotation(rotated_annotations, rotated_txt_path)

    print("Processing complete. Images and annotations are organized in 'data/train', 'data/val', and 'data/test'.")

except Exception as e:
    print(f"An error occurred: {e}")
