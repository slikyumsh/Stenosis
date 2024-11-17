import os
import random
import tifffile
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import glob
import cv2

DATA_YAML_PATH = 'C:/Users/edimv/Desktop/stenosis/data.yaml'
model_path = 'YOLO_Stenosis_Detection/YOLOv8m_training5/weights/best.pt'
model = YOLO(model_path)


def load_annotations(annotation_path, img_width, img_height):
    """
    Load annotations from a YOLO-formatted TXT file and convert to absolute pixel coordinates.
    """
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
            # Convert normalized coordinates to absolute pixel coordinates
            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            width = width_norm * img_width
            height = height_norm * img_height
            annotations.append([label, x_center, y_center, width, height])
    return annotations

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in format (x_center, y_center, width, height).
    """
    # Convert boxes to (x1, y1, x2, y2)
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    # Calculate intersection coordinates
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    # Compute intersection area
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute areas of boxes
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Compute IoU
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0  # Avoid division by zero
    iou = inter_area / union_area
    return iou

def test_iou(test_folder):
    """
    Test IoU on a set of images and annotations using YOLO model predictions.
    """
    iou_results = []

    image_paths = glob.glob(os.path.join(test_folder, "*.bmp"))

    for img_path in image_paths:
        annotation_path = img_path.replace(".bmp", ".txt")
        
        if not os.path.exists(annotation_path):
            print(f"Annotation missing for {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}")
            continue
        img_height, img_width = img.shape[:2]

        results = model(img_path)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"No predictions for {img_path}")
            continue 

        predictions = results[0].boxes.xywh.cpu().numpy()  

        gt_boxes = load_annotations(annotation_path, img_width, img_height)

        for pred_box in predictions:
            max_iou = 0
            for gt_box in gt_boxes:
                gt_box_coords = gt_box[1:] 
                iou = calculate_iou(pred_box[0:4], gt_box_coords)
                max_iou = max(max_iou, iou)
            iou_results.append(max_iou)

    return iou_results


if __name__ == '__main__':
    TEST_FOLDER = "data/test"
    iou_results = test_iou(TEST_FOLDER)
    if iou_results:
        average_iou = sum(iou_results) / len(iou_results)
        print("Average IoU on test set:", average_iou)
    else:
        print("Failed to calculate IoU - no predictions for the selected images.")