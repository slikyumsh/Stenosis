import os
import glob
import cv2
import time
import numpy as np
from ultralytics import YOLO

# Set the device for inference ('cpu' or 'cuda')
DEVICE = 'cuda'  # Change to 'cpu' if you want to run on CPU

DATA_YAML_PATH = 'C:/Users/edimv/Desktop/stenosis/data.yaml'
model_path = 'YOLO_Stenosis_Detection/YOLOv8m_training5/weights/best.pt'
model = YOLO(model_path)
model.to(DEVICE)  # Set the device

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

def draw_boxes(img, pred_boxes, gt_boxes):
    """
    Draw predicted and ground truth boxes on the image.
    Predicted boxes in green, ground truth boxes in red.
    """
    # Draw predicted boxes
    for pred in pred_boxes:
        box = pred['box']
        label = pred['label']
        confidence = pred['conf']
        # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        x_center, y_center, width, height = box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color for predictions
        # Put label and confidence
        cv2.putText(img, f'Pred {label}:{confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw ground truth boxes
    for gt in gt_boxes:
        label = gt[0]
        box = gt[1:]  # [x_center, y_center, width, height]
        x_center, y_center, width, height = box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for ground truths
        # Put label
        cv2.putText(img, f'GT {label}', (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return img

def test_iou(test_folder):
    """
    Test IoU on a set of images and annotations using YOLO model predictions.
    Also calculates average times, metrics, and saves samples.
    """
    iou_results = []
    preprocessing_times = []
    processing_times = []
    postprocessing_times = []
    per_image_data = []  # To store data per image for analysis
    low_iou_count = 0
    total_TP = 0
    total_FP = 0
    total_FN = 0

    image_paths = glob.glob(os.path.join(test_folder, "*.bmp"))
    if not os.path.exists('samples1'):
        os.makedirs('samples1')

    for img_path in image_paths:
        # Preprocessing
        t1 = time.time()
        annotation_path = img_path.replace(".bmp", ".txt")
        img_filename = os.path.basename(img_path)
        
        if not os.path.exists(annotation_path):
            print(f"Annotation missing for {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}")
            continue
        img_height, img_width = img.shape[:2]

        gt_boxes = load_annotations(annotation_path, img_width, img_height)
        t2 = time.time()
        preprocessing_times.append(t2 - t1)

        # Processing
        t3 = time.time()
        results = model(img, device=DEVICE)
        t4 = time.time()
        processing_times.append(t4 - t3)

        # Post-processing
        t5 = time.time()
        if results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"No predictions for {img_path}")
            iou_results.append(0)
            per_image_data.append({
                'img_path': img_path,
                'max_iou': 0,
                'ious': [0],
                'TP': 0,
                'FP': 0,
                'FN': len(gt_boxes),
                'pred_boxes': [],
                'gt_boxes': gt_boxes
            })
            total_FN += len(gt_boxes)
            if 0 < 0.3:
                low_iou_count += 1
            t6 = time.time()
            postprocessing_times.append(t6 - t5)
            continue

        # Get predicted boxes and confidences
        predictions = results[0].boxes.xywh.cpu().numpy()  # [x_center, y_center, width, height]
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        labels = results[0].boxes.cls.cpu().numpy()  # Class labels

        pred_boxes = []
        for i in range(len(predictions)):
            pred_box = predictions[i]
            confidence = confidences[i]
            label = int(labels[i])
            pred_boxes.append({'box': pred_box, 'conf': confidence, 'label': label})

        # Sort predictions by confidence
        pred_boxes.sort(key=lambda x: x['conf'], reverse=True)

        matched_gt = []
        ious = []
        TP = 0
        FP = 0
        FN = 0
        for pred in pred_boxes:
            pred_box = pred['box']
            pred_conf = pred['conf']
            pred_label = pred['label']
            best_iou = 0
            best_gt_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue  # Already matched
                gt_label = gt_box[0]
                gt_box_coords = gt_box[1:]
                iou = calculate_iou(pred_box, gt_box_coords)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= 0.5:
                TP += 1
                total_TP += 1
                matched_gt.append(best_gt_idx)
            else:
                FP += 1
                total_FP += 1
            ious.append(best_iou)

        FN = len(gt_boxes) - len(matched_gt)
        total_FN += FN

        max_iou = max(ious) if ious else 0
        if max_iou < 0.3:
            low_iou_count += 1

        iou_results.append(max_iou)
        per_image_data.append({
            'img_path': img_path,
            'max_iou': max_iou,
            'ious': ious,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'pred_boxes': pred_boxes,
            'gt_boxes': gt_boxes
        })
        t6 = time.time()
        postprocessing_times.append(t6 - t5)

    # Compute average times
    avg_preprocessing_time = sum(preprocessing_times) / len(preprocessing_times)
    avg_processing_time = sum(processing_times) / len(processing_times)
    avg_postprocessing_time = sum(postprocessing_times) / len(postprocessing_times)

    # Compute metrics
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

    # mAP calculation (simplified for mAP50 and mAP50-95)
    # For mAP50, IoU threshold = 0.5
    # For mAP50-95, IoU thresholds from 0.5 to 0.95 with step 0.05

    # Collect all predictions and ground truths
    all_detections = []
    all_annotations = []

    for data in per_image_data:
        preds = data['pred_boxes']
        gts = data['gt_boxes']
        # Append predictions
        image_detections = []
        for pred in preds:
            image_detections.append([pred['label'], pred['conf'], *pred['box']])
        all_detections.append(image_detections)
        # Append ground truths
        image_annotations = []
        for gt in gts:
            image_annotations.append([gt[0], *gt[1:]])
        all_annotations.append(image_annotations)

    # Compute AP for IoU thresholds
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    APs = []
    for iou_thresh in iou_thresholds:
        ap = calculate_ap(all_detections, all_annotations, iou_thresh)
        APs.append(ap)
    mAP50 = APs[0]
    mAP50_95 = np.mean(APs)

    # Save best and worst examples with drawn boxes
    per_image_data.sort(key=lambda x: x['max_iou'], reverse=True)
    best_samples = per_image_data[:3]
    worst_samples = per_image_data[-3:]

    for sample in best_samples:
        img = cv2.imread(sample['img_path'])
        pred_boxes = sample['pred_boxes']
        gt_boxes = sample['gt_boxes']
        img_with_boxes = draw_boxes(img, pred_boxes, gt_boxes)
        img_name = os.path.basename(sample['img_path'])
        save_path = os.path.join('samples1', f'good_{img_name}')
        cv2.imwrite(save_path, img_with_boxes)

    for sample in worst_samples:
        img = cv2.imread(sample['img_path'])
        pred_boxes = sample['pred_boxes']
        gt_boxes = sample['gt_boxes']
        img_with_boxes = draw_boxes(img, pred_boxes, gt_boxes)
        img_name = os.path.basename(sample['img_path'])
        save_path = os.path.join('samples1', f'bad_{img_name}')
        cv2.imwrite(save_path, img_with_boxes)

    # Return results
    results = {
        'average_iou': sum(iou_results) / len(iou_results) if iou_results else 0,
        'avg_preprocessing_time': avg_preprocessing_time,
        'avg_processing_time': avg_processing_time,
        'avg_postprocessing_time': avg_postprocessing_time,
        'precision': precision,
        'recall': recall,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'low_iou_count': low_iou_count
    }
    return results

def calculate_ap(all_detections, all_annotations, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) at a specific IoU threshold.
    """
    # Flatten all detections and annotations
    detections = []
    annotations = []
    for i in range(len(all_detections)):
        detections.extend([[i, *det] for det in all_detections[i]])
        annotations.extend([[i, *ann] for ann in all_annotations[i]])

    if len(annotations) == 0:
        return 0

    # Sort detections by confidence
    detections.sort(key=lambda x: x[2], reverse=True)

    TP = np.zeros(len(detections))
    FP = np.zeros(len(detections))

    detected_annotations = []

    for d_idx, detection in enumerate(detections):
        image_idx = detection[0]
        detection_label = detection[1]
        detection_conf = detection[2]
        detection_box = detection[3:]

        gt_annotations = [ann for ann in annotations if ann[0] == image_idx and ann[1] == detection_label]
        max_iou = 0
        matched_ann_idx = -1
        for ann_idx, ann in enumerate(gt_annotations):
            ann_box = ann[2:]
            iou = calculate_iou(detection_box, ann_box)
            if iou > max_iou:
                max_iou = iou
                matched_ann_idx = ann_idx
        if max_iou >= iou_threshold and (image_idx, matched_ann_idx) not in detected_annotations:
            TP[d_idx] = 1
            detected_annotations.append((image_idx, matched_ann_idx))
        else:
            FP[d_idx] = 1

    cumulative_TP = np.cumsum(TP)
    cumulative_FP = np.cumsum(FP)
    recalls = cumulative_TP / len(annotations)
    precisions = cumulative_TP / (cumulative_TP + cumulative_FP)

    # Avoid division by zero
    recalls = np.concatenate(([0], recalls))
    precisions = np.concatenate(([1], precisions))

    # Compute AP
    AP = 0
    for i in range(1, len(recalls)):
        AP += (recalls[i] - recalls[i - 1]) * precisions[i]
    return AP

if __name__ == '__main__':
    TEST_FOLDER = "data/test"
    results = test_iou(TEST_FOLDER)
    print("Average IoU on test set:", results['average_iou'])
    print("Average preprocessing time per image:", results['avg_preprocessing_time'])
    print("Average processing time per image:", results['avg_processing_time'])
    print("Average post-processing time per image:", results['avg_postprocessing_time'])
    print("Precision:", results['precision'])
    print("Recall:", results['recall'])
    print("mAP@0.5:", results['mAP50'])
    print("mAP@0.5:0.95:", results['mAP50_95'])
    print("Number of examples with IoU below 0.3:", results['low_iou_count'])
