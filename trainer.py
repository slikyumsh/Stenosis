import os
from ultralytics import YOLO
import torch
from clearml import Task
import os

# Set ClearML configuration programmatically
os.environ['CLEARML_API_ACCESS_KEY'] = "HOJXSS6CYC2U1TXWOESFB3XT4624E9"
os.environ['CLEARML_API_SECRET_KEY'] = "BqTgnG9SeBkbQvcKukhuZXmUMuo4Dp7G-sJ_jtzoBI7EEFQDxcsgGNER8rHguU9_OwQ"
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'      # Replace with your ClearML API server URL if different
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'       # Replace with your ClearML Web server URL if different
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'   # Replace with your ClearML Files server URL if different


def main():
    # Initialize ClearML task
    task = Task.init(project_name='YOLO_Stenosis_Detection', task_name='Training')

    # Paths
    DATA_YAML_PATH = 'C:/Users/edimv/Desktop/stenosis/data.yaml'
    OUTPUT_DIR = 'C:/Users/edimv/Desktop/stenosis/runs'

    # Choose device (CPU or GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Initialize the model
    model = YOLO('yolov8n.pt')  # Use 'yolov8n.yaml' for training from scratch

    # Train the model
    model.train(
        data=DATA_YAML_PATH,
        epochs=35,
        imgsz=800,  # You can specify (height, width) tuple if needed
        workers=8,
        amp=True,
        device=device,
        project='YOLO_Stenosis_Detection',
        name='YOLOv8n_training'
    )

    # Evaluate the model on the validation set
    val_results = model.val(
        data=DATA_YAML_PATH,
        split='val',
        device=device
    )

    # Log validation metrics to ClearML
    task.get_logger().report_scalar(
        title='Validation mAP50',
        series='mAP50',
        value=val_results.box.map50,
        iteration=0
    )
    task.get_logger().report_scalar(
        title='Validation mAP50-95',
        series='mAP50-95',
        value=val_results.box.map,
        iteration=0
    )
    # Log Validation IoU
    if hasattr(val_results.box, 'iou'):
        val_iou = val_results.box.iou
        task.get_logger().report_scalar(
            title='Validation IoU',
            series='IoU',
            value=val_iou,
            iteration=0
        )
    else:
        print("IoU metric not found in val_results.")

    # Evaluate the model on the test set
    test_results = model.val(
        data=DATA_YAML_PATH,
        split='test',
        device=device
    )

    # Log test metrics to ClearML
    task.get_logger().report_scalar(
        title='Test mAP50',
        series='mAP50',
        value=test_results.box.map50,
        iteration=0
    )
    task.get_logger().report_scalar(
        title='Test mAP50-95',
        series='mAP50-95',
        value=test_results.box.map,
        iteration=0
    )
    # Log Test IoU
    if hasattr(test_results.box, 'iou'):
        test_iou = test_results.box.iou
        task.get_logger().report_scalar(
            title='Test IoU',
            series='IoU',
            value=test_iou,
            iteration=0
        )
    else:
        print("IoU metric not found in test_results.")

    print('Training and evaluation complete.')

if __name__ == '__main__':
    main()