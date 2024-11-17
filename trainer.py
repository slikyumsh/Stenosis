import os
from ultralytics import YOLO
import torch
from clearml import Task
import os
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

    
os.environ['CLEARML_API_ACCESS_KEY'] = os.environ.get("CLEARML_API_ACCESS_KEY")
os.environ['CLEARML_API_SECRET_KEY'] = os.environ.get("CLEARML_API_SECRET_KEY")
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'      
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'       
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'   


def main():
    task = Task.init(project_name='YOLO_Stenosis_Detection', task_name='Training')

    DATA_YAML_PATH = 'C:/Users/edimv/Desktop/stenosis/data.yaml'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = YOLO('yolov8m.pt')

    model.train(
        data=DATA_YAML_PATH,
        epochs=70,
        imgsz=800, 
        workers=12,
        batch = 16,
        amp=True,
        device=device,
        auto_augment = None,
        project='YOLO_Stenosis_Detection',
        name='YOLOv8m_training'
    )

    val_results = model.val(
        data=DATA_YAML_PATH,
        split='val',
        device=device
    )

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

    test_results = model.val(
        data=DATA_YAML_PATH,
        split='test',
        device=device
    )

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
   

    print('Training and evaluation complete.')

if __name__ == '__main__':
    main()