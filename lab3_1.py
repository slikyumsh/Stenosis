import os
import glob
import cv2
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO

DEVICE = 'cuda'  # Change to 'cpu' if you want to run on CPU

DATA_YAML_PATH = 'C:/Users/edimv/Desktop/stenosis/data.yaml'
model_path = 'YOLO_Stenosis_Detection/YOLOv8m_training5/weights/best.pt'
model = YOLO(model_path)
model.to(DEVICE) 

# Export the model to ONNX format
model.export(format="onnx", imgsz = 800)  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO("best.onnx")

# Run inference
results = onnx_model("data/test/14_002_5_0017.bmp", imgsz = 800, device = DEVICE)