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

DATA_YAML_PATH = 'C:/Users/edimv/Desktop/stenosis/data.yaml'
OUTPUT_DIR = 'C:/Users/edimv/Desktop/stenosis/data'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')

model_path1 = 'yolov8n.pt'
model1 = YOLO(model_path1)

model1.train(data=DATA_YAML_PATH, epochs=35, imgsz=(800, 800), workers = 8, amp = True)