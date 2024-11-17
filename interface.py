import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# Загрузка модели YOLO
model_path1 = 'YOLO_Stenosis_Detection/YOLOv8m_training5/weights/best.pt'
model = YOLO(model_path1)

# Путь к папке с видео
video_folder = 'C:/Users/edimv/Desktop/video_test'

# Поиск всех видео с #input
input_videos = [f for f in Path(video_folder).glob('*#input*.avi')]

for video_path in input_videos:
    
    cap = cv2.VideoCapture(str(video_path))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Определяем путь для сохранения обработанного видео
    output_path = str(video_path).replace('#input', '#yolo')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

       
        results = model(frame)
        
        # Проходим по результатам детекций и рисуем bounding boxes
        annotated_frame = frame.copy()
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # координаты бокса
            confidence = result.conf[0]  
            label = f"stenosis {confidence:.2f}"

            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

           
            font_scale = 0.25 
            font_thickness = 1  
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)  
            cv2.putText(annotated_frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)  

        
        out.write(annotated_frame)

    # Освобождаем ресурсы
    cap.release()
    out.release()

print("Обработка завершена.")
