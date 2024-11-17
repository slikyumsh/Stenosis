import cv2
from ultralytics import YOLO

# Пути к файламdata\test\14_002_5_0017.txt
image_path = 'data/test/14_002_5_0017.bmp'
true_boxes_file = 'data/test/14_002_5_0017.txt'  # Истинные координаты (из тренировочных данных)
model_path = 'YOLO_Stenosis_Detection/YOLOv8m_training5/weights/best.pt'  # Модель YOLO

# Загрузка изображения
image = cv2.imread(image_path)

# Загрузка модели YOLO
model = YOLO(model_path)

# Функция для чтения координат боксов из файла
def read_boxes_from_txt(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.split()
            x_center, y_center, width, height = map(float, data[1:5])
            # Преобразование относительных координат (x_center, y_center, width, height) в абсолютные (x1, y1, x2, y2)
            img_width, img_height = image.shape[1], image.shape[0]
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            boxes.append((x1, y1, x2, y2))
    return boxes

# Чтение истинных боксов
true_boxes = read_boxes_from_txt(true_boxes_file)

# Функция для отрисовки боксов
def draw_boxes(image, boxes, color, label):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Отрисовка истинных боксов (например, красным)
draw_boxes(image, true_boxes, (0, 0, 255), "True")

# Используем модель YOLO для предсказания боксов
results = model(image)

# Преобразуем предсказанные боксы в формат (x1, y1, x2, y2)
predicted_boxes = []
for result in results[0].boxes:
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Преобразование координат бокса
    predicted_boxes.append((x1, y1, x2, y2))

# Отрисовка предсказанных боксов (например, зеленым)
draw_boxes(image, predicted_boxes, (0, 255, 0), "Predicted")

# Вместо cv2.imshow(), сохраняем изображение
output_image_path = 'image_with_bboxes10.jpg'
cv2.imwrite(output_image_path, image)
print(f"Изображение сохранено как {output_image_path}")

