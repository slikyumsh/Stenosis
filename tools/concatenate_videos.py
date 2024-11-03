from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
video_folder = 'C:/Users/edimv/Desktop/video_test'
# Пути к видеофайлам
videos = [
    ("14-008 (1)#input.avi", "Input"),
    ("14-008 (1)#ssd_resnet50_v1_fpn.avi", "SSD ResNet50 v1 FPN"),
    ("14-008 (1)#rfcn_resnet101_coco.avi", "RFCN ResNet101 COCO"),
    ("14-008 (1)#yolo.avi", "YOLO")
]


# Путь для сохранения результата
output_file = "output_video.mp4"

# Список для хранения обработанных видео
clips = []

# Проходим по каждому видеофайлу
for video_path, text in videos:
    # Загружаем видеофайл
    clip = VideoFileClip(video_folder + "/" + video_path)
    
    # Создаем текстовый клип для добавления в видео
    txt_clip = TextClip(text, fontsize=24, color='white')
    txt_clip = txt_clip.set_position(("left", "top")).set_duration(clip.duration)
    
    # Композиция видео с текстом
    video = CompositeVideoClip([clip, txt_clip])
    
    clips.append(video)

# Склеиваем видео
final_clip = CompositeVideoClip(clips, method="compose")

# Сохраняем финальное видео в формате MP4
final_clip.write_videofile(output_file, codec='libx264')
