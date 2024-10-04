import xml.etree.ElementTree as ET
import os 

# Convert XML annotations to YOLO format
def convert_to_yolo_format(xml_file, img_width=800, img_height=800):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Extract bounding box coordinates
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # YOLO format: class x_center y_center width height (all normalized)
        x_center = (xmin + xmax) / (2 * img_width)
        y_center = (ymin + ymax) / (2 * img_height)
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height
        yolo_annotation = f"0 {x_center} {y_center} {box_width} {box_height}\n"
        
        # Write annotation to a .txt file with the same name as the image
        txt_filename = xml_file.replace('.xml', '.txt')
        with open(txt_filename, 'w') as f:
            f.write(yolo_annotation)

# Process all XML files in train and val directories
def process_annotations(data_dir):
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                convert_to_yolo_format(xml_path)

process_annotations('data/train')
process_annotations('data/val')
