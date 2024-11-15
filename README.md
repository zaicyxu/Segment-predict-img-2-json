# Segment-predict-img-2-json
This script is used to view and modify the result graph measured by the YOLO model graph using labelme.

# Background

This script is used to view and modify the result graph measured by the YOLO model graph using labelme.

# Coding

## When there are only pictures for predict results

```python
import cv2
import os
import json

def extract_boxes_from_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = (0, 50, 50)
    upper_red = (10, 255, 255)
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x+w, y+h))

    return boxes

def image_to_labelme_json(image_path, output_path, class_name):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    boxes = extract_boxes_from_image(image)

    shapes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        shape = {
            "label": class_name,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    json_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

def process_directory(image_folder, output_folder, class_name):
    os.makedirs(output_folder, exist_ok=True)
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.bmp')):
            image_path = os.path.join(image_folder, image_file)
            output_path = os.path.join(output_folder, image_file.replace('.png', '.json').replace('.jpg', '.json').replace('.bmp', '.json'))
            image_to_labelme_json(image_path, output_path, class_name)

if __name__ == "__main__":
    image_folder = r"D:\work\DATA\Analysisdata\predicted_images"
    output_folder = r"D:\work\DATA\Analysisdata\labelme_json"
    class_name = "1"  

    process_directory(image_folder, output_folder, class_name)
```

## When there are both txt and pic.

```python
import os
import json

def yolo_to_labelme(yolo_result_folder, img_folder, output_folder, class_names):
    # 遍历文件夹中的predict结果文件
    for yolo_file in os.listdir(yolo_result_folder):
        if yolo_file.endswith('.txt'):
            yolo_path = os.path.join(yolo_result_folder, yolo_file)
            
            # 获取对应的图像文件
            img_file = yolo_file.replace('.txt', '.jpg')  
            img_path = os.path.join(img_folder, img_file)
            if not os.path.exists(img_path):
                img_file = yolo_file.replace('.txt', '.png')  
                img_path = os.path.join(img_folder, img_file)
            if not os.path.exists(img_path):
                continue

            import cv2
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            shapes = []
            with open(yolo_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height
                    confidence = float(parts[5])
                    
                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2
                    x_max = x_center + bbox_width / 2
                    y_max = y_center + bbox_height / 2

                    points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                    shape = {
                        "label": class_names[class_id],
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    shapes.append(shape)

            json_data = {
                "version": "4.5.6",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_file,
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width
            }

            output_path = os.path.join(output_folder, yolo_file.replace('.txt', '.json'))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    yolo_result_folder = r"D:\work\DATA\Analysisdata\yolo_results"
    img_folder = r"D:\work\DATA\Analysisdata\images"
    output_folder = r"D:\work\DATA\Analysisdata\labelme_json"
    class_names = ["0", "1", "2"] 

    os.makedirs(output_folder, exist_ok=True)
    yolo_to_labelme(yolo_result_folder, img_folder, output_folder, class_names)
```
