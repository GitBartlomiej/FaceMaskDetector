import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load your YOLO model with custom or pre-trained weights
model = YOLO('runs/detect/train/weights/best.pt')

image_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/with_mask_small/'
base_dataset_dir = './dataset'

# Create subdirectories for train, valid, test, their images, and labels
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(base_dataset_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dataset_dir, split, 'labels'), exist_ok=True)

# Create a directory for not_detected images
not_detected_dir = os.path.join(base_dataset_dir, 'not_detected', 'not_detected_images')
os.makedirs(not_detected_dir, exist_ok=True)

images = os.listdir(image_dir)

for image_name in images:
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    results = model.predict(image)

    # If no masks are detected, save image to not_detected and continue to the next image
    if len(results[0].boxes) == 0:
        not_detected_image_path = os.path.join(not_detected_dir, image_name)
        cv2.imwrite(not_detected_image_path, image)
        continue

    # Process images with detected masks
    annotated_image_path = os.path.join(base_dataset_dir, 'train', 'images', image_name)
    annotation_file_path = os.path.join(base_dataset_dir, 'train', 'labels', os.path.splitext(image_name)[0] + '.txt')

    with open(annotation_file_path, 'w') as f:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            class_id = box.cls[0].item()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            x_center_n = round(x_center / image_width, 8)
            y_center_n = round(y_center / image_height, 8)
            width_n = round(width / image_width, 8)
            height_n = round(height / image_height, 8)
            f.write(f"{class_id} {x_center_n} {y_center_n} {width_n} {height_n}\n")

        # Draw bounding boxes on the image
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = results[0].names[box.cls[0].item()]
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imwrite(annotated_image_path, image)

print('Annotations and images with detections have been saved.')
