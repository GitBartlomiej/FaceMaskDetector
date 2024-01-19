import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load your YOLO model with custom or pre-trained weights
model = YOLO('./runs/detect/train/weights/best.pt')

# Directory where the images are located for annotation
image_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/with_mask_small/'

# Base directory where the new dataset will be created
base_dataset_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/yolov8_dataset2/'
train_labels_dir = os.path.join(base_dataset_dir, 'train', 'labels')


# Create subdirectories for train, valid, test, their images, and labels
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(base_dataset_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dataset_dir, split, 'labels'), exist_ok=True)

# Get list of images
images = os.listdir(image_dir)

for image_name in images:
    # Skip if not an image
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Full path to the image
    image_path = os.path.join(image_dir, image_name)
    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Run prediction
    results = model.predict(image)

    # Path for saving the annotated image
    annotated_image_path = os.path.join(base_dataset_dir, 'train', 'images', image_name)

    # Draw bounding boxes and labels on the image
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = results[0].names[box.cls[0].item()]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save the annotated image
    cv2.imwrite(annotated_image_path, image)

    # Annotation file path
    # Open a file to write annotations in the train/labels directory
    annotation_file_name = os.path.splitext(image_name)[0] + '.txt'
    annotation_file_path = os.path.join(train_labels_dir, annotation_file_name)

    # Open a file to write annotations in the train/labels directory
    annotation_file_name = os.path.splitext(image_name)[0] + '.txt'
    with open(os.path.join(train_labels_dir, annotation_file_name), 'w') as f:
        for box in results[0].boxes:
            class_id = box.cls[0].item()
            # Get the bounding box coordinates in pixel values
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

            # Calculate the bounding box center, width, and height in pixel values
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            # Normalize the coordinates
            x_center_n = round(x_center / image_width, 8)
            y_center_n = round(y_center / image_height, 8)
            width_n = round(width / image_width, 8)
            height_n = round(height / image_height, 8)
            f.write(f"{class_id} {x_center_n} {y_center_n} {width_n} {height_n}\n")

print('Annotations and images with detections have been saved.')
