import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load your YOLO model with custom or pre-trained weights
model = YOLO('./runs/detect/train/weights/best.pt')

# Directory where the images are located for annotation
image_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/with_mask/'

# Base directory where the new dataset will be created
base_dataset_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/yolov8_dataset/'

# Create subdirectories for train, valid, test, and their images and labels
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(base_dataset_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dataset_dir, split, 'labels'), exist_ok=True)

# Assume all images for annotation are for training, change as necessary
train_images_dir = os.path.join(base_dataset_dir, 'train', 'images')
train_labels_dir = os.path.join(base_dataset_dir, 'train', 'labels')

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
    # Copy image to the train/images directory
    cv2.imwrite(os.path.join(train_images_dir, image_name), image)

    # Run prediction
    results = model.predict(image)

    # Open a file to write annotations in the train/labels directory
    annotation_file_name = image_name.rsplit('.', 1)[0] + '.txt'
    with open(os.path.join(train_labels_dir, annotation_file_name), 'w') as f:
        for box in results[0].boxes:
            # Get class ID
            class_id = box.cls[0].item()
            # Move the box tensor to CPU and convert to numpy array
            box_data = box.xywh[0].cpu().numpy()
            # Get bounding box coordinates, normalized by image width and height
            x_center, y_center, width, height = (box_data / np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).tolist()
            # Write to file
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print('Annotations and images copied to the new dataset structure.')
