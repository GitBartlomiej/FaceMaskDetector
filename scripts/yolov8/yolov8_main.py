import cv2
from ultralytics import YOLO
import os
import concurrent.futures
import csv

# Load the YOLO model
model = YOLO('runs/detect/train3/weights/best.pt')

# Base directory where the images are located
base_source_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/metamorphic_transformed_dataset'

# Base directory where the images with detections will be saved
base_output_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/metamorphic_test_detection'
os.makedirs(base_output_dir, exist_ok=True)

# Path for the log file
detection_log_path = os.path.join(base_output_dir, 'detection_log.csv')

# Prepare the CSV log file and write the header
with open(detection_log_path, 'w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Folder', 'Image', 'Detected'])

# Function to process images
def process_images(subdir):
    source_dir = os.path.join(base_source_dir, subdir)
    output_dir = os.path.join(base_output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Process all images in the subdirectory
    for image_name in os.listdir(source_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_dir, image_name)
            image = cv2.imread(image_path)
            results = model.predict(image)

            detected = 'No'
            # Draw bounding boxes and labels on the image if masks are detected
            if results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = results[0].names[box.cls[0].item()]
                    if label == 'Mask':  # Assuming 'mask' is a label for masks
                        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        image = cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        detected = 'Yes'

            # Save the image
            cv2.imwrite(os.path.join(output_dir, image_name), image)

            # Log the detection result
            with open(detection_log_path, 'a', newline='') as log_file:
                log_writer = csv.writer(log_file)
                log_writer.writerow([subdir, image_name, detected])

    print(f'Finished processing images in {subdir}')

# Execute the image processing in parallel for each subdirectory
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_images, subdirectories)

print('All images have been processed and saved with detections.')
