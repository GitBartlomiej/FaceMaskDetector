import cv2
from ultralytics import YOLO
import os
import concurrent.futures

# Load the YOLO model
model = YOLO('runs/detect/train3/weights/best.pt')

# Base directory where the images are located
base_source_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/metamorphic_transformed_dataset'

# Base directory where the images with detections will be saved
base_output_dir = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/metamorphic_test_detection'
os.makedirs(base_output_dir, exist_ok=True)

# Get all subdirectories in the base directory
subdirectories = [d for d in os.listdir(base_source_dir) if os.path.isdir(os.path.join(base_source_dir, d))]

# Create corresponding subdirectories in the output directory
for subdir in subdirectories:
    os.makedirs(os.path.join(base_output_dir, subdir), exist_ok=True)


def process_images(subdir):
    source_dir = os.path.join(base_source_dir, subdir)
    output_dir = os.path.join(base_output_dir, subdir)

    # Process all images in the subdirectory
    for image_name in os.listdir(source_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_dir, image_name)
            image = cv2.imread(image_path)
            results = model.predict(image)

            # Check if any masks were detected and save those images
            if results[0].boxes:
                # Draw bounding boxes and labels on the image
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = results[0].names[box.cls[0].item()]
                    if label == 'Mask':  # Assuming 'mask' is a label for masks
                        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        image = cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Save the image with detections
                cv2.imwrite(os.path.join(output_dir, image_name), image)
    print(f'Finished processing images in {subdir}')


# Run the image processing in parallel for each subdirectory
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_images, subdirectories)

print('All images have been processed and saved with detections.')
