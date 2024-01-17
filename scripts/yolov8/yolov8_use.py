import cv2
from ultralytics import YOLO

# Load your YOLO model with custom or pre-trained weights
model = YOLO('./runs/detect/train/weights/best.pt')

# Run predictions on an image
image_path = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/dataset/test/with_mask/black5.jpg'
image = cv2.imread(image_path)
results = model.predict(image)

# Draw bounding boxes and labels on the image
for box in results[0].boxes:
    # Extract the bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    # Determine the class label
    label = results[0].names[box.cls[0].item()]
    # Draw the bounding box rectangle and label on the image
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    image = cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the image
cv2.imshow('YOLOv8 Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
