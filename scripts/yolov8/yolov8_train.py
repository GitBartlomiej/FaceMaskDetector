from ultralytics import YOLO
model = YOLO('yolov8n.pt') # załaduj wstępnie wytrenowany model
model.train(data='/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/Mask_Classify.v7i.yolov8/data.yaml', epochs=100, imgsz=832, device=0)
