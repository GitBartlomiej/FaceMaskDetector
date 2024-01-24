from yolov8.yolov8_main import Yolov8ImageDetector
from imagenet.imagenet_main import ImageNetImageDetector

yolov8_model = Yolov8ImageDetector()
imagenet_model = ImageNetImageDetector()

# yolov8_model.run_detection()
imagenet_model.run_detection()

