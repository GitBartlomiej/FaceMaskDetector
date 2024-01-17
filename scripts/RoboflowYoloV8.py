import os
from roboflow import Roboflow
rf = Roboflow(api_key="JxCMjosaIa8gVEhzGMq6")
project = rf.workspace().project("mask-aify8")
model = project.version(3).model


path=os.getcwd()
path=os.path.join(path, 'dataset')

# declare train_path
path_train = os.path.join(path, 'train')

# declare test_path
path_test = os.path.join(path, 'test')

# Ścieżka do folderu ze zdjęciami oryginalnymi
original_images_path = path_train

image_path = "/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/dataset/train/with_mask/0_0_0.jpeg"

# infer on a local image
print(model.predict(
    image_path,
    confidence=40, overlap=30).json())

# visualize your prediction
model.predict(image_path, confidence=40, overlap=30).save("prediction.jpg")