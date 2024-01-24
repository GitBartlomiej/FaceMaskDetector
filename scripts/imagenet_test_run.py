import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import cv2
import keras
import numpy as np

# Wczytanie modelu
# path = os.getcwd() + "/scripts/imagenet.h5"
imagenet = keras.models.load_model("./nn_models/imagenet.h5")

# Wczytanie obrazka
image_path = "/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/Datasets/with_mask/0_0_006ajtxFly1g5eb2v9iw4j30u00u0mzf.jpg"  # Zmień na rzeczywistą ścieżkę do obrazka
img = cv2.imread(image_path)

# Przygotowanie obrazu do analizy
resized_img = cv2.resize(img, (224, 224))  # Dostosuj do wymiarów oczekiwanych przez model
normalized_img = resized_img / 255.0
reshaped_img = np.reshape(normalized_img, (1, 224, 224, 3))

# Predykcja maski
result = imagenet.predict(reshaped_img)

# Wyświetlanie wyniku
if result[0][0] > 0.5:
    text = "No mask! Score: {:.2f}".format(result[0][0])
    cv2.putText(img, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
else:
    text = "Mask. Score: {:.2f}".format(result[0][0])
    cv2.putText(img, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

# Wyświetlenie obrazka z wynikami
cv2.imshow('Result', img)
cv2.waitKey(0)

# Zakończenie programu
cv2.destroyAllWindows()
