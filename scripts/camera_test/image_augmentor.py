import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

# Ścieżka do folderu z klatkami
input_folder = 'klatki_filmu'

# Tworzenie folderów dla obróconych obrazów
output_folder = 'obrocone_obrazy'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(11):  # Od 0 do 10 stopni
    os.makedirs(os.path.join(output_folder, f"{i}_stopien"), exist_ok=True)

# Tworzenie folderu dla zoomu
zoom_folder = 'zoom'
os.makedirs(zoom_folder, exist_ok=True)

# Wczytywanie i przetwarzanie klatek
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = load_img(img_path)  # Wczytanie obrazu
        x = img_to_array(img)  # Konwersja do tablicy numpy
        x = np.expand_dims(x, axis=0)  # Zmiana kształtu do (1, wysokość, szerokość, kanały)

        # Aplikacja rotacji i zapisanie obrazów
        for i in range(11):  # Od 0 do 10 stopni
            # Inicjalizacja generatora z rotacją
            datagen = ImageDataGenerator(rotation_range=i)
            # Utworzenie iteratora z opcjami zapisu
            iterator = datagen.flow(x, batch_size=1, save_to_dir=os.path.join(output_folder, f"{i}_stopien"),
                                    save_prefix=f"{i}_stopien_", save_format='png')
            iterator.next()  # Generowanie i zapisywanie obrazu

# Aplikacja zoomu i zapisanie obrazów
zoom_factors = [0.5, 0.75, 1.0, 1.25, 1.5]  # Przykładowe wartości zoomu
for zoom in zoom_factors:
    # Inicjalizacja generatora z zoomem
    datagen = ImageDataGenerator(zoom_range=[zoom, zoom])

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = load_img(img_path)  # Wczytanie obrazu
            x = img_to_array(img)  # Konwersja do tablicy numpy
            x = np.expand_dims(x, axis=0)  # Zmiana kształtu do (1, wysokość, szerokość, kanały)

            # Utworzenie iteratora z opcjami zapisu
            iterator = datagen.flow(x, batch_size=1, save_to_dir=zoom_folder,
                                    save_prefix=f'zoom_{zoom}_', save_format='png')
            iterator.next()  # Generowanie i zapisywanie obrazu

print("Zakończono przetwarzanie obrazów.")
