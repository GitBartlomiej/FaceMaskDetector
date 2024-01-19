import cv2
import os

# Nazwa pliku wejściowego
input_video = 'output.avi'

# Utworzenie nowego folderu na zdjęcia
output_folder = 'movie_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicjalizacja obiektu VideoCapture
cap = cv2.VideoCapture(input_video)

# Sprawdzenie, czy udało się otworzyć plik wideo
if not cap.isOpened():
    print("Błąd: Nie można otworzyć pliku wideo.")
    exit()

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Przerwanie pętli, jeśli nie ma więcej klatek

    # Zapisanie klatki jako zdjęcie
    frame_path = os.path.join(output_folder, f"klatka_{frame_number:04d}.png")
    cv2.imwrite(frame_path, frame)
    frame_number += 1

# Zwolnienie obiektu cap i zamknięcie wszystkich okien
cap.release()
cv2.destroyAllWindows()

print(f"Zapisano {frame_number} klatek.")
