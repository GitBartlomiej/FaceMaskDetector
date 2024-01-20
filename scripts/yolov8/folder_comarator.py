import os

# Ścieżki do folderów
train_images_dir = './dataset/train/images'
train_labels_dir = './dataset/train/labels'

# Lista nazw plików w folderze images bez rozszerzenia
image_files = set(os.path.splitext(file)[0] for file in os.listdir(train_images_dir) if file.endswith(('.png', '.jpg', '.jpeg')))

# Lista nazw plików w folderze labels bez rozszerzenia
label_files = set(os.path.splitext(file)[0] for file in os.listdir(train_labels_dir) if file.endswith('.txt'))

# Znajdź obrazy bez odpowiadających im plików etykiet i usuń je
for image_file in image_files:
    if image_file not in label_files:
        os.remove(os.path.join(train_images_dir, image_file + '.jpg'))  # Usuń plik obrazu
        print(f"Removed image: {image_file}.jpg")
