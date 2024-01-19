import os
import numpy as np
import cv2
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

# Path to the folder with frames
input_folder = 'movie_frames'

# Folders for different transformations
transformations = {
    'rotated': 'rotated_images',
    'zoomed': 'zoom',
    'moved': 'moved_images',
    'tilted': 'tilted_images',
    'blurred': 'blurred_images',
    'scaled': 'scaled_images',
    'dirty': 'dirty_images',
    'illuminated': 'illuminated_images'
}

# Creating folders for each transformation
for transform in transformations.values():
    os.makedirs(transform, exist_ok=True)

# Loading and processing frames
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = load_img(img_path)  # Loading the image
        x = img_to_array(img)  # Converting to a numpy array
        x = np.expand_dims(x, axis=0)  # Reshaping to (1, height, width, channels)

        # Apply transformations using ImageDataGenerator
        # Rotation
        for i in range(11):  # From 0 to 10 degrees
            datagen = ImageDataGenerator(rotation_range=i)
            iterator = datagen.flow(x, batch_size=1, save_to_dir=os.path.join(transformations['rotated'], f"{i}_degree"),
                                    save_prefix=f"{i}_degree_", save_format='png')
            iterator.next()

        # Zoom
        zoom_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
        for zoom in zoom_factors:
            datagen = ImageDataGenerator(zoom_range=[zoom, zoom])
            iterator = datagen.flow(x, batch_size=1, save_to_dir=transformations['zoomed'],
                                    save_prefix=f'zoom_{zoom}_', save_format='png')
            iterator.next()

        # Load original image using OpenCV for other transformations
        original_img = cv2.imread(img_path)

        # Moving
        # ... (Add your code for moving transformation using OpenCV)

        # Tilting
        # ... (Add your code for tilting transformation using OpenCV)

        # Blurring
        # ... (Add your code for blurring transformation using OpenCV)

        # Scaling
        # ... (Add your code for scaling transformation using OpenCV)

        # Adding Dirt
        # ... (Add your code for adding dirt transformation using OpenCV)

        # Changing Illumination
        # ... (Add your code for changing illumination transformation using OpenCV)

print("Finished processing images.")
