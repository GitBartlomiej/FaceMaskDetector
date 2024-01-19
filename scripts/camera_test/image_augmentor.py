import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

# Path to the folder with frames
input_folder = 'movie_frames'

# Creating folders for rotated images
output_folder = 'rotated_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(11):  # From 0 to 10 degrees
    os.makedirs(os.path.join(output_folder, f"{i}_degree"), exist_ok=True)

# Creating a folder for zoom
zoom_folder = 'zoom'
os.makedirs(zoom_folder, exist_ok=True)

# Loading and processing frames
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = load_img(img_path)  # Loading the image
        x = img_to_array(img)  # Converting to a numpy array
        x = np.expand_dims(x, axis=0)  # Reshaping to (1, height, width, channels)

        # Applying rotation and saving images
        for i in range(11):  # From 0 to 10 degrees
            # Initializing the generator with rotation
            datagen = ImageDataGenerator(rotation_range=i)
            # Creating an iterator with save options
            iterator = datagen.flow(x, batch_size=1, save_to_dir=os.path.join(output_folder, f"{i}_degree"),
                                    save_prefix=f"{i}_degree_", save_format='png')
            iterator.next()  # Generating and saving the image

# Applying zoom and saving images
zoom_factors = [0.5, 0.75, 1.0, 1.25, 1.5]  # Example zoom values
for zoom in zoom_factors:
    # Initializing the generator with zoom
    datagen = ImageDataGenerator(zoom_range=[zoom, zoom])

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = load_img(img_path)  # Loading the image
            x = img_to_array(img)  # Converting to a numpy array
            x = np.expand_dims(x, axis=0)  # Reshaping to (1, height, width, channels)

            # Creating an iterator with save options
            iterator = datagen.flow(x, batch_size=1, save_to_dir=zoom_folder,
                                    save_prefix=f'zoom_{zoom}_', save_format='png')
            iterator.next()  # Generating and saving the image

print("Finished processing images.")
