import os
import numpy as np
import cv2
from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator

# Path to the folder with frames
input_folder = 'movie_frames'

# Main folder for all metamorphic transformations
main_output_folder = 'metamorphic_transforms'
os.makedirs(main_output_folder, exist_ok=True)

# Subfolders for each transformation
transformations = ['rotation', 'zoom', 'movement', 'tilting', 'blurring', 'scaling', 'dirt', 'illumination']
for transform in transformations:
    os.makedirs(os.path.join(main_output_folder, transform), exist_ok=True)

# Loading and processing frames
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        img_path = os.path.join(input_folder, filename)
        img = load_img(img_path)  # Loading the image
        x = img_to_array(img)  # Converting to a numpy array
        x = np.expand_dims(x, axis=0)  # Reshaping to (1, height, width, channels)

        # Apply rotation using ImageDataGenerator
        datagen = ImageDataGenerator(rotation_range=10)
        iterator = datagen.flow(x, batch_size=1, save_to_dir=os.path.join(main_output_folder, 'rotation'),
                                save_prefix='rotation_', save_format='png')
        iterator.next()

        # Apply zoom using ImageDataGenerator
        datagen = ImageDataGenerator(zoom_range=[1.0, 1.5])
        iterator = datagen.flow(x, batch_size=1, save_to_dir=os.path.join(main_output_folder, 'zoom'),
                                save_prefix='zoom_', save_format='png')
        iterator.next()

        # Load original image using OpenCV for other transformations
        original_img = cv2.imread(img_path)

        # Movement (Translation)
        M = np.float32([[1, 0, 10], [0, 1, 10]])  # Shift by 10 pixels in x and y direction
        moved_img = cv2.warpAffine(original_img, M, (original_img.shape[1], original_img.shape[0]))
        cv2.imwrite(os.path.join(main_output_folder, 'movement', f"movement_{filename}"), moved_img)

        # Tilting (Rotation)
        center = (original_img.shape[1] // 2, original_img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, 10, 1)  # Rotate by 10 degrees
        tilted_img = cv2.warpAffine(original_img, M, (original_img.shape[1], original_img.shape[0]))
        cv2.imwrite(os.path.join(main_output_folder, 'tilting', f"tilting_{filename}"), tilted_img)

        # Blurring
        blurred_img = cv2.GaussianBlur(original_img, (5, 5), 0)
        cv2.imwrite(os.path.join(main_output_folder, 'blurring', f"blurring_{filename}"), blurred_img)

        # Scaling
        scaled_img = cv2.resize(original_img, None, fx=1.5, fy=1.5)
        cv2.imwrite(os.path.join(main_output_folder, 'scaling', f"scaling_{filename}"), scaled_img)

        # Adding Dirt (Adding Noise)
        noise = np.random.randint(0, 50, original_img.shape, dtype='uint8')
        dirty_img = cv2.add(original_img, noise)
        cv2.imwrite(os.path.join(main_output_folder, 'dirt', f"dirt_{filename}"), dirty_img)

        # Changing Illumination (Adjusting Brightness)
        brightness_matrix = np.ones(original_img.shape, dtype='uint8') * 50  # Increase brightness
        illuminated_img = cv2.add(original_img, brightness_matrix)
        cv2.imwrite(os.path.join(main_output_folder, 'illumination', f"illumination_{filename}"), illuminated_img)

print("Finished processing images.")
