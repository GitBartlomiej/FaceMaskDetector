import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
 # Używa tylko jednego, określonego GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPU available")
    try:
        # Ustawia maksymalne użycie pamięci na 50% dostępnej pamięci GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5732 // 2)]
        )
    except RuntimeError as e:
        # Błąd wywołany gdy konfiguracja pamięci jest ustawiona po inicjalizacji GPU
        print(e)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import cv2
import numpy as np
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


class ImageNetImageDetector:
    def __init__(self):
        self.model = load_model('/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/nn_models/'
                                'imagenet.h5')
        self.base_source_dir = ('/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/'
                                'metamorphic_transformed_dataset')
        self.base_output_dir = ('/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/'
                                'metamorphic_test_detection/imagenet')

        os.makedirs(self.base_output_dir, exist_ok=True)
        self.subdirectories = [d for d in os.listdir(self.base_source_dir) if
                               os.path.isdir(os.path.join(self.base_source_dir, d))]
        for subdir in self.subdirectories:
            os.makedirs(os.path.join(self.base_output_dir, subdir), exist_ok=True)

    @staticmethod
    def preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def process_images(self, subdir):
        source_dir = os.path.join(self.base_source_dir, subdir)
        output_dir = os.path.join(self.base_output_dir, subdir)

        results_list = []
        for image_name in os.listdir(source_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print('Processing image: ', image_name)
                image_path = os.path.join(source_dir, image_name)
                output_image_path = os.path.join(output_dir, image_name)
                if os.path.exists(output_image_path):
                    continue

                processed_image = self.preprocess_image(image_path)
                prediction = self.model.predict(processed_image)
                detected = 'No'
                original_img = cv2.imread(image_path)  # Read the original image using OpenCV
                if prediction[0][0] > 0.5:
                    detected = 'Yes'
                    text = "Mask Score: {:.2f}".format(prediction[0][0])
                    cv2.putText(original_img, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                    print(text)
                else:
                    text = "No Mask. Score: {:.2f}".format(prediction[0][0])
                    print(text)
                    cv2.putText(original_img, text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

                cv2.imwrite(output_image_path, original_img)

                results_list.append({'Folder': subdir, 'Image': image_name, 'Detected': detected})


        return results_list

    # def run_detection(self):
    #     all_results = []
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    #         futures = [executor.submit(self.process_images, subdir) for subdir in self.subdirectories]
    #         for future in concurrent.futures.as_completed(futures):
    #             all_results.extend(future.result())
    #     return all_results

    def run_detection(self):
        all_results = []
        for subdir in self.subdirectories:
            results = self.process_images(subdir)
            all_results.extend(results)
        return all_results

    def plot_results(self, detection_results):
        detection_log_df = pd.DataFrame(detection_results)
        if not detection_log_df.empty:
            for folder in detection_log_df['Folder'].unique():
                folder_df = detection_log_df[detection_log_df['Folder'] == folder]
                detection_counts = folder_df['Detected'].value_counts()
                plt.figure(figsize=(10, 6))
                detection_counts.plot(kind='bar')
                plt.title(f'Detection Results for {folder}')
                plt.xlabel('Detected Masks')
                plt.ylabel('Number of Images')
                plt.savefig(os.path.join(self.base_output_dir, f'detection_results_{folder}.png'))
                plt.close()
        else:
            print("No data available for plotting.")

# Usage example
detector = ImageNetImageDetector()
detection_results = detector.run_detection()
detector.plot_results(detection_results)
