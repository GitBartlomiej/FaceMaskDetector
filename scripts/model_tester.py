import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Wyłącza GPU

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_evaluate_model(model_path, test_data_path, target_size, batch_size):
    # Załaduj model
    model = load_model(model_path)

    # Ustaw generator danych testowych
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        test_data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # Ewaluacja modelu
    model_eval = model.evaluate(test_set)

    # Wyświetl wyniki ewaluacji
    print(f"Model evaluation ({model_path}):", model_eval)

    # Opcjonalnie: Zwróć wyniki
    return model_eval

# Ścieżki do modeli
conv_model_path = './nn_models/Conv.h5'
conv_nocol_model_path = './nn_models/Conv_nocol.h5'
imagenet_model_path = './nn_models/imagenet.h5'

# Ścieżka do danych testowych
test_data_path = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/dataset/test/'

# Testowanie każdego modelu
# conv_eval = load_and_evaluate_model(conv_model_path, test_data_path, (150, 100), 10)
# conv_nocol_eval = load_and_evaluate_model(conv_nocol_model_path, test_data_path, (150, 100), 10)
# imagenet_eval = load_and_evaluate_model(imagenet_model_path, test_data_path, (224, 224), 32) # Przykładowy inny rozmiar dla modelu ImageNet

