import glob
import os
import os
import pathlib
import random
import shutil
import warnings

import keras.utils.image_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

os.chdir('..')
image = keras.utils.image_utils.load_img('test_files/test2.jpg', target_size=(48, 48))
image_array = keras.utils.image_utils.img_to_array(image)
input_image = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
print(input_image.shape)
new_model = tf.keras.models.load_model('saved_model')

new_model.summary()
predictions = new_model.predict(input_image)
print(predictions[0])
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def plot_image(i, predictions_array, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    color = 'blue'

    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         color=color))


def plot_value_array(i, predictions_array, class_names):
    plt.grid(False)
    plt.xticks(range(7), rotation=45)
    plt.yticks([])
    thisplot = plt.bar(class_names, predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')




i = 0
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], image)
#plt.subplots(figsize=(4, 3))
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], class_names)
plt.show()

# TODO używać tego niżej żeby sprawdzać wiele zdjęć na raz podczas testów, tylko trzeba to poprawić najpierw
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

# TODO kod od obsługi kamery
#import cv2
#
# # Inicjalizacja obiektu przechwytującego obraz z kamery
# cap = cv2.VideoCapture(0)  # 0 oznacza indeks kamery w systemie (może być inny w zależności od konfiguracji)
#
# # Sprawdzenie, czy kamera została poprawnie otwarta
# if not cap.isOpened():
#     print("Nie można otworzyć kamery")
#     exit()
#
# # Odczyt obrazu z kamery w pętli
# while True:
#     # Odczyt klatki z kamery
#     ret, frame = cap.read()
#
#     # Sprawdzenie, czy klatka została poprawnie odczytana
#     if not ret:
#         print("Błąd odczytu klatki z kamery")
#         break
#
#     # Wyświetlenie klatki
#     cv2.imshow('Kamera', frame)
#
#     # Przerwanie pętli po naciśnięciu klawisza 'q'
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# # Zamknięcie obiektu przechwytującego obraz z kamery i zamknięcie okna
# cap.release()
# cv2.destroyAllWindows()

