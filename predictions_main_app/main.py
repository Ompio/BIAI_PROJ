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
image = keras.utils.image_utils.load_img('test_files/c5.png', target_size=(48, 48))
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
    plt.xticks(range(7))
    plt.yticks([])
    thisplot = plt.bar(class_names, predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')


i = 0
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], image)
plt.subplots(figsize=(15, 15))
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], class_names)
plt.show()


