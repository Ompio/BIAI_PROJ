import glob
import os
import os
import pathlib
import random
import shutil
import warnings

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

os.chdir('..')
new_model = tf.keras.models.load_model('saved_model')
img = Image.open('test/Happy/850.jpg')

predictions = new_model.predict('fer2013/validation/Happy/850.jpg')
print(predictions[0])
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def plot_image(i, predictions_array, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)

  color = 'blue'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                color=color))

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)


#sprawdź prognozy
  i = 0
  plt.figure(figsize=(6, 3))
  plt.subplot(1, 2, 1)
  plot_image(i, predictions[i], img)
  plt.subplot(1, 2, 2)
  plot_value_array(i, predictions[i])
  plt.show()


