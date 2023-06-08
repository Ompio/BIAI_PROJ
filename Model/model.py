import glob
import os
import random
import shutil
import warnings

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs Available:", len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.chdir('..')
print(os.getcwd())
os.chdir('fer2013')
emotions_dirs = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
if os.path.isdir('test/Angry') is False:
    os.makedirs('test/Angry')
    os.makedirs('test/Disgust')
    os.makedirs('test/Fear')
    os.makedirs('test/Happy')
    os.makedirs('test/Neutral')
    os.makedirs('test/Sad')
    os.makedirs('test/Surprise')
    for emotion in emotions_dirs:
        for c in random.sample(glob.glob(f'train/{emotion}/*'), 50):
            shutil.move(c, f'test/{emotion}')

train_path = 'train'
valid_path = 'validation'
test_path = 'test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(48, 48), classes=emotions_dirs, batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(48, 48), classes=emotions_dirs, batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(48, 48), classes=emotions_dirs, batch_size=10, shuffle=False)

#print(f"traint b: {train_batches.n}")

imgs, labels = next(train_batches);

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(imgs)
print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(48, 48, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(units=7, activation='softmax'),
])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)
os.chdir('..')
model.save('saved_model')
new_model = tf.keras.models.load_model('saved_model')
new_model.summary()
