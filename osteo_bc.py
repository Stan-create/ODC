# -*- coding: utf-8 -*-
"""
Osteo bc
@author: Stanislav Pavlovich Gamenyuk

"""

from google.colab import drive
drive.mount ('/content/drive')

import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_df(tr_path):
    classes = []
    class_paths = []
    files = os.listdir(tr_path)
    for file in files:
        label_dir = os.path.join(tr_path, file)
        label = os.listdir(label_dir)
        for image in label:
            image_path = os.path.join(label_dir, image)
            class_paths.append(image_path)
            classes.append(file)
    image_classes = pd.Series(classes, name='Class')
    image_paths = pd.Series(class_paths, name='Class Path')
    tr_df = pd.concat([image_paths, image_classes], axis=1)
    return tr_df

tr_df = train_df('/content/drive/MyDrive/Colab Notebooks/Osteo binary classification/OS Collected Data')
tr_df

train_df, dummies_df = train_test_split(tr_df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(dummies_df, test_size=0.5, random_state=42)

batch_size = 16
img_size = (224,224)

tr_gen = ImageDataGenerator(brightness_range=(.8,1.2), vertical_flip=True)

gen = ImageDataGenerator()


tr_gen = tr_gen.flow_from_dataframe(train_df, x_col='Class Path',
                                 y_col='Class',batch_size=batch_size,
                                 target_size=img_size)

valid_gen = gen.flow_from_dataframe(valid_df, x_col='Class Path',
                                    y_col='Class',batch_size=batch_size,
                                    target_size=img_size)

ts_gen = gen.flow_from_dataframe(test_df, x_col='Class Path',
                                y_col='Class',batch_size=batch_size,
                                 target_size=img_size,shuffle=False)

class_dict = tr_gen.class_indices
classes = list(class_dict.keys())
images, labels = next(tr_gen)

plt.figure(figsize= (20, 20))

for i in range(16):
    plt.subplot(4,4,i+1)
    image = (images[i]/255)
    plt.imshow(image)
    index = np.argmax(labels[i])
    class_name = classes[index]
    plt.title(class_name, color='k', fontsize=20)
plt.show()

num_classes = len(classes)
num_classes

img_shape=(224,224,3)
base_model = tf.keras.applications.Xception(include_top= False, weights= "imagenet",
                            input_shape= img_shape, pooling= 'max')

"""# Keras Tuner"""

pip install -U keras-tuner

import keras_tuner
from keras_tuner.tuners import Hyperband
from tensorflow.keras import utils

def build_model(hp):
  model = Sequential([
      base_model,
      Flatten(),
      Dropout(0.3),
      Dense(units=hp.Int('units_hidden',
                                min_value=128,
                                max_value=600,
                                step=32),
                   activation=hp.Choice('activation',
                                        values=['selu',
                                                'gelu',
                                                'relu',
                                                'elu',
                                                'swish',
                                                'tanh'])),
                      Dropout(0.25),
      Dense(num_classes, activation='softmax')])

  model.compile(
      optimizer=hp.Choice('optimizer',
                          values=['nadam',
                                  'adadelta',
                                  'adagrad',
                                  'adam',
                                  'adamax',
                                  'ftrl',
                                  'rmsprop']),
      loss='categorical_crossentropy',
      metrics = ['accuracy'])
  return model

tuner = Hyperband(
    build_model,
    objective='val_accuracy',

    max_epochs=100,
    directory='test_directory_HB'
    )

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search_space_summary()

tuner.search(
    tr_gen,
    validation_data=valid_gen,
    batch_size=batch_size,
    epochs = 30,
    callbacks=[stop_early]
)

tuner.results_summary()

models = tuner.get_best_models(num_models=1)

for model in models:
  model.summary(),
  model.evaluate(ts_gen)
  print()

"""# Модель"""

model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate= 0.3),
    Dense(128, activation= 'relu'),
    Dropout(rate= 0.25),
    Dense(num_classes, activation= 'softmax')
])

model.compile(Adamax(learning_rate= 0.001),
              loss= 'binary_crossentropy',
              metrics= ['accuracy'])
model.build((None, 224, 224, 3))
model.summary()

history = model.fit(tr_gen,
                 epochs=7,
                 validation_data=valid_gen,
                 shuffle= True)

test_loss, test_acc = model.evaluate(ts_gen, verbose=1)

model.save("OsteoNet.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.legend(['тренировочная выборка', 'оценочная выборка'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Функция потерь')
plt.ylabel('Потери')
plt.xlabel('Эпоха')
plt.legend(['тренировочная выборка', 'оценочная выборка'], loc='upper left')
plt.show()

print(test_acc)

pred = model.predict(ts_gen)
pred = np.argmax(pred, axis=1)

labels = (tr_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred2 = [labels[k] for k in pred]

from sklearn.metrics import accuracy_score

y_test = test_df['Class']
print(classification_report(y_test, pred2))
print("Accuracy of the Model:","{:.1f}%".format(accuracy_score(y_test, pred2)*100))

classes=list(tr_gen.class_indices.keys())
print (classes)

model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/Osteo binary classification/Osteo_Net.keras')

class_dict = ts_gen.class_indices
classes = list(class_dict.keys())
images, labels = next(ts_gen)
x = 6

plt.figure(figsize=(30, 30))

plt.subplot(4,4,i+1)
image = images[x] / 255
plt.imshow(image)
index = np.argmax(labels[i])
class_name = classes[index]
plt.title(class_name, fontsize=20)
plt.axis('off')

plt.figure(figsize=(30,30))
sample = tf.expand_dims(images[x], 0)
pred = model.predict(sample)

score = tf.nn.softmax(pred[0])
title = "{}".format(classes[np.argmax(score)])

ax = plt.subplot(4,4,i+1)
image = images[x] / 255
plt.imshow(image)
plt.title(title)
plt.axis('off')