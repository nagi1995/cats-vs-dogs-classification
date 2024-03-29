# -*- coding: utf-8 -*-
"""cats_vs_dogs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n9FMJmARxnsxktfdc2vvpPPN_Unb0xvH
"""

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
# change permission
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c dogs-vs-cats

!unzip "/content/train.zip" -d "./"
!unzip "/content/test1.zip" -d "./"

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.util import plot_model
from tensorflow.keras.callbacks import *
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import random

tf.__version__

len(os.listdir("/content/test1")), len(os.listdir("/content/train"))

for i in random.sample(range(len(os.listdir("/content/train"))), 10):
  print(cv2.imread("/content/train/" + os.listdir("/content/train")[i]).shape)

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

dataset_path = "./train"

filenames = os.listdir(dataset_path)
categories = []
for f_name in filenames:
    
    if(f_name.split(".")[0] == "dog"):
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({'filename':filenames, 'category':categories})
df.head()

reduce_lr = ReduceLROnPlateau(monitor = "val_loss", 
                              factor = .4642, 
                              patience = 3, 
                              verbose = 1, 
                              min_delta = 0.001, 
                              mode = "min")
earlystop = EarlyStopping(monitor = "val_loss", 
                          patience = 10, 
                          verbose = 1, 
                          mode = "min"
                          )

tf.keras.backend.clear_session()
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (Image_Width, Image_Height, Image_Channels)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

df["category"] = df["category"].replace({0:'cat', 1:'dog'})
train_df, validate_df = train_test_split(df, test_size = 0.2, random_state = 42)
train_df = train_df.reset_index(drop = True)
validate_df = validate_df.reset_index(drop = True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

train_datagen = ImageDataGenerator(rotation_range = 15, rescale = 1./255, shear_range = 0.1, zoom_range = 0.2, horizontal_flip = True, width_shift_range = 0.1, height_shift_range = 0.1)
train_generator = train_datagen.flow_from_dataframe(train_df, dataset_path, x_col = 'filename', y_col = 'category', target_size = Image_Size, class_mode = 'categorical', batch_size = batch_size)

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, dataset_path, x_col = 'filename', y_col = 'category', target_size = Image_Size, class_mode = 'categorical', batch_size = batch_size)

checkpoint = ModelCheckpoint(filepath = "./1/weights.h5", 
                             monitor = "val_loss", 
                             verbose = 1, 
                             save_best_only = True,
                             mode = "min")

callbacks_list = [checkpoint, reduce_lr, earlystop]
epochs = 150
history = model.fit(train_generator, 
                    epochs = epochs, 
                    validation_data = validation_generator, 
                    validation_steps = total_validate//batch_size, 
                    steps_per_epoch = total_train//batch_size, 
                    callbacks = callbacks_list)

tf.keras.backend.clear_session()
model = Sequential()
model.add(SeparableConv2D(32, (3, 3), input_shape = (Image_Width, Image_Height, Image_Channels)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(SeparableConv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(SeparableConv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

checkpoint = ModelCheckpoint(filepath = "./2/weights.h5", 
                             monitor = "val_loss", 
                             verbose = 1, 
                             save_best_only = True,
                             mode = "min")

callbacks_list = [checkpoint, reduce_lr, earlystop]
epochs = 60
history = model.fit(train_generator, 
                    epochs = epochs, 
                    validation_data = validation_generator, 
                    validation_steps = total_validate//batch_size, 
                    steps_per_epoch = total_train//batch_size, 
                    callbacks = callbacks_list)

