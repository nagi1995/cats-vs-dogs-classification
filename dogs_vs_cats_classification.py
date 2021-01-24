# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:51:02 2020

@author: Nagesh
code is taken from https://data-flair.training/blogs/cats-dogs-classification-deep-learning-project-beginners/ and modified
"""

#%%
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

#%%

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

#%%

dataset_path = "dogs-vs-cats/train"
test_path = "dogs-vs-cats/test"

filenames = os.listdir(dataset_path)
categories = []
for f_name in filenames:
    
    if(f_name.split(".")[0] == "dog"):
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({'filename':filenames, 'category':categories})

#%%

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (Image_Width, Image_Height, Image_Channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

print("model summary: ", model.summary())


#%%

df["category"] = df["category"].replace({0:'cat', 1:'dog'})
train_df, validate_df = train_test_split(df, test_size = 0.2, random_state = 42)
train_df = train_df.reset_index(drop = True)
validate_df = validate_df.reset_index(drop = True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size = 15

#%%

train_datagen = ImageDataGenerator(rotation_range = 15, rescale = 1./255, shear_range = 0.1, zoom_range = 0.2, horizontal_flip = True, width_shift_range = 0.1, height_shift_range = 0.1)
train_generator = train_datagen.flow_from_dataframe(train_df, dataset_path, x_col = 'filename', y_col = 'category', target_size = Image_Size, class_mode = 'categorical', batch_size = batch_size)

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, dataset_path, x_col = 'filename', y_col = 'category', target_size = Image_Size, class_mode = 'categorical', batch_size = batch_size)

test_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
test_generator = train_datagen.flow_from_dataframe(train_df, test_path, x_col='filename', y_col='category', target_size=Image_Size, class_mode='categorical', batch_size=batch_size)

#%%

epochs = 10
history = model.fit_generator(train_generator, epochs = epochs, validation_data = validation_generator, validation_steps = total_validate//batch_size, steps_per_epoch = total_train//batch_size) 


#%%

model.save("model.h5")

#%%

test_filenames = os.listdir(test_path)
test_df = pd.DataFrame({'filename' : test_filenames})
nb_samples = test_df.shape[0]


#%%
predict = model.predict_generator(test_generator, steps = np.ceil(nb_samples/batch_size))

#%%

test_df["category"] = np.argmax(predict, axis = -1)


label_map = dict((v, k) for k, v in test_generator.class_indices.items())
test_df["category"].replace(label_map)
test_df["category"] = test_df["category"].replace({"dog" : 1, "cat" : 0})


#%%

sample_test = test_df.head(10)
sample_test.head()
plt.figure(figsize = (12, 24))

for index, row in sample_test.iterrows():
    filename = row["filename"]
    category = row["category"]
    img = load_img(os.path.join(dataset_path, filename), target_size = Image_Size)
    plt.subplot(5, 2, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')')
    plt.tight_layout()
    plt.show()

