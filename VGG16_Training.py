import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tkinter as TK
from tkinter.filedialog import askdirectory, askopenfile

#---------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------VGG16-TRAINING--------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------


training_path = askdirectory(title = "Select The Training Folder")
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=training_path,target_size=(224,224),batch_size = 1000)
testing_path = askdirectory(title = "Select The Testing Folder")
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=testing_path, target_size=(224,224),batch_size = 1000)

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))
model.summary()

from keras.optimizers import Adam
opt = Adam(lr=0.001)

#validation_steps = len(testdata)//batch_size # if you have validation data 

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
checkpoint = ModelCheckpoint("VGG16_Stromal_Training.h5", monitor="val_accuracy", verbose=1, save_best_only=True, save_weights_only=False, mode="auto", period=1)
early = EarlyStopping(monitor="val_accuracy", min_delta=0, patience=20, verbose=1, mode="auto")
hist = model.fit_generator(steps_per_epoch = 10 , generator = traindata, validation_data = testdata, validation_steps = 10, epochs = 100,callbacks=[checkpoint,early]) # Change step_per_epoch value and validation_steps to equal values to one another.