# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 00:18:07 2022

@author: Ryzen

Project 1
Heart Disease Dataset
"""
#1. Import the packages
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, losses, metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime, os
import pandas as pd

#2. Import and read dataset 
data_path = r"C:\Users\Ryzen\Documents\tensorflow\data\heart.csv"
df = pd.read_csv(data_path)

#%%
#Inspect is there any NA value
print(df.isna().sum())

#%%
#3. Data preprocessing
# Split data into features and labels
features = df.copy()
labels = features.pop('target')

#%%
#One-hot encode for all the categorical features
features = pd.get_dummies(features)

#%%
#Convert dataframe into numpy array
features = np.array(features)
labels = np.array(labels)

#%%
#Perform train-test split
SEED=0
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=SEED)

#Data normalization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#%%
#4. Build a NN that overfits easily
nIn = x_test.shape[-1]
nClass = len(np.unique(y_test))

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(nIn,)))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(nClass, activation='softmax'))

#5. View your model
model.summary()

#%%
#6. Compile model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#7. Define callback functions
base_log_path = r"C:\Users\Ryzen\Documents\tensorflow\tb_logs"
log_path= os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '__Heart_Disease_Dataset')
es = EarlyStopping(monitor='val_loss', patience=10)
tb = TensorBoard(log_dir=log_path)

#Train model
history = model.fit(x_train,y_train, validation_data=(x_test, y_test), batch_size=128, epochs=10000, callbacks=[es,tb])

#%%
from numba import cuda 
device = cuda.get_current_device()
device.reset()