#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:26:59 2017

@author: javed
"""

import numpy as np
from tensorflow import keras
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import  LSTM, GRU, Bidirectional, Flatten, GlobalAveragePooling1D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io
import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import BatchNormalization

from keras_multi_head import MultiHeadAttention
from tensorflow.keras.layers import MultiHeadAttention

np.random.seed(7)

#mat = scipy.io.loadmat('gyro_truncated_norm.mat')
mat = scipy.io.loadmat('infrared_concat4_two_fc512_bs16_rnn.mat')
sname =          'Score_infrared_concat4_two_fc512_bs16_rnn.mat'
final_weights_path = 'sa.h5'


x_train = mat['x1']
y_train = mat['y1']
x_test = mat['x2']
y_test = mat['y2']

y1 = mat['y1']
y2 = mat['y2']

data_dim = 512*3#1024+1024#+2048
#data_dim = 1024
#data_dim = 2048
timesteps = 4

num_classes = 12
nb_epoch = 11
model_path = 'models'

x_train = x_train.reshape(x_train.shape[0], timesteps,data_dim)
x_test = x_test.reshape(x_test.shape[0], timesteps, data_dim)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

start_time = time.time()

nunits = 64
#
model = Sequential()
model.add(Bidirectional(LSTM(nunits, return_sequences = True), input_shape=(timesteps, data_dim)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # return a single vector of dimension 32
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
#              optimizer='RMSprop',
              optimizer = RMSprop(lr=0.0005),
              metrics=['accuracy'])


callbacks_list = [
     ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True)
]

history=model.fit(x_train, y_train,
          batch_size=8, epochs=nb_epoch,
          callbacks=callbacks_list,
          validation_data=(x_test, y_test))

model.load_weights(final_weights_path)

y_pred = model.predict(x_test)
scipy.io.savemat(sname, mdict={'y_pred': y_pred})

