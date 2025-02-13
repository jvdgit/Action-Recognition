# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:59:44 2020

@author: Javed
"""
import tensorflow as tf
from tensorflow import keras

#import keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as k

#from tensorflow.keras.layers.core import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import regularizers
from tensorflow.keras.layers import ConvLSTM2D, GlobalAveragePooling3D, ZeroPadding3D, Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications import imagenet_utils

import scipy.io
from tensorflow.keras.callbacks import *

from i3d_inception import Inception_Inflated3d

K.clear_session()

NUM_FRAMES = 16
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
NUM_CHANNELS = 3

NUM_CLASSES = 12

dropout_prob = 0.7

mat0 =      scipy.io.loadmat('color_frames16x4_train_1.mat')
mat1 =      scipy.io.loadmat('color_frames16x4_train_2.mat')
mat2 =      scipy.io.loadmat('color_frames16x4_test.mat')


final_weights_path = 'weights_color_frames16x4_i3d_two_fc512_bs8_new.hdf5'

x_train = np.concatenate((mat0['x1_1'], mat0['x1_2'], mat0['x1_3'], mat0['x1_4'], mat1['x1_1'], mat1['x1_2'], mat1['x1_3'], mat1['x1_4']), axis=4)
y_train = np.transpose(np.concatenate((mat0['y1'], mat1['y1']), axis=1))

x_test = mat2['x2']
y_test = np.transpose(mat2['y2'])


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


x_train = np.transpose(x_train,(4,3,0,1,2))
x_test = np.transpose(x_test,(4,3,0,1,2))

y_train = y_train-1
y_test = y_test-1

#print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)


model = Inception_Inflated3d(
                include_top=False,
                weights='flow_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_CHANNELS),
                classes=NUM_CLASSES)



model.compile(Adam(lr=.0003), loss='categorical_crossentropy', metrics=['accuracy'])

#model.summary()

def scheduler(epoch):
    if epoch == 10:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, .00005)
        print('Learning rate changed!')
    if epoch == 15:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, .00001)
        print('Learning rate changed!')    
    return K.get_value(model.optimizer.lr)

change_lr = LearningRateScheduler(scheduler)
#
callbacks_list = [
      ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
      change_lr
]
#
#
history=model.fit(x_train, y_train,
      batch_size=8, epochs=20,
      callbacks=callbacks_list,
      validation_data=(x_test, y_test))
