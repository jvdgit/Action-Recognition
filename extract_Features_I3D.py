import tensorflow as tf
from tensorflow import keras

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as k

from tensorflow.keras.optimizers import *
from tensorflow.keras import regularizers
from tensorflow.keras.layers import ConvLSTM2D, GlobalAveragePooling3D, ZeroPadding3D, Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications import imagenet_utils
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
import scipy.io
from tensorflow.keras.callbacks import *
from tensorflow.python.keras.backend import eager_learning_phase_scope


from i3d_inception import Inception_Inflated3d
K.clear_session()

NUM_FRAMES = 16
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
NUM_CHANNELS = 3
NUM_CLASSES = 12

dropout_prob = 0.7

mat0 =      scipy.io.loadmat('infrared_frames16x4_train_1.mat')
mat1 =      scipy.io.loadmat('infrared_frames16x4_train_2.mat')
mat2 =      scipy.io.loadmat('infrared_frames16x4_test.mat')

x_train = np.concatenate((mat0['x1_1'], mat0['x1_2'], mat0['x1_3'], mat0['x1_4'], mat1['x1_1'], mat1['x1_2'], mat1['x1_3'], mat1['x1_4']), axis=4)
y_train = np.transpose(np.concatenate((mat0['y1'], mat1['y1']), axis=1))

x_test = mat2['x2']
y_test = np.transpose(mat2['y2'])

fname =  'weights_infrared_frames16x4_i3d_two_fc512_bs8.hdf5'
sname = 'Features_infrared_frames16x4_i3d_two_fc512_bs8.mat'
pname =    'Score_infrared_frames16x4_i3d_two_fc512_bs8.mat'
##


x_train = np.transpose(x_train,(4,3,0,1,2))
x_test = np.transpose(x_test,(4,3,0,1,2))

print(x_train.shape)


#
y_train = y_train-1
y_test = y_test-1

y1 = y_train
y2 = y_test
#print(x_train.shape)

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Inception_Inflated3d(
                include_top=False,
                weights='rgb_imagenet_and_kinetics',
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_CHANNELS),
                classes=NUM_CLASSES)
#
model.load_weights(fname)

#model.summary()

get_last_layer_output = K.function([model.layers[0].input],
                                  [model.layers[-2].output])


trainFeatures = np.zeros((len(x_train), 512))

i = 0

eager_learning_phase_scope(value=1)
for i in range(len(x_train)):
    trainFeatures[i,:] = get_last_layer_output(x_train[i:i+1,:,:,:])[0]
    if i%100 == 0:
        print(i)
# #
testFeatures = np.zeros((len(x_test), 512))

eager_learning_phase_scope(value=0)

for i in range(len(x_test)):
        testFeatures[i,:] = get_last_layer_output(x_test[i:i+1,:,:,:])[0]
        if i%100 == 0:
            print(i)
            
y_pred = model.predict(x_test)
scipy.io.savemat(sname, mdict={'trainFeatures': trainFeatures, 'testFeatures': testFeatures, 'y1':y1, 'y2':y2})
scipy.io.savemat(pname, mdict={'y_pred': y_pred})

print('\nFeatures extracted successfully!!')