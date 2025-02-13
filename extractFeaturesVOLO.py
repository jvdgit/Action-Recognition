import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import  LSTM, GRU, Bidirectional, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
import scipy.io
from tensorflow.keras.callbacks import *
from tensorflow.python.keras.backend import eager_learning_phase_scope

from keras_cv_attention_models import visualizing, gpvit

from PIL import Image
import glob


num_classes = 12
epochs = 25

mat0 =         scipy.io.loadmat('infrared_ofi4_train.mat')
mat1 =         scipy.io.loadmat('infrared_ofi4_test.mat')
fname =  'weights_infrared_ofi4_GPViT_L2_two_fc512_bs16.hdf5'
sname = 'Features_infrared_ofi4_GPViT_L2_two_fc512_bs16.mat'
pname =    'Score_infrared_ofi4_GPViT_L2_two_fc512_bs16.mat'

x_train = mat0['x1']
y_train = np.transpose(mat0['y1'])
x_test = mat1['x2']
y_test = np.transpose(mat1['y2'])


x_train = np.transpose(x_train,(3,0,1,2))
x_test = np.transpose(x_test,(3,0,1,2))

y_train = y_train-1
y_test = y_test-1

y1 = y_train
y2 = y_test

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


mobile = gpvit.GPViT_L2()

#mobile.summary()
x = mobile.layers[-2].output

x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.6)(x)
x = Dense(512, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes,activation='softmax')(x)
#
model = Model(inputs=mobile.input, outputs=predictions)

# model.summary()

model.load_weights(fname)

get_last_layer_output = K.function([model.layers[0].input],
                                  [model.layers[-3].output])

trainFeatures = np.zeros((len(x_train), 512))

eager_learning_phase_scope(value=1)

for i in range(len(x_train)):
    trainFeatures[i,:] = get_last_layer_output(x_train[i:i+1,:,:,:])[0]
    if i%100 == 0:
        print(i)
    
print('\nTrain extracted successfully!!')

testFeatures = np.zeros((len(x_test), 512))
       
eager_learning_phase_scope(value=0)

for i in range(len(x_test)):
        testFeatures[i,:] = get_last_layer_output(x_test[i:i+1,:,:,:])[0]
        if i%100 == 0:
            print(i)

y_pred = model.predict(x_test)

print(sname)
scipy.io.savemat(sname, mdict={'trainFeatures': trainFeatures, 'testFeatures': testFeatures, 'y1':y1, 'y2':y2})
scipy.io.savemat(pname, mdict={'y_pred': y_pred})

print('\nFeatures extracted successfully!!')