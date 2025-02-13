import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
# import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Add,  LSTM, GRU, Bidirectional, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
import scipy.io
from tensorflow.keras.callbacks import *

K.clear_session()


num_classes = 12
epochs = 50

mat0 =         scipy.io.loadmat('infrared_ofi4_train.mat')
mat2 =         scipy.io.loadmat('infrared_ofi4_test.mat')
final_weights_path =   'weights_infrared_ofi4_GPViT_L2_two_fc512_bs16.hdf5'
#
x_train = mat0['x1']
y_train = np.transpose(mat0['y1'])

x_test = mat2['x2']
y_test = np.transpose(mat2['y2'])



x_train = np.transpose(x_train,(3,0,1,2))
x_test = np.transpose(x_test,(3,0,1,2))

y_train = y_train-1
y_test = y_test-1

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

mobile = GPViT_L2()


# mobile.summary()
x = mobile.layers[-2].output
# # model.summary()
# x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes,activation='softmax')(x)
# #
model = Model(inputs=mobile.input, outputs=predictions)
#

model.compile(Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

def scheduler(epoch):

    if epoch == 6: 
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, .00005)
        print('Learning rate changed!')
        
    if epoch == 10:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, .00001)
        print('Learning rate changed!')
    return K.get_value(model.optimizer.lr)

change_lr = LearningRateScheduler(scheduler)

callbacks_list = [
      ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
      change_lr
]


history=model.fit(x_train, y_train,
      batch_size=16, epochs=16,
      callbacks=callbacks_list,
      validation_data=(x_test, y_test))

