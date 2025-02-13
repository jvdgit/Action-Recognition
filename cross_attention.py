#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:26:59 2017

@author: javed
"""

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import  Conv1D, LSTM, GRU, Concatenate, Bidirectional, Flatten, GlobalAveragePooling1D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io
# import matplotlib.pyplot as plt
import os
from tensorflow.keras.optimizers import SGD, RMSprop, Adam



#from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.layers import BatchNormalization
#from keras_self_attention import SeqSelfAttention
#from temporal_pooling import TemporalAveragePooling2D
# fix random seed for reproducibility
np.random.seed(7)

# Cross-Attention Layer
class CrossAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.projection_dim = embed_dim // num_heads
        
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        scale = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(scale)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, projection_dim)

    def call(self, inputs):
        query, key_value = inputs
        batch_size = tf.shape(query)[0]
        
        query = self.query_dense(query)
        key = self.key_dense(key_value)
        value = self.value_dense(key_value)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)
        return output
    
    def get_config(self):
        config = super(CrossAttention, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads
        })
        return config

# Model definition with Cross-Attention
def create_cross_attention_model(input_shape1, input_shape2, num_classes, embed_dim=512, num_heads=16, dropout_rate=0.4):
    # Define two input sequences
    input1 = layers.Input(shape=input_shape1)
    input2 = layers.Input(shape=input_shape2)
    
    # Cross-attention layer
    cross_attention1 = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)([input1, input2])
    cross_attention2 = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)([input2, input1])
    
    cross_attention = Concatenate(name="Concatenate")([cross_attention1, cross_attention2])
    
    # Pooling layer to reduce sequence dimension
    pooled_output = layers.GlobalAveragePooling1D()(cross_attention)
    # 
    
    # Fully connected layers
    fc1 = layers.Dense(64, activation='relu')(pooled_output)
    dropout = layers.Dropout(dropout_rate)(fc)
    fc2 = layers.Dense(64, activation='relu')(fc1)
    dropout = layers.Dropout(dropout_rate)(fc2)
    
    # Output classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(dropout)
    
    # Build the model
    model = models.Model(inputs=[input1, input2], outputs=outputs)
    return model


mat = scipy.io.loadmat('infrared_bilstm_fc256_bs8_rnn_rnn.mat')
mat2 = scipy.io.loadmat('color_bilstm_fc256_bs8_rnn_rnn.mat')
sname =          'score.mat'
final_weights_path = 'wt.h5'


x_train = mat['x1']
y_train = mat['y1']
x_test = mat['x2']
y_test = mat['y2']

x_train2 = mat2['x1']
y_train2 = mat2['y1']
x_test2 = mat2['x2']
y_test2 = mat2['y2']

y1 = mat['y1']
y2 = mat['y2']

data_dim = 256
embed_dim = data_dim
num_heads = 8
timesteps = 1
dropout_rate = 0.3

num_classes = 12
nb_epoch = 16
model_path = 'models'

x_train = x_train.reshape(x_train.shape[0], timesteps,data_dim)
x_test = x_test.reshape(x_test.shape[0], timesteps, data_dim)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train2 = x_train.reshape(x_train.shape[0], timesteps,data_dim)
x_test2 = x_test.reshape(x_test.shape[0], timesteps, data_dim)
x_train2 = x_train.astype('float32')
x_test2 = x_test.astype('float32')


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


y_train2 = keras.utils.to_categorical(y_train2, num_classes)
y_test2 = keras.utils.to_categorical(y_test2, num_classes)

#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

start_time = time.time()

nunits = 64

input_shape1 = (timesteps,data_dim)

input_shape2 =  (timesteps,data_dim)


model = create_cross_attention_model(input_shape1, input_shape2, num_classes, embed_dim, num_heads, dropout_rate)

# model.summary()


model.compile(loss='categorical_crossentropy',
                # optimizer='RMSprop',
                optimizer = RMSprop(lr=0.0005),
              metrics=['accuracy'])

# print(model.summary())


callbacks_list = [
     ModelCheckpoint(final_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True)
]

history=model.fit([x_train2, x_train], y_train,
          batch_size=8, epochs=nb_epoch,
          callbacks=callbacks_list,
          validation_data=([x_test2, x_test], y_test))

