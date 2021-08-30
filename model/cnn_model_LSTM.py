#!/usr/bin/env python
# coding: utf-8

import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, LSTM, Reshape, TimeDistributed, Permute
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, MaxPooling3D, BatchNormalization, Conv3D, Concatenate
from keras.layers.merge import add
from keras import regularizers


def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
    # Input
    img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    action_input = Input(shape = (num_frames, num_actions))
    
    # Layers for Conv_LSTM // for image processing    
    conv_x1 = ConvLSTM2D(16, (3, 3), strides = (8,8) , return_sequences= True, padding='same', 
                         kernel_regularizer= regularizers.l2(0.0001))(img_input)    
    conv_x2 = ConvLSTM2D(16, (3, 3), strides = (2,2), activation = 'relu', return_sequences= True,  padding='same'
                         , kernel_regularizer= regularizers.l2(0.0001))(conv_x1)
    conv_x3 = ConvLSTM2D(32, (3, 3), strides = (1,1), activation = 'relu', return_sequences= True,  padding='same',
                         kernel_regularizer= regularizers.l2(0.0001))(conv_x2)
    conv_x = TimeDistributed(Flatten())(conv_x3)
    # Layer for action space
    act_x1 = LSTM(10, return_sequences= True)(action_input)
    
    # FC layer
    x = Concatenate()([conv_x, act_x1])    
    x = TimeDistributed(Dense(1024, activation = 'relu'))(x)
    x = Dropout(0.5)(x)
    # Collision channel        
    coll = TimeDistributed(Dense(output_dim))(x)    
    
    # Define steering-collision model
    model = Model(inputs=[img_input, action_input], outputs=[coll])
    print(model.summary())

    return model


