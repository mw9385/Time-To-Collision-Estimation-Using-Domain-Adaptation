#!/usr/bin/env python
# coding: utf-8

import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, LSTM, Reshape, TimeDistributed, Permute
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, MaxPooling3D, BatchNormalization, Conv3D, Concatenate
from keras.layers.merge import add
from keras import regularizers, optimizers, losses, metrics, models
import keras.backend as K

"""
Model parameter tuning
- Fix the number of data: 1,000
- Check the validation loss and validation MAE
"""

"First_model"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
# #     action_input = Input(shape = (num_frames, num_actions))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)      
#     x = Activation('relu')(conv_x)    
#     x = Dropout(0.5)(x)  
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model


"Second_model: model_test_2"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
#     conv_x1 = Dropout(0.5)(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
#     conv_x2 = Dropout(0.5)(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
#     conv_x3 = Dropout(0.5)(conv_x3)
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)      
#     x = Activation('relu')(conv_x)    
#     x = Dropout(0.5)(x)  
#     x = Dense(512)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.5)(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"Third_model: model_test_3"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)      
#     x = Activation('relu')(conv_x)    
#     x = Dense(512)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.5)(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"Fourth_model: model_test_4"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(128, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)      
#     x = Activation('relu')(conv_x)            
#     x = Dropout(0.5)(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"Fifth_model: model_test_5"
"Adding fully connected layer is bad (Revised model 4)"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(128, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)      
#     x = Activation('relu')(conv_x)            
#     x = Dropout(0.5)(x)
#     x = Dense(512)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.5)(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"Sixth_model: model_test6"
"Remove batch normalization and fully connected layer"
"Removing batch normalization results in bad loss"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(128, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = Activation('relu')(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)      
#     x = Activation('relu')(conv_x)            
#     x = Dropout(0.5)(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"7th_model: model_test7"
"Revised version of model3"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
#     conv_x1 = Dropout(0.5)(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
#     conv_x2 = Dropout(0.5)(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
#     conv_x3 = Dropout(0.5)(conv_x3)
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)      
#     x = Activation('relu')(conv_x)    
#     x = Dropout(0.5)(x)  
#     x = Dense(512)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.5)(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"8th_model: model_test8"
"Revised version of model 7"
"Remove Dropout"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.5)(x)
#     x = Dense(512)(x)
#     x = Activation('relu')(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"9th_model: model_test9"
"Remove one layer"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(64, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)              
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x2)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.5)(x)

#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"10th_model: model_test_10"
"Remove one layer"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(128, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(64, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)              
    
#     conv_x3 = ConvLSTM2D(64, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)              
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x2)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.5)(x)    

#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"12th_model: model_test_12"
"Revised from model 11"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.5)(x)
#     x = Dense(2048)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.5)(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"11th_model: model_test_11"
"Revised from model 8 --> time_to_coll_models (V1, V2, ... etc.)"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.5)(x)
#     x = Dense(1024)(x)
#     x = Activation('relu')(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model


"Time to collision model V5"
"Applying Monte Carlo dropout"
"Adding normal dropout will depreciate the model performance."
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
#     conv_x1 = Dropout(0.5)(conv_x1, training = True)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)
#     conv_x2 = Dropout(0.5)(conv_x2, training = True)

    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
#     conv_x3 = Dropout(0.5)(conv_x3, training = True)
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.5)(x, training = True)
#     x = Dense(1024)(x)
#     x = Activation('relu')(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model


"Time to collision model: mc_rnn_dropout"
"Applying Monte Carlo dropout --> Very bad"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001),
#                          recurrent_dropout= 0.2)(img_input, training=True)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001),
#                          recurrent_dropout= 0.2)(conv_x1, training=True)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)

    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001),
#                          recurrent_dropout= 0.2)(conv_x2, training=True)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.5)(x, training = True)
#     x = Dense(1024)(x)
#     x = Activation('relu')(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

"Time to collision model: FC mc dropout"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(img_input)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x1)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)

    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
#                          kernel_regularizer= regularizers.l2(0.0001))(conv_x2)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.5)(x, training = True)
#     x = Dense(1024)(x)
#     x = Activation('relu')(x)
#     # Collision channel        
#     coll = (Dense(output_dim))(x)    
    
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [coll])  
#     print(model.summary())
#     return model

# "Bayseian NN: V_1 ~ V_4"
# def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
#     # Input
#     img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
#     # First layer
#     conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
#                         recurrent_dropout= 0.2)(img_input, training=True)    
#     conv_x1 = BatchNormalization()(conv_x1)
#     conv_x1 = Activation('relu')(conv_x1)
    
#     # Second layer
#     conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same',                          
#                         recurrent_dropout= 0.2)(conv_x1, training=True)    
#     conv_x2 = BatchNormalization()(conv_x2)
#     conv_x2 = Activation('relu')(conv_x2)

    
#     # Third layer
#     conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same',                          
#                         recurrent_dropout= 0.2)(conv_x2, training=True)    
#     conv_x3 = BatchNormalization()(conv_x3)
#     conv_x3 = Activation('relu')(conv_x3)        
    
#     # Fully connected layer
#     conv_x = Flatten()(conv_x3)   
#     x = Activation('relu')(conv_x)
#     x = Dropout(0.2)(x, training = True)
#     x = Dense(1024)(x)
# #     x = Activation('relu')(x)
#     # Collision channel      
#     mean = Dense(output_dim, name='mean')(x)
#     variance = Dense(output_dim, name='variance')(x)
               
#     # Define steering-collision model
#     model = Model(inputs = [img_input], outputs = [mean, variance])     
#     print(model.summary())
#     return model

"Bayseian NN: V_5 & 6"
"Add more filters in the layer"

def cnn_lstm(img_width, img_height, img_channel, num_frames, output_dim, num_actions):
    
    # Input
    img_input = Input(shape=(num_frames, img_width, img_height, img_channel))
    
    # First layer
    conv_x1 = ConvLSTM2D(32, (5, 5), strides = (2,2), return_sequences= True, padding='same', 
                         kernel_regularizer= regularizers.l2(0.0001),
                         recurrent_dropout=0.2)(img_input, training = True)    
    conv_x1 = BatchNormalization()(conv_x1)
    conv_x1 = Activation('relu')(conv_x1)
    
    # Second layer
    conv_x2 = ConvLSTM2D(32, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
                         kernel_regularizer= regularizers.l2(0.0001),
                         recurrent_dropout = 0.2)(conv_x1, training = True)    
    conv_x2 = BatchNormalization()(conv_x2)
    conv_x2 = Activation('relu')(conv_x2)

    # Third layer
    conv_x3 = ConvLSTM2D(16, (3, 3), strides = (2,2), return_sequences= True, padding='same', 
                         kernel_regularizer= regularizers.l2(0.0001),
                         recurrent_dropout = 0.2)(conv_x2, training = True)    
    conv_x3 = BatchNormalization()(conv_x3)
    conv_x3 = Activation('relu')(conv_x3)        
    
    # Fully connected layer
    conv_x = Flatten()(conv_x3)   
    x = Activation('relu')(conv_x)
    x = Dropout(0.2)(x, training = True)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    # Collision channel        
    mean = Dense(output_dim, name='mean')(x)
    variance = Dense(output_dim, name='variance')(x) 
    
    # Define steering-collision model
    model = Model(inputs = [img_input], outputs = [mean, variance])  
    print(model.summary())
    return model
