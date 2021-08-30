#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Add
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers


# In[2]:


def VGG_16(width, height, num_channels):
    # define model 1
    model_input_1 = Input(shape=(width, height, num_channels))
    # block 1
    model_x1 = Conv2D(64, (3,3), padding = 'same')(model_input_1)
    model_x1 = Activation('relu')(model_x1)
    model_x1 = Conv2D(64, (3,3), padding = 'same')(model_x1)
    model_x1 = Activation('relu')(model_x1)
    model_x1 = MaxPooling2D((4,4), strides = (4,4))(model_x1)
    # block 2
    model_x1 = Conv2D(256, (3,3), padding = 'same')(model_x1)
    model_x1 = Activation('relu')(model_x1)
    model_x1 = Conv2D(256, (3,3), padding = 'same')(model_x1)
    model_x1 = Activation('relu')(model_x1)
    model_x1 = MaxPooling2D((2,2), strides = (2,2))(model_x1)
    # block 3
    model_x1 = Conv2D(512, (3,3), padding = 'same')(model_x1)
    model_x1 = Activation('relu')(model_x1)
    model_x1 = Conv2D(512, (3,3), padding = 'same')(model_x1)
    model_x1 = Activation('relu')(model_x1)
    model_x1 = MaxPooling2D((2,2), strides = (2,2))(model_x1)
    # FC 1
    model_fc1 = Flatten()(model_x1)
    
    # define model 2
    model_input_2 = Input(shape=(width, height, num_channels))
    # block 1
    model_x2 = Conv2D(64, (3,3), padding = 'same')(model_input_2)
    model_x2 = Activation('relu')(model_x2)
    model_x2 = Conv2D(64, (3,3), padding = 'same')(model_x2)
    model_x2 = Activation('relu')(model_x2)
    model_x2 = MaxPooling2D((4,4), strides = (4,4))(model_x2)
    # block 2
    model_x2 = Conv2D(256, (3,3), padding = 'same')(model_x2)
    model_x2 = Activation('relu')(model_x2)
    model_x2 = Conv2D(256, (3,3), padding = 'same')(model_x2)
    model_x2 = Activation('relu')(model_x2)
    model_x2 = MaxPooling2D((2,2), strides = (2,2))(model_x2)
    # block 3
    model_x2 = Conv2D(512, (3,3), padding = 'same')(model_x2)
    model_x2 = Activation('relu')(model_x2)
    model_x2 = Conv2D(512, (3,3), padding = 'same')(model_x2)
    model_x2 = Activation('relu')(model_x2)
    model_x2 = MaxPooling2D((2,2), strides = (2,2))(model_x2)
    # FC 1
    model_fc2 = Flatten()(model_x2)
    
    # define model 3
    model_input_3 = Input(shape=(width, height, num_channels))
    # block 1
    model_x3 = Conv2D(64, (3,3), padding = 'same')(model_input_3)
    model_x3 = Activation('relu')(model_x3)
    model_x3 = Conv2D(64, (3,3), padding = 'same')(model_x3)
    model_x3 = Activation('relu')(model_x3)
    model_x3 = MaxPooling2D((4,4), strides = (4,4))(model_x3)
    # block 2
    model_x3 = Conv2D(256, (3,3), padding = 'same')(model_x3)
    model_x3 = Activation('relu')(model_x3)
    model_x3 = Conv2D(256, (3,3), padding = 'same')(model_x3)
    model_x3 = Activation('relu')(model_x3)
    model_x3 = MaxPooling2D((2,2), strides = (2,2))(model_x3)
    # block 3
    model_x3 = Conv2D(512, (3,3), padding = 'same')(model_x3)
    model_x3 = Activation('relu')(model_x3)
    model_x3 = Conv2D(512, (3,3), padding = 'same')(model_x3)
    model_x3 = Activation('relu')(model_x3)
    model_x3 = MaxPooling2D((2,2), strides = (2,2))(model_x3)
    # FC 1
    model_fc3 = Flatten()(model_x3)
    
    # define model 4
    model_input_4 = Input(shape=(width, height, num_channels))
    # block 1
    model_x4 = Conv2D(64, (3,3), padding = 'same')(model_input_4)
    model_x4 = Activation('relu')(model_x4)
    model_x4 = Conv2D(64, (3,3), padding = 'same')(model_x4)
    model_x4 = Activation('relu')(model_x4)
    model_x4 = MaxPooling2D((4,4), strides = (4,4))(model_x4)
    # block 2
    model_x4 = Conv2D(256, (3,3), padding = 'same')(model_x4)
    model_x4 = Activation('relu')(model_x4)
    model_x4 = Conv2D(256, (3,3), padding = 'same')(model_x4)
    model_x4 = Activation('relu')(model_x4)
    model_x4 = MaxPooling2D((2,2), strides = (2,2))(model_x4)
    # block 3
    model_x4 = Conv2D(512, (3,3), padding = 'same')(model_x4)
    model_x4 = Activation('relu')(model_x4)
    model_x4 = Conv2D(512, (3,3), padding = 'same')(model_x4)
    model_x4 = Activation('relu')(model_x4)
    model_x4 = MaxPooling2D((2,2), strides = (2,2))(model_x4)    
    # FC 1
    model_fc4 = Flatten()(model_x4)
    
    # define model 5
    model_input_5 = Input(shape=(width, height, num_channels))
    # block 1
    model_x5 = Conv2D(64, (3,3), padding = 'same')(model_input_5)
    model_x5 = Activation('relu')(model_x5)
    model_x5 = Conv2D(64, (3,3), padding = 'same')(model_x5)
    model_x5 = Activation('relu')(model_x5)
    model_x5 = MaxPooling2D((4,4), strides = (4,4))(model_x5)
    # block 2
    model_x5 = Conv2D(256, (3,3), padding = 'same')(model_x5)
    model_x5 = Activation('relu')(model_x5)
    model_x5 = Conv2D(256, (3,3), padding = 'same')(model_x5)
    model_x5 = Activation('relu')(model_x5)
    model_x5 = MaxPooling2D((2,2), strides = (2,2))(model_x5)
    # block 3
    model_x5 = Conv2D(512, (3,3), padding = 'same')(model_x5)
    model_x5 = Activation('relu')(model_x5)
    model_x5 = Conv2D(512, (3,3), padding = 'same')(model_x5)
    model_x5 = Activation('relu')(model_x5)
    model_x5 = MaxPooling2D((2,2), strides = (2,2))(model_x5)
    # FC 1
    model_fc5 = Flatten()(model_x5)    
    
    # define model 6
    model_input_6 = Input(shape=(width, height, num_channels))
    # block 1
    model_x6 = Conv2D(64, (3,3), padding = 'same')(model_input_6)
    model_x6 = Activation('relu')(model_x6)
    model_x6 = Conv2D(64, (3,3), padding = 'same')(model_x6)
    model_x6 = Activation('relu')(model_x6)
    model_x6 = MaxPooling2D((4,4), strides = (4,4))(model_x6)
    # block 2
    model_x6 = Conv2D(256, (3,3), padding = 'same')(model_x6)
    model_x6 = Activation('relu')(model_x6)
    model_x6 = Conv2D(256, (3,3), padding = 'same')(model_x6)
    model_x6 = Activation('relu')(model_x6)
    model_x6 = MaxPooling2D((2,2), strides = (2,2))(model_x6)
    # block 3
    model_x6 = Conv2D(512, (3,3), padding = 'same')(model_x6)
    model_x6 = Activation('relu')(model_x6)
    model_x6 = Conv2D(512, (3,3), padding = 'same')(model_x6)
    model_x6 = Activation('relu')(model_x6)
    model_x6 = MaxPooling2D((2,2), strides = (2,2))(model_x6)
    # FC 1
    model_fc6 = Flatten()(model_x6)
    
    # Fully connected layers
    fc_1 = Concatenate()([model_fc1,model_fc2,model_fc3,model_fc4,model_fc5,model_fc6])
    fc_1 = Dense(2048, activation='relu')(fc_1)
    # mean
    mean = Dense(1)(fc_1)
    # uncertainty
    std = Dense(1)(fc_1)
    
    
    model = Model(inputs = [model_input_1,
                            model_input_2,
                            model_input_3,
                            model_input_4,
                            model_input_5,
                            model_input_6                                                      
                           ],                   
                  output = [mean, std])
    return model
    
    
    


# In[3]:


model = VGG_16(128, 128, 3)
model.summary()
keras.utils.plot_model(model, "model.png", show_shapes=True)


# In[7]:


test_dataset = np.zeros([100, 6, 128, 128, 3])
sample = test_dataset[5]

a = np.zeros([1, 128, 128, 3])
model.predict([a, a, a, a, a, a])


# In[ ]:




