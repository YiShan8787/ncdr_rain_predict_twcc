# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:45:35 2021

@author: user
"""

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import concatenate

import visualkeras

weather_frames, weather_channels, station_frames, station_channels, rows, columns = 4,3, 24, 3,210,340

#encode model

weather_video = Input(shape=(weather_frames,
                     rows,
                     columns,
                     weather_channels))

temp_video = Input(shape=(station_frames,
                     rows,
                     columns,
                     station_channels))

huminity_video = Input(shape=(station_frames,
                     rows,
                     columns,
                     station_channels))

#vgg model

vgg_weather = VGG16(input_shape=(rows,
                              columns,
                              weather_channels),
                 weights="imagenet",
                 include_top=False)
vgg_weather.trainable = False

vgg_temp = VGG16(input_shape=(rows,
                              columns,
                              station_channels),
                 weights="imagenet",
                 include_top=False)
vgg_temp.trainable = False

vgg_huminity = VGG16(input_shape=(rows,
                              columns,
                              station_channels),
                 weights="imagenet",
                 include_top=False)
vgg_huminity.trainable = False

#cnn out

cnn_out_weather = GlobalAveragePooling2D()(vgg_weather.output)

cnn_out_temp = GlobalAveragePooling2D()(vgg_temp.output)

cnn_out_huminity = GlobalAveragePooling2D()(vgg_huminity.output)

#cnn model 

cnn_weather = Model(vgg_weather.input, cnn_out_weather)

cnn_temp = Model(vgg_temp.input, cnn_out_temp)

cnn_huminity = Model(vgg_huminity.input, cnn_out_huminity)

#encode frame

weather_encoded_frames = TimeDistributed(cnn_weather)(weather_video)

temp_encoded_frames = TimeDistributed(cnn_temp)(temp_video)

huminity_encoded_frames = TimeDistributed(cnn_huminity)(huminity_video)

# LSTM

weather_encoded_sequence = LSTM(256)(weather_encoded_frames)

temp_encoded_sequence = LSTM(256)(temp_encoded_frames)

huminity_encoded_sequence = LSTM(256)(huminity_encoded_frames)

#concate

encoded_sequence = concatenate([weather_encoded_sequence, temp_encoded_sequence, huminity_encoded_sequence])

# dense layer

hidden_layer = Dense(1024, activation="relu")(encoded_sequence)
outputs = Dense(2, activation="softmax")(hidden_layer)

#test_hidden = Dense(1024, activation="relu")(weather_encoded_sequence)
#test_out = Dense(2, activation="softmax")(test_hidden)

# build all model

model = Model(inputs = [weather_video, temp_video, huminity_video], outputs= [outputs])
#model = Model(inputs = huminity_video, outputs= test_out)
model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')

#visualkeras.layered_view(model).show() # display using your system viewer
#visualkeras.layered_view(model, to_file='output.png') # write to disk
#visualkeras.layered_view(model, to_file='output.png').show() # write and show

#main_input = Input(shape=(100,),dtype='int32',name='main_input')
#auxiliary_input = Input(shape=(5,),name='aux_input')
#test=[weather_video, temp_video, huminity_video]
#print(test)