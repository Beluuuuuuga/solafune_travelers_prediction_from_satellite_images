from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Input


def v2_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(512,512,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten()) 
    model.add(Dense(64))
    model.add(Activation('linear'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    sgd = SGD(lr=0.01, decay=1e-7, momentum=.9)
        
    # model.compile(loss='mse', optimizer='rmsprop')
    model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')

    return model


def v3_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(512,512,3))) # 1
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 2
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3)) # 3
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 4
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3)) # 5
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 6
    model.add(Dropout(0.25))

    model.add(Flatten()) 
    model.add(Dense(64)) # 7
    model.add(Activation('linear'))
    model.add(Dropout(0.5))
    model.add(Dense(1)) # 8

    model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')

    return model


def vgg16(imgsize):
    
    input_tensor = (imgsize,imgsize, 3)
    # base_model_max = VGG16(input_shape=input_tensor, weights=None, include_top=False, pooling='max')
    base_model_max = VGG16(input_shape=input_tensor, weights='imagenet', include_top=False, pooling='max')

    for layer in base_model_max.layers[:15]:
        # from 1st to 15th freeze
        layer.trainable = False
    
    for layer in base_model_max.layers[15:]:
        # from 16th activate
        layer.trainable = True

    # for layer in base_model_max.layers:
    #     # from 16th activate
    #     layer.trainable = True
    
    model = Sequential()
    model.add(base_model_max)
    model.add(Flatten())
    # model.add(GlobalMaxPooling2D()) # 4 tensor to 2 tensor(matrix)

    model.add(Dense(512)) # 全結合層
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64)) # 全結合層
    model.add(Activation('linear'))
    model.add(Dropout(0.5))

    model.add(Dense(1)) # 出力層
    # model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')
    model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.RMSprop(lr=1e-5))

    return model