from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D, Concatenate
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


def v4_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(512,512,3))) # 1
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 2
    model.add(Dropout(0.5))

    model.add(Flatten()) 
    model.add(Dense(64)) # 3
    model.add(Activation('linear'))
    model.add(Dropout(0.5))
    model.add(Dense(1)) # 4

    model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')

    return model

# 論文で紹介あったモデル
# http://cs229.stanford.edu/proj2017/final-reports/5237321.pdf
# 層数は一緒でバッチ正規化入れてみた
def v5_model():

    model = Sequential()
    model.add(Convolution2D(32, 5, 5, input_shape=(512,512,3))) # 1 # kernel 3->5
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 2
    # model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3)) # 3
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 4
    # model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3)) # 5
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 6
    # model.add(Dropout(0.25))

    model.add(Flatten()) 
    model.add(Dense(512)) # 7 # 128->512
    # model.add(BatchNormalization())
    model.add(Activation('linear'))
    model.add(Dropout(0.5))
    model.add(Dense(1)) # 8

    model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.RMSprop(lr=1e-4))

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
    model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.RMSprop(lr=1e-3))

    return model