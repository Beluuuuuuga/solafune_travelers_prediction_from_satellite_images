from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D

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