
import sys
import os
import pathlib
import glob
import math
import random
import argparse
import time

import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, ModelCheckpoint
import tensorflow as tf


def vgg16(imgsize):
    
    input_tensor = (imgsize,imgsize, 3)
    base_model_max = VGG16(input_shape=input_tensor, weights='imagenet', include_top=False, pooling='max')

    for layer in base_model_max.layers[:15]:
        # from 1st to 15th freeze
        layer.trainable = False
    
    for layer in base_model_max.layers[15:]:
        # from 16th activate
        layer.trainable = True

    model = Sequential()
    model.add(base_model_max)
    model.add(Flatten())

    model.add(Dense(512)) # 全結合層
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64)) # 全結合層
    model.add(Activation('linear'))
    model.add(Dropout(0.5))

    model.add(Dense(1)) # 出力層
    model.compile(loss='mean_squared_logarithmic_error', optimizer=tf.keras.optimizers.RMSprop(lr=1e-5))

    return model

def get_datagen(mode, df, target_size, batch_size):
    if mode == 'train':
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            shear_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )

        train_datagenerator = train_datagen.flow_from_dataframe(
            df,
            "data/trainimage/image",
            x_col='image',
            y_col='traveler',
            target_size=target_size,
            class_mode="raw", # for regression
            batch_size=batch_size,
            seed=42
        )
    
        return train_datagenerator

    elif mode == 'valid':
        valid_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagenerator = valid_datagen.flow_from_dataframe(
            df,
            "data/testimage/image",
            x_col='image',
            y_col='traveler',
            target_size=target_size,
            class_mode="raw", # for regression
            batch_size=batch_size,
            seed=42
        )

        return valid_datagenerator


if __name__ == "__main__":

    # csv読み込み
    train = pd.read_csv('csvs/train/traindataset_anotated.csv', names=["image","traveler"]) # headerあり読み込み
    valid = pd.read_csv('csvs/train/testdataset_anotated.csv', names=["image","traveler"]) # headerあり読み込み

    # ハイパーパラメータ
    batch_size=10
    epochs = 200
    imgsize = 224
    target_size = (imgsize, imgsize)

    # データセット取得
    train_datagenerator = get_datagen("train", train, target_size, batch_size)
    valid_datagenerator = get_datagen("valid", valid, target_size, batch_size)
    
    # 早期終了
    early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto') # val_loss
    
    # モデル読み込み
    model = vgg16(imgsize)

    # 学習
    history = model.fit(train_datagenerator,
                    epochs=epochs,
                    validation_data=valid_datagenerator,
                    verbose=1,
                    shuffle=True,
                    callbacks=[early_stop])


    # モデル保存
    model.save("solafune_tutorial.h5")

    # 推論
    upload = pd.read_csv('csvs/train/uploadfile.csv', names=["image","traveler"]) # headerあり読み込み
    evaluate_iter = pathlib.Path('data/evaluatemodel/image').glob('*.jpg')
    
    for evaluate_path in tqdm(evaluate_iter):
        _path = str(evaluate_path)
        _path = _path.split('/')[-1]

        # 推論前処理部分
        evaluate_img = cv2.imread(str(evaluate_path))
        evaluate_img =  cv2.cvtColor(evaluate_img, cv2.COLOR_BGR2RGB)
        evaluate_img =  cv2.resize(evaluate_img, (imgsize, imgsize))
        evaluate_img = np.array(evaluate_img / 255.)
        evaluate_img = evaluate_img.reshape(1, imgsize, imgsize, 3)

        predict_num = int(model.predict(evaluate_img))
        upload.loc[upload['image'] == _path, 'traveler'] = int(predict_num)

    upload.to_csv("submit_solafune_tutorial.csv", header=False, index=False)
