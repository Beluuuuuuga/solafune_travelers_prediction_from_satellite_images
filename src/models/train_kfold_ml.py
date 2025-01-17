import sys
import pandas as pd
import numpy as np
import os
import pathlib
import glob
import math
import random
import argparse
import time

import cv2
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor as RFR

from models import v2_model, v3_model, v4_model, vgg16
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, ModelCheckpoint
import tensorflow as tf

from ImageDataAugmentor.image_data_augmentor import *
import albumentations

# 学習率スケジューリング
def step_decay(epoch):
    # initial_lrate = 0.001 # 学習率の初期値
    initial_lrate = 0.01 # 学習率の初期値 ~v9
    # initial_lrate = 0.005 # 学習率の初期値 v11
    drop = 0.5 # 減衰率は50%
    epochs_drop = 10.0 # 10エポックごとに減衰
    # epochs_drop = 5.0 # 10エポックごとに減衰
    # epochs_drop = 10.0 # 10エポックごとに減衰
    # epochs_drop = 15.0 # 10エポックごとに減衰
    lrate = initial_lrate * math.pow(
        drop,
        math.floor((epoch) / epochs_drop)
    )
    return lrate

# 乱数設定
def set_randvalue(value):
    # Set a seed value
    seed_value= value 
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

if __name__ == "__main__":

    # 再現のため乱数を設定
    seed_value = 42
    set_randvalue(seed_value)

    # args = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--makem', help='model name will be maked')
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--kfoldsn', default=3) # default 3
    parser.add_argument('--ttan', default=3) # default 3
    parser.add_argument('--usem', help='select model')
    parser.add_argument('--imgsize', default=512)
    parser.add_argument('--lrateflag', action='store_true', help='learning rate scheduler')
    
    args = parser.parse_args()

    model_name_prefix = args.makem # 保存のモデル名
    epochs = int(args.epochs) # epochs
    KFOLDNUM = args.kfoldsn
    model_name = args.usem
    imgsize = int(args.imgsize) # basemodel:512, vgg16:224
    lrateflag = args.lrateflag
    ttan = int(args.ttan)

    # csv読み込み
    train1 = pd.read_csv('traindataset_anotated_kfold1.csv', names=["image","traveler"]) # headerあり読み込み
    train2 = pd.read_csv('traindataset_anotated_kfold2.csv', names=["image","traveler"]) # headerあり読み込み
    train3 = pd.read_csv('traindataset_anotated_kfold3.csv', names=["image","traveler"]) # headerあり読み込み
    df = [train1, train2, train3]

    kfolds = [
        [[0,1],[2]],
        [[1,2],[0]],
        [[2,0],[1]]
        ]


    target_size = (imgsize, imgsize)
    batch_size = 10

    # 早期終了
    early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

    # Cross Validation
    total_val_loss = []
    summary_upload = pd.read_csv('uploadfile.csv', names=["image","traveler1","traveler2","traveler3","traveler4","inter_traveler"]) # headerあり読み込み
    summary_upload["inter_traveler"] = 0
    # kf = KFold(n_splits=KFOLDNUM, shuffle=True, random_state=None)
    # for i, (k_train, k_test) in enumerate(kf.split(df), 1): # 1からスタート

    for i, kfolds_idxes in enumerate(kfolds, 1):

        train_idxes, valid_idx = kfolds_idxes[0], kfolds_idxes[1][0]
        tr_idx1, tr_idx2 = train_idxes[0], train_idxes[1]
        train = pd.concat([df[tr_idx1], df[tr_idx2]])
        test = df[valid_idx]

        print("Fold",i)
        print(i, train_idxes)

        # train = df.iloc[k_train]
        # test = df.iloc[k_test]

        # for k fold column
        column_name = "traveler" + str(i)
        summary_upload[column_name] = 0

        AUGMENTATIONS = albumentations.Compose([
            # albumentations.Transpose(p=0.5),
            albumentations.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0.5, p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.GaussNoise(mean=100, p=0.5),
            albumentations.Normalize(std=(0.9,0.9,0.9),p=0.2),
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            albumentations.OneOf([
            albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
        ],p=1),
        ])

        # ToDo: 後で関数にする
        train_datagen = ImageDataAugmentor(
            rescale=1./255,
            augment=AUGMENTATIONS, augment_seed=seed_value
            # preprocess_input=None
            # rotation_range=15,
            # shear_range=0.2,
            # horizontal_flip=True,
            # vertical_flip=True,
            # width_shift_range=0.1,
            # height_shift_range=0.1,
            # fill_mode='nearest',
            # zca_whitening=True # ZCA白色化 
        )

        train_datagenerator = train_datagen.flow_from_dataframe(
            train,
            # "data/trainimage/image",
            # "data/mergeimage/image",
            "data/mergecropped1700image/image",
            x_col='image',
            y_col='traveler',
            target_size=target_size,
            class_mode="raw", # for regression
            batch_size=batch_size,
            seed=seed_value
        )

        valid_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagenerator = valid_datagen.flow_from_dataframe(
            test,
            # "data/testimage/image",
            # "data/mergeimage/image",
            "data/mergecropped1700image/image",
            x_col='image',
            y_col='traveler',
            target_size=target_size,
            class_mode="raw", # for regression
            batch_size=batch_size,
            seed=seed_value
        )

        inference_datagen = ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest',
            zca_whitening=True # ZCA白色化 
        )

        # 学習
        # model = vgg16(imgsize)
        model = None
        if model_name == "v2_model":
            model = v2_model()
        elif model_name == "v3_model":
            model = v3_model()
        elif model_name == "v4_model":
            model = v4_model()
        elif model_name == "rmforest":
            model = RFR(n_jobs=-1, random_state=seed_value)

        # train_x, train_y = np.empty(0), np.empty(0)
        train_y = []
        cnt = 0
        train_x = []
        for x, y in train_datagenerator:
            # if cnt > len(train): break
            for idx in range(batch_size):
                if cnt > len(train) - 1: break
                train_x.append(x[idx])
                train_y.append(y[idx])
            # train_x = train_x.append(x.ravel())
            # train_y = np.append(train_y, y)
                cnt += 1
                print(cnt)

        start = time.time()
        model.fit(train_x, train_y)
        end = time.time()
        print("学習完了")
        print(e-s)

        # checkpointの設定
        model_path = 'models/' + model_name_prefix +  '_' + "fold" + str(i) + "_best_model.hdf5"

        # 推論
        upload = pd.read_csv('uploadfile.csv', names=["image","traveler"]) # headerあり読み込み
        evaluate_iter = pathlib.Path('data/evaluatemodel/image').glob('*.jpg')
        
        for evaluate_path in tqdm(evaluate_iter):
            
            _path = str(evaluate_path)
            _path = _path.split('/')[-1]
            evaluate_img = cv2.imread(str(evaluate_path))
            evaluate_img =  cv2.cvtColor(evaluate_img, cv2.COLOR_BGR2RGB)
            evaluate_img =  cv2.resize(evaluate_img, (imgsize, imgsize))
            evaluate_img = np.array(evaluate_img / 255.)
            evaluate_img = evaluate_img.reshape(1, imgsize, imgsize, 3)
            # predict_num = new_model.predict(evaluate_img)[0][0]

            # step数だけTTA
            predictions = []
            cnt = 0
            for _ in range(ttan):
                cnt += 1
                print("TTA step:", cnt)
                preds = model.predict(inference_datagen.flow(evaluate_img, batch_size=1, shuffle=False, seed=seed_value))
                predictions.append(preds)
    
            predict_num = int(np.mean(predictions, axis=0)[0][0])

            upload.loc[upload['image'] == _path, 'traveler'] = int(predict_num)
            summary_upload.loc[upload['image'] == _path, column_name] = int(predict_num)
            summary_upload.loc[upload['image'] == _path, 'inter_traveler'] += int(predict_num) / KFOLDNUM # あらかじめ割ったものを足す

        fold_loss = 999
        sub_csv_path = 'csvs/submit/' + model_name_prefix +  '_' + str(i) + '_' + str(fold_loss) + '.csv'
        upload.to_csv(sub_csv_path, header=False, index=False)

    # val loss
    # string = ""
    # for i, loss in enumerate(total_val_loss,1):
    #     _str = "Fold " + str(i) + ": " + str(loss) + ", "
    #     print(_str)
    #     string += _str
    # print(string)

    summary_upload["traveler"] =  summary_upload["inter_traveler"].apply(lambda x: round(x))
    sub_csv_path = 'csvs/submit/' + model_name_prefix + '_inter' + '.csv'
    summary_upload.to_csv(sub_csv_path, header=True, index=False)

    # ave_loss = sum(total_val_loss)/len(total_val_loss)
    # print("Validation lossの平均値: ", ave_loss)

    # 提出用にカラム削除
    del summary_upload['traveler1']
    del summary_upload['traveler2']
    del summary_upload['traveler3']
    del summary_upload['traveler4']
    del summary_upload['inter_traveler']
    ave_loss = 999
    sub_csv_path = 'csvs/submit/' + model_name_prefix + '_submit' + '_' + str(ave_loss) + '.csv'
    summary_upload.to_csv(sub_csv_path, header=False, index=False)
