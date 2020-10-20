import sys
import pandas as pd
import numpy as np
import os
import pathlib
import glob
import math

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from models import v2_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback


# 学習率スケジューリング
def step_decay(epoch):
    # initial_lrate = 0.001 # 学習率の初期値
    initial_lrate = 0.01 # 学習率の初期値
    drop = 0.5 # 減衰率は50%
    # epochs_drop = 10.0 # 10エポックごとに減衰
    epochs_drop = 10.0 # 10エポックごとに減衰
    lrate = initial_lrate * math.pow(
        drop,
        math.floor((epoch) / epochs_drop)
    )
    return lrate


if __name__ == "__main__":

    args = sys.argv
    model_name_prefix = args[1] # 保存のモデル名
    epochs = int(args[2]) # epochs
    lrate_flg = args[3] # True or False
    log_flg = args[4] # True or False

    # csv読み込み
    train = pd.read_csv('traindataset_anotated.csv', names=["image","traveler"]) # headerあり読み込み
    train_nomarlized = train.copy()
    train_nomarlized['image'] = train['image']
    train_nomarlized["traveler"] = np.log1p(train["traveler"])

    test = pd.read_csv('testdataset_anotated.csv', names=["image","traveler"]) # headerあり読み込み
    test_nomarlized = test.copy()
    test_nomarlized['image'] = test['image']
    test_nomarlized["traveler"] = np.log1p(test["traveler"])

    if log_flg == "log_True":
        train = train_nomarlized
        test = test_nomarlized
    else: pass

    target_size = (512, 512)
    batch_size = 4

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        shear_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zca_whitening=True # ZCA白色化 
    )

    train_datagenerator = train_datagen.flow_from_dataframe(
        train,
        "data/trainimage/image",
        x_col='image',
        y_col='traveler',
        target_size=target_size,
        class_mode="raw" # for regression
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagenerator = valid_datagen.flow_from_dataframe(
        test,
        "data/testimage/image",
        x_col='image',
        y_col='traveler',
        target_size=target_size,
        class_mode="raw" # for regression
    )

    # 学習
    model = v2_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')

    # 動的学習率変化
    history = None
    if lrate_flg == "lrate_True":
        lrate = LearningRateScheduler(step_decay)
        history = model.fit(train_datagenerator,
    #                     steps_per_epoch=int(total_train//batch_size),
                        epochs=epochs,
                        validation_data=valid_datagenerator,
    #                     validation_steps=int(total_valid//batch_size),
                        verbose=1,
                        shuffle=True,
                        callbacks=[early_stop, lrate])
    elif lrate_flg == "lrate_False":
        history = model.fit(train_datagenerator,
    #                     steps_per_epoch=int(total_train//batch_size),
                        epochs=epochs,
                        validation_data=valid_datagenerator,
    #                     validation_steps=int(total_valid//batch_size),
                        verbose=1,
                        shuffle=True,
                        callbacks=[early_stop])

    # 結果保存
    model_path = 'models/' + model_name_prefix + '_model.h5'
    model.save(model_path)

    hist_df = pd.DataFrame(history.history)
    csv_his_path = 'csvs/history/' +  model_name_prefix + '_history.csv'
    hist_df.to_csv(csv_his_path)

    plt.figure()
    hist_df[['loss', 'val_loss']].plot()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    fig_path = 'reports/' + model_name_prefix + '_loss.png'    
    plt.savefig(fig_path)
    plt.close()

    # 推論
    upload = pd.read_csv('uploadfile.csv', names=["image","traveler"]) # headerあり読み込み
    evaluate_iter = pathlib.Path('data/evaluatemodel/image').glob('*.jpg')
    for evaluate_path in evaluate_iter:
        _path = str(evaluate_path)
        _path = _path.split('/')[-1]
        evaluate_img = cv2.imread(str(evaluate_path))
        evaluate_img =  cv2.cvtColor(evaluate_img, cv2.COLOR_BGR2RGB)
        evaluate_img =  cv2.resize(evaluate_img, (512, 512))
        evaluate_img = np.array(evaluate_img / 255.)
        evaluate_img = evaluate_img.reshape(1, 512, 512, 3)
        train_y_log1p = model.predict(evaluate_img)[0][0]

        # 対数変換を元に戻す
        if log_flg == "log_True":
            train_y_log1p = np.exp(train_y_log1p) - 1

        upload.loc[upload['image'] == _path, 'traveler'] = int(train_y_log1p)
    
    sub_csv_path = 'csvs/submit/' + model_name_prefix + '.csv'
    upload.to_csv(sub_csv_path, header=False, index=False)