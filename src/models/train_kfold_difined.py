import sys
import pandas as pd
import numpy as np
import os
import pathlib
import glob
import math
import random

import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from models import v2_model, vgg16
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback, ModelCheckpoint
import tensorflow as tf

# 学習率スケジューリング
def step_decay(epoch):
    # initial_lrate = 0.001 # 学習率の初期値
    initial_lrate = 0.01 # 学習率の初期値 ~v9
    # initial_lrate = 0.005 # 学習率の初期値 v11
    drop = 0.5 # 減衰率は50%
    # epochs_drop = 10.0 # 10エポックごとに減衰
    epochs_drop = 10.0 # 10エポックごとに減衰
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

    args = sys.argv
    model_name_prefix = args[1] # 保存のモデル名
    epochs = int(args[2]) # epochs
    lrate_flg = args[3] # lrate_True or lrate_False
    KFOLDNUM = int(args[4])

    # imgsize = 224 # 画像サイズ 後で動的にする
    imgsize = 512 # 画像サイズ 後で動的にする


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

        # ToDo: 後で関数にする
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
            # "data/trainimage/image",
            "data/mergeimage/image",
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
            "data/mergeimage/image",
            x_col='image',
            y_col='traveler',
            target_size=target_size,
            class_mode="raw", # for regression
            batch_size=batch_size,
            seed=seed_value
        )

        # 学習
        model = v2_model()
        # model = vgg16(imgsize)

        # checkpointの設定
        model_path = 'models/' + model_name_prefix +  '_' + "fold" + str(i+1) + "_best_model.hdf5"
        checkpoint = ModelCheckpoint(
                    filepath=model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    period=1)

        # 動的学習率変化
        history = None
        if lrate_flg == "lrate_True":
            lrate = LearningRateScheduler(step_decay)
            history = model.fit(train_datagenerator,
        #                     steps_per_epoch=int(total_train//batch_size),
                            epochs=epochs,
                            # epochs=1,
                            validation_data=valid_datagenerator,
        #                     validation_steps=int(total_valid//batch_size),
                            verbose=1,
                            shuffle=True,
                            callbacks=[early_stop, lrate, checkpoint])
        elif lrate_flg == "lrate_False":
            history = model.fit(train_datagenerator,
        #                     steps_per_epoch=int(total_train//batch_size),
                            epochs=epochs,
                            validation_data=valid_datagenerator,
        #                     validation_steps=int(total_valid//batch_size),
                            verbose=1,
                            shuffle=True,
                            callbacks=[early_stop, checkpoint])

        # 結果保存
        # model_path = 'models/' + model_name_prefix +  '_' + str(i) + '_model.h5'
        # model.save(model_path)
        # 最もlossが小さいモデルをロード
        new_model = tf.keras.models.load_model(model_path)

        hist_df = pd.DataFrame(history.history)
        csv_his_path = 'csvs/history/' +  model_name_prefix +  '_' + str(i) + '_history.csv'
        hist_df.to_csv(csv_his_path)

        plt.figure()
        hist_df[['loss', 'val_loss']].plot()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        fig_path = 'reports/' + model_name_prefix +  '_' + str(i) + '_loss.png'    
        plt.savefig(fig_path)
        plt.close()

        # val loss 追加
        fold_loss = min(history.history['val_loss'])
        total_val_loss.append(fold_loss) # 最小値を取得

        # 推論
        upload = pd.read_csv('uploadfile.csv', names=["image","traveler"]) # headerあり読み込み
        evaluate_iter = pathlib.Path('data/evaluatemodel/image').glob('*.jpg')
        for evaluate_path in evaluate_iter:
            _path = str(evaluate_path)
            _path = _path.split('/')[-1]
            evaluate_img = cv2.imread(str(evaluate_path))
            evaluate_img =  cv2.cvtColor(evaluate_img, cv2.COLOR_BGR2RGB)
            evaluate_img =  cv2.resize(evaluate_img, (imgsize, imgsize))
            evaluate_img = np.array(evaluate_img / 255.)
            evaluate_img = evaluate_img.reshape(1, imgsize, imgsize, 3)
            predict_num = new_model.predict(evaluate_img)[0][0]

            upload.loc[upload['image'] == _path, 'traveler'] = int(predict_num)
            summary_upload.loc[upload['image'] == _path, column_name] = int(predict_num)
            summary_upload.loc[upload['image'] == _path, 'inter_traveler'] += int(predict_num) / KFOLDNUM # あらかじめ割ったものを足す

        sub_csv_path = 'csvs/submit/' + model_name_prefix +  '_' + str(i) + '_' + str(fold_loss) + '.csv'
        upload.to_csv(sub_csv_path, header=False, index=False)

    # val loss
    string = ""
    for i, loss in enumerate(total_val_loss,1):
        _str = "Fold " + str(i) + ": " + str(loss) + ", "
        print(_str)
        string += _str
    print(string)

    summary_upload["traveler"] =  summary_upload["inter_traveler"].apply(lambda x: round(x))
    sub_csv_path = 'csvs/submit/' + model_name_prefix + '_inter' + '.csv'
    summary_upload.to_csv(sub_csv_path, header=True, index=False)

    ave_loss = sum(total_val_loss)/len(total_val_loss)
    print("Validation lossの平均値: ", ave_loss)

    # 提出用にカラム削除
    del summary_upload['traveler1']
    del summary_upload['traveler2']
    del summary_upload['traveler3']
    del summary_upload['traveler4']
    del summary_upload['inter_traveler']
    sub_csv_path = 'csvs/submit/' + model_name_prefix + '_submit' + '_' + str(ave_loss) + '.csv'
    summary_upload.to_csv(sub_csv_path, header=False, index=False)
