# Reference: https://qiita.com/fujin/items/128ed7188f7e7df74f2c

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import feature
import cv2
import math
import random
import os

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression
from skimage.feature import hog
import xgboost as xgb # 勾配ブースティング
import lightgbm as lgbm


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
    # tf.random.set_seed(seed_value)

def get_img(mode, path, df, prepro = None):
    images = []
    # resize = 100
    # resize = 200
    # resize = 300
    # resize = 400
    # resize = 500
    resize = 600
    # cropsize = 600
    # cropsize = 800
    # cropsize = 1000
    
    if prepro == "pca":
        print("PCA")
        prepro = PCA(0.40)
    
    if mode == 'train' or mode == 'test':
        for i in df.ravel():
            
            imgpath_ = path + i
            img = cv2.imread(str(imgpath_)) # color
            # img = cv2.imread(str(imgpath_),0) # gray

            img =  cv2.resize(img, (resize, resize))
            img = np.array(img / 255.)
            img = np.ravel(img)
            images.append(img)
        
        return images
    else:
        for i in df.ravel():
            imgpath_ = path + i
            
            img = cv2.imread(str(imgpath_)) # color
            # img = cv2.imread(str(imgpath_),0) # gray

            img =  cv2.resize(img, (resize, resize))
            img = np.array(img / 255.)
            img = np.ravel(img)
            images.append(img)
        
        return images


if __name__ == "__main__":

    # 再現のため乱数を設定
    seed_value = 42
    set_randvalue(seed_value)

    # traindataset_anotated_hazuretinuki
    # train = pd.read_csv('csvs/train/traindataset_anotated_kfold1.csv', names=["image","traveler"]) # headerあり読み込み
    train = pd.read_csv('csvs/train/traindataset_anotated_kfold2.csv', names=["image","traveler"]) # headerあり読み込み
    # train = pd.read_csv('csvs/train/traindataset_anotated_kfold3.csv', names=["image","traveler"]) # headerあり読み込み
    # train = pd.read_csv("csvs/train/traindataset_anotated.csv", names=["image","traveler"])
    # train = pd.read_csv("csvs/train/train_merge_2.csv", names=["image","traveler"])
    # train = pd.read_csv("csvs/train/traindataset_anotated_2.csv", names=["image","traveler"])
    # train = pd.read_csv("csvs/train/traindataset_anotated_3.csv", names=["image","traveler"])
    # test = pd.read_csv("csvs/train/testdataset_anotated.csv", names=["image","traveler"])
    evaluation = pd.read_csv("csvs/train/uploadfile.csv", names=["image","traveler"])

    train["traveler2"] =  train["traveler"] / 10000
    train_traveler = train["traveler2"].ravel()
    # test["traveler2"] =  test["traveler"] / 10000
    # test_traveler = test["traveler2"].ravel()

    # pcaなし
    # train_images = get_img('train', "data/mergecropped600image/image/", train["image"]) # 600
    train_images = get_img('train', "data/mergecropped600imageHSV/image/", train["image"]) # 600
    # test_images = get_img('test', "data/mergecropped600image/image/", test["image"]) # 600
    # evaluate_images = get_img('eval', "data/evaluatemodel/image/", evaluation["image"]) # 600
    evaluate_images = get_img('eval', "data/evaluatemodelcropped600HSV/image/", evaluation["image"]) # 600
    # train_images = get_img('train', "data/mergecropped800image/image/", train["image"]) # 800
    # test_images = get_img('test', "data/mergecropped800image/image/", test["image"]) # 800
    # evaluate_images = get_img('eval', "data/evaluatemodel/image/", evaluation["image"]) # 800
    # train_images = get_img('train', "data/mergecropped800image/image/", train["image"]) # 1000
    # test_images = get_img('test', "data/mergecropped800image/image/", test["image"]) # 1000
    # evaluate_images = get_img('eval', "data/evaluatemodel/image/", evaluation["image"]) # 1000

    # xgboost
    train_images = np.array(train_images) # on
    # test_images = np.array(test_images)
    evaluate_images = np.array(evaluate_images) # on

    # model = RFR(n_jobs=-1, random_state=seed_value) # randomforest
    model = xgb.XGBRegressor() # xgboost # on
    # model = LinearRegression() # 線形回帰
    model.fit(train_images, train_traveler) # on

    predictions = model.predict(evaluate_images) # on

    for i in predictions:
        print(int(i*10000))
