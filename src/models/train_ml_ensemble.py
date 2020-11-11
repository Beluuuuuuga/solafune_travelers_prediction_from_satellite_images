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

def bgr2hog(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_image = hog(img, orientations=8, pixels_per_cell=(20, 20),
                    cells_per_block=(2, 2), visualize=False, block_norm="L2", transform_sqrt=False)
    return hog_image

def center_crop(mode, img, Lx, Ly):
    x, y = None, None
    if mode == 'gray':
        x, y = img.shape
    else:
        x, y, _ = img.shape
    startx = x//2 - (Lx//2)
    starty = y//2 - (Ly//2)    
    return img[startx:startx+Lx, starty:starty+Ly]

def get_img(mode, path, df):
    images = []
    # resize = 100
    # resize = 200
    resize = 300
    # resize = 400
    # resize = 500
    # resize = 600
    # cropsize = 600
    # cropsize = 800
    # cropsize = 1000
    
    if mode == 'train' or mode == 'test':
        for i in df.ravel():
            imgpath_ = path + i
            # img = cv2.imread(str(imgpath_)) # color
            img = cv2.imread(str(imgpath_),0) # gray
            img =  cv2.resize(img, (resize, resize))
            img = np.array(img / 255.)
            images.append(img)
        
        return images
    else:
        for i in df.ravel():
            imgpath_ = path + i
            # img = cv2.imread(str(imgpath_)) # color
            img = cv2.imread(str(imgpath_),0) # gray
            img =  cv2.resize(img, (resize, resize))
            img = np.array(img / 255.)
            images.append(img)
        
        return images



if __name__ == "__main__":

    # 再現のため乱数を設定
    seed_value = 42
    set_randvalue(seed_value)

    evaluation = pd.read_csv("csvs/train/uploadfile.csv", names=["image","traveler"])

    train = pd.read_csv("csvs/train/knn/train_merge_knn_28.csv", names=["image","traveler"])

    train["traveler2"] =  train["traveler"] / 10000
    train_traveler = train["traveler2"].ravel()

    # pcaなし
    # train_images = get_img('train', "data/mergecropped600image/image/", train["image"]) # 600
    train_images = get_img('train', "data/mergecropped600imageHSV/image/", train["image"]) # 600
    # evaluate_images = get_img('eval', "data/evaluatemodel/image/", evaluation["image"]) # 600
    evaluate_images = get_img('eval', "data/evaluatemodelcropped600HSV/image/", evaluation["image"]) # 600
    # train_images = get_img('train', "data/mergecropped800image/image/", train["image"]) # 800
    # evaluate_images = get_img('eval', "data/evaluatemodel/image/", evaluation["image"]) # 800
    # train_images = get_img('train', "data/mergecropped800image/image/", train["image"]) # 1000
    # evaluate_images = get_img('eval', "data/evaluatemodel/image/", evaluation["image"]) # 1000

    # SVR
    # svr = SVR() # on
    # svr.fit(train_images, train_traveler) # on
    # pred = svr.predict(test_images)
    # score = r2_score(pred, test_traveler)
    # predictions = svr.predict(evaluate_images) # on

    # xgboost
    train_images = np.array(train_images) # on
    evaluate_images = np.array(evaluate_images) # on

    model = xgb.XGBRegressor() # xgboost # on
    model.fit(train_images, train_traveler) # on

    predictions = model.predict(evaluate_images) # on

    for i in predictions:
        print(int(i*10000))


