import pathlib
import glob

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import xgboost as xgb # 勾配ブースティング
from sklearn.linear_model import LinearRegression
# from sklearn import cross_validation, tree, linear_model

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_DIC = {
    "v18_1":"models/v18_fold2_best_model.hdf5",
    "v18_2":"models/v18_fold3_best_model.hdf5",
    "v18_3":"models/v18_fold4_best_model.hdf5",
    "v19_1":"models/v19_fold2_best_model.hdf5",
    "v19_2":"models/v19_fold3_best_model.hdf5",
    "v19_3":"models/v19_fold4_best_model.hdf5",
    "v22_1":"models/v22_fold2_best_model.hdf5",
    "v22_2":"models/v22_fold3_best_model.hdf5",
    "v22_3":"models/v22_fold4_best_model.hdf5"
}

KFOLDS = [
    [[0,1],[2]],
    [[1,2],[0]],
    [[2,0],[1]]
    ]

NFOLDS = 3


def get_img(test_path, imgsize):

    test_img = cv2.imread(str(test_path))
    test_img =  cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img =  cv2.resize(test_img, (imgsize, imgsize))
    test_img = np.array(test_img / 255.)
    test_img = test_img.reshape(1, imgsize, imgsize, 3)
    return test_img


def get_oof(model_prefix, dfs, upload):
    total_train, total_test = 80, 58
    oof_train = np.zeros((total_train,))
    oof_test_skf = np.empty((NFOLDS, total_test))
    imgsize = 512

    oof_test_cnt = 0
    oof_eval_cnt = 0

    for i, kfolds_idxes in tqdm(enumerate(KFOLDS, 1)):
        train_idxes, valid_idx = kfolds_idxes[0], kfolds_idxes[1][0]
        test = dfs[valid_idx]

        model_version = model_prefix + '_' + str(i)
        model = tf.keras.models.load_model(MODEL_DIC[model_version])

        # validation
        for j in range(test.shape[0]):
            test_path = test.iloc[j]["image"]
            test_path = 'data/mergeimage/image/' + test_path

            test_img = get_img(test_path, imgsize)
            predict_num = model.predict(test_img)[0][0]

            oof_train[oof_test_cnt] = predict_num
            oof_test_cnt += 1
        
        # evaluation
        oof_test = np.zeros((total_test,))
        for k in range(upload.shape[0]):
            evaluate_path = upload.iloc[k]["image"]
            print(evaluate_path)
            evaluate_path = 'data/evaluatemodel/image/' + evaluate_path

            evaluate_img = get_img(evaluate_path, imgsize)
            predict_num = model.predict(evaluate_img)[0][0]
            
            oof_test[k] = predict_num
            
            
        oof_test_skf[oof_eval_cnt, :] = oof_test
        oof_eval_cnt += 1

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

if __name__ == "__main__":
    
    # evaluate_iter = pathlib.Path('models').glob('v19*.hdf5')
    # for path in evaluate_iter:
    #     print(path)



    # csv読み込み
    train1 = pd.read_csv('traindataset_anotated_kfold1.csv', names=["image","traveler"]) # headerあり読み込み
    train2 = pd.read_csv('traindataset_anotated_kfold2.csv', names=["image","traveler"]) # headerあり読み込み
    train3 = pd.read_csv('traindataset_anotated_kfold3.csv', names=["image","traveler"]) # headerあり読み込み
    

    mergedf = pd.concat([train1, train2, train3])
    y_train = mergedf['traveler'].ravel()
    # print(y_train.shape)
    # print(y_train)
    # exit()

    upload = pd.read_csv('uploadfile.csv', names=["image","traveler"]) # headerあり読み込み
    images = upload['image']


    dfs = [train1, train2, train3]
    model_prefix1 = "v18"
    model_prefix2 = "v22"

    oof_train1, oof_test1 = get_oof(model_prefix1, dfs, upload)
    oof_train2, oof_test2 = get_oof(model_prefix2, dfs, upload)

    

    
    oof_train1 = oof_train1.flatten()
    oof_train1 = oof_train1.reshape(80,1)
    oof_train1 = np.array(oof_train1, dtype='int')
    oof_train2 = oof_train2.flatten()
    oof_train2 = oof_train2.reshape(80,1)
    oof_train2 = np.array(oof_train2, dtype='int')

    oof_test1 = np.array(oof_test1, dtype='int')
    oof_test2 = np.array(oof_test2, dtype='int')

    # StackingSubmission = pd.DataFrame({
    #                                     model_prefix1: oof_train1,
    #                                     model_prefix2: oof_train2 })
    # StackingSubmission.to_csv("csvs/submit/v28_StackingSubmission.csv", index=False)
    print('oof_train1.shape : ', oof_train1.shape)
    print('oof_train2.shape : ', oof_train2.shape)
    print('oof_test1.shape : ', oof_test1.shape)
    print('oof_test2.shape : ', oof_test2.shape)
    
    x_train = np.concatenate([oof_train1, oof_train2], axis=1)
    # x_train = np.concatenate(( oof_train1, oof_train1), axis=1)
    x_test = np.concatenate([oof_test1, oof_test2], axis=1)
    # x_test = np.concatenate(( oof_test1, oof_test1), axis=1)
    print('x_train.shape : ', x_train.shape)
    print('x_test.shape : ', x_test.shape)


    # モデルのインスタンス作成
    # mod = xgb.XGBRegressor()
    # mod.fit(x_train, y_train)
    # predictions = mod.predict(x_test)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)

    predictions = np.array(predictions, dtype='int')




    # gbm = xgb.XGBClassifier(
    # #learning_rate = 0.02,
    # n_estimators= 2000,
    # max_depth= 4,
    # min_child_weight= 2,
    # #gamma=1,
    # gamma=0.9,                        
    # subsample=0.8,
    # colsample_bytree=0.8,
    # objective= 'binary:logistic',
    # nthread= -1,
    # scale_pos_weight=1).fit(x_train, y_train)
    # predictions = gbm.predict(x_test)

    # CSVファイルの作成 
    StackingSubmission = pd.DataFrame({ 'image': images,
                                        'traveler': predictions })
    StackingSubmission.to_csv("csvs/submit/v28_submit2.csv", header=False, index=False)



    
    
    # print(oof_train)

    # rows = 0
    # for df in dfs:
    #     rows += df.shape[0]
    #     print(df.shape)

    # print(rows)