import sys
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

from models import v2_model
from tensorflow.keras.callbacks import EarlyStopping

if __name__ == "__main__":

    args = sys.argv
    model_name_prefix = args[1] # 保存のモデル名

    # csv読み込み
    train = pd.read_csv('traindataset_anotated.csv', names=["image","traveler"]) # headerあり読み込み
    test = pd.read_csv('testdataset_anotated.csv', names=["image","traveler"]) # headerあり読み込み

    target_size = (512, 512)
    batch_size = 4

    train_datagen = ImageDataGenerator(rescale=1./255)
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

    history = model.fit(train_datagenerator,
#                     steps_per_epoch=int(total_train//batch_size),
                    # epochs=200,
                    epochs=1,
                    validation_data=valid_datagenerator,
#                     validation_steps=int(total_valid//batch_size),
                    verbose=1,
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