import pandas as pd
import os
import sys

if __name__ == "__main__":
    df1 = pd.read_csv("csvs/submit/v9_submit.csv", header=None)
    # df2 = pd.read_csv("csvs/submit/v13_submit.csv", header=None)
    df3 = pd.read_csv("csvs/submit/v15_submit.csv", header=None)

    print(df1.head())
    # df1.iloc[:,1:] += df2.iloc[:,1:]
    df1.iloc[:,1:] += df3.iloc[:,1:]
    
    new_col = [x//2 for x in df1.loc[:,1].to_list()]
    s = pd.Series(new_col)

    del df1[1]
    df1[1] = s
    
    # model_name_prefix = ''
    sub_csv_path = 'csvs/submit/v17_submit.csv'
    df1.to_csv(sub_csv_path, header=False, index=False)



    