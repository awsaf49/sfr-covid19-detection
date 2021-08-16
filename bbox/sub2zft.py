import numpy as np
import os, shutil
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

import argparse

def sub2zft(df):
    """ Converts submission.csv to zfturbo format.

    Args:
        df : DataFrame (submission.csv)
    Returns:
        DataFrame: zfturbo formtted csv
    """
    data = []
    # print('converting:')
    for idx in tqdm(range(df.shape[0]), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',desc='sub2zft '):
        row = df.iloc[idx]
        image_id = row["id"]
        strings = np.array(row["PredictionString"].strip().split(' ')).reshape(-1, 6)
        for string in strings:
            data.append([image_id]+string.tolist())
    pred_df = pd.DataFrame(data, columns=["ImageID","LabelName","Conf","XMin","YMin","XMax","YMax"])
    zft_df = pred_df[["ImageID","LabelName","Conf","XMin","XMax","YMin","YMax"]]
    zft_df[["Conf","XMin","XMax","YMin","YMax"]] = zft_df[["Conf","XMin","XMax","YMin","YMax"]].astype(np.float64)
    return zft_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, default='csv/submission.csv', help='path of submission.csv')
    parser.add_argument('--save-path', type=str, default='csv/zft.csv', help='save path')
    opt = parser.parse_args()

    CSV_PATH = opt.csv_path

    sub_df = pd.read_csv(CSV_PATH)
    
    # idx = sub_df.id.map(lambda x: True if 'image' in x else False)
    # sub_df = sub_df.loc[idx]

    zft_df = sub2zft(sub_df)
    if len(opt.save_path):
        zft_df.to_csv(opt.save_path, index=False)
    print();print(zft_df.head(5))




    
