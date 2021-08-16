import numpy as np
import os, shutil
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

import argparse

def get_predstring(row):
    string = [row['LabelName'], row['Conf'], row['XMin'], row['YMin'], row['XMax'], row['YMax']]
    row['PredictionString'] = ' '.join(string)
    return row

def zft2sub(df):
    """ Converts zfturbo format to submission.csv format.
    Args:
        df : DataFrame (zfturbo formatted csv)
    Returns:
        DataFrame: submission.csv
    """
    df[["XMin","XMax","YMin","YMax"]] = df[["XMin","XMax","YMin","YMax"]].astype(int)
    df = df.astype(str)
    tqdm.pandas(desc='zft2sub ', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    df = df.progress_apply(get_predstring, axis=1)
    sub_df = pd.DataFrame(df.groupby('ImageID')["PredictionString"].apply(list).
    apply(lambda x: ' '.join(x)).reset_index(drop=False)).rename({"ImageID":"id"}, axis=1)
    return sub_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-path', type=str, default='csv/zft.csv', help='path of submission.csv')
    parser.add_argument('--save-path', type=str, default='', help='save path')
    opt = parser.parse_args()

    CSV_PATH = opt.csv_path

    zft_df = pd.read_csv(CSV_PATH)
    
    # idx = sub_df.id.map(lambda x: True if 'image' in x else False)
    # sub_df = sub_df.loc[idx]

    sub_df = zft2sub(zft_df)
    if len(opt.save_path):
        sub_df.to_csv(opt.save_path, index=False)
    print();print(sub_df.head(5))




    
