import numpy as np
import os, shutil
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
from map_boxes import mean_average_precision_for_boxes
import argparse
from sub2zft import sub2zft


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-csv', type=str, default='csv/submission.csv', help='path of submission.csv')
    parser.add_argument('--gt-csv', type=str, default='csv/gt.csv', help='path of gt.csv')
    opt = parser.parse_args()

    SUB_PATH = opt.sub_csv
    GT_PATH   = opt.gt_csv

    gt_df  = pd.read_csv(GT_PATH)
    gt_df  = gt_df[(gt_df[["XMin","XMax","YMin","YMax"]]>=0).all(axis=1)]
    gt     = gt_df[["ImageID","LabelName","XMin","XMax","YMin","YMax"]].values

    sub = pd.read_csv(SUB_PATH)
    print('\nconverting sub to pred:')
    pred_df = sub2zft(sub)
    # pred_df.to_csv('csv/tmp_pred.csv',index=False)
    pred_df = pd.merge(pred_df, gt_df[['ImageID','Width','Height']].drop_duplicates(), on='ImageID', how='left')
    pred_df[["XMin", "YMin"]] = pred_df[["XMin", "YMin"]].values/pred_df[["Width", "Height"]].values
    pred_df[["XMax", "YMax"]] = pred_df[["XMax", "YMax"]].values/pred_df[["Width", "Height"]].values
    # def fix_study(row):
    #     if 'study' in row['ImageID']:
    #         row['Conf'], row['XMin'], row['XMax'], row['YMin'], row['YMax'] = 1, 0, 1, 0, 1
    #     if row['LabelName']=='none':
    #         row['XMin'], row['XMax'], row['YMin'], row['YMax'] = 0, 1, 0, 1
    #     return row
    # pred_df = pred_df.apply(fix_study, axis=1)
    pred    = pred_df[["ImageID","LabelName","Conf","XMin","XMax","YMin","YMax"]].values
    pred_df.to_csv('csv/pred.csv', index=False)

    print('\ncalculating mAP:')
    _ = mean_average_precision_for_boxes(gt, pred, 
    exclude_not_in_annotations=True, 
    verbose=True)