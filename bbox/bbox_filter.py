import numpy as np
import os, shutil
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
import argparse
from sub2zft import sub2zft
from zft2sub import zft2sub
from functools import partial
import json

class BboxFilter:
    """takes bboxes as input and filters out un-realistic bboxes"""
    def __init__(self, feature_cols=[
                                     'x_min', 'x_max','y_min', 'y_max',
                                     'x_center', 'y_center', 
                                     'w', 'h', 
                                     'ar', 
                                     'area',
                                     ]):
        self.zft_cols      = ['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax']
        self.feature_cols  = feature_cols
        self.filter_params = None

    def get_features(self, df, normalize=True):
        """takes dataframe and returns df with features"""
        df = df.copy()
        df[['x_min', 'y_min']]       = df[['XMin', 'YMin']]
        df[['x_max', 'y_max']]       = df[['XMax', 'YMax']]
        if normalize:
            df[['x_min', 'y_min']]   /= df[['Width', 'Height']].values
            df[['x_max', 'y_max']]   /= df[['Width', 'Height']].values
        df[['x_center', 'y_center']] = (df[['x_min', 'y_min']].values + df[['x_max', 'y_max']].values)/2
        df[['w', 'h']]               = df[['x_max', 'y_max']].values - df[['x_min', 'y_min']].values
        df['area']                   = df['w'] * df['h']
        df['ar']                     = df['w'] / df['h']
        return df

    def fit(self, df, normalize=True, save=False):
        """fits dataframe to find filter params"""
        print(f'\nFitting bbox filter with {len(df)} rows')
        df = df.query("LabelName=='opacity'")
        df = df[(df[['XMin', 'XMax', 'YMin', 'YMax']] >= 0).all(axis=1)] # remove negative bboxes (! there is one in train)
        df = df[['XMin', 'XMax', 'YMin', 'YMax']].astype(float)
        df = self.get_features(df, normalize)
        # df.to_csv('csv/feature.csv',index=False)
        filter_params = df[self.feature_cols].agg([min, max]).T.values
        filter_params = dict(zip(self.feature_cols, filter_params.tolist()))
        self.filter_params = filter_params
        print('filter params:\n',filter_params);print()
        if save:
            with open('json/filter_params.json', 'w') as f:
                json.dump(filter_params, f)
        return

    def filter_one(self, row, func):
        """filter one row"""
        if self.filter_params is None:
            raise ValueError('Filter not fitted yet')
        features = row[self.feature_cols].values
        filter_params = np.array([self.filter_params[f] for f in self.feature_cols])
        check    = (filter_params - np.repeat(features[:, None], 2, axis=1)) # value needs to be middle of min and max
        check    = np.prod(check, axis=1)
        row['filter'] = func(check<=0) # (+)*(-) =(-1) so within limit
        return row

    def filter(self, sub_df, func=np.all):
        """filter submission.csv dataframe"""
        zft_df = sub2zft(sub_df)
        zft_df = pd.merge(zft_df, sub_df[['id', 'Width', 'Height']].drop_duplicates(),
                         left_on='ImageID', right_on='id', how='left')

        bbox_idx = zft_df.LabelName=='opacity'
        bbox_df  = zft_df.loc[bbox_idx]

        bbox_df = self.get_features(bbox_df)

        tqdm.pandas(desc='filtering ',  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        bbox_df   = bbox_df.progress_apply(partial(self.filter_one, func=func), axis=1)
        print('\n\t== Summary ==')
        print('-'*40)
        s0 = "{:10s} | {:10s} | {:10s}".format('  Before', '  After', ' Removed')
        print(s0)
        print('-'*40)
        filter_df = bbox_df.query("filter==True")
        s1 = "{:10d} | {:10d} | {:10d}".format(len(bbox_df), len(filter_df), len(bbox_df) - len(filter_df))
        print(s1)
        print('-'*40)
        df = pd.concat((zft_df.loc[~bbox_idx][self.zft_cols], filter_df[self.zft_cols]), axis=0)
        sub_df = zft2sub(df)

        return sub_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-csv', type=str, default='csv/submission-wh.csv', help='path of submission.csv with width, height')
    parser.add_argument('--train-csv', type=str, default='csv/gt.csv', help='path of gt.csv')
    parser.add_argument('--feature-cols', type=str, default='x_min,x_max,y_min,y_max,x_center,y_center,w,h,ar,area', 
                                            help='columns to be used as features')
    parser.add_argument('--save-csv', type=str, default='csv/submission-wh-filtered.csv', help='path of filtered submission.csv')
    opt = parser.parse_args()

    SUB_PATH     = opt.sub_csv
    TRAIN_PATH   = opt.train_csv
    FEATURE_COLS = opt.feature_cols.split(',')

    sub_df   = pd.read_csv(SUB_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    bbox_filter = BboxFilter(feature_cols=FEATURE_COLS)
    bbox_filter.fit(train_df, normalize=False)
    sub_df = bbox_filter.filter(sub_df)
    print('saving to',opt.save_csv)
    print('done :)\n')
    sub_df.to_csv(opt.save_csv, index=False)