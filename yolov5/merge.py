import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
import pandas as pd

from ensemble_boxes import weighted_boxes_fusion, nms, soft_nms, non_maximum_weighted
# from score import score


LABELS = ['opacity']
       
# handling labels 
labels_2_number = {}
number_2_labels = {}

label_no = 0
for l in LABELS:
    labels_2_number[l] = label_no
    number_2_labels[label_no] = l
    label_no += 1


def get_all_csv(directory_name):
    sub_path = os.path.join(directory_name, '*.csv')
    submissions = [pd.read_csv(sub) for sub in glob.glob(sub_path)]
    return submissions



def merge_data(submissions, l2n=labels_2_number, n2l=number_2_labels, iou_thr=0.55, skip_box_thr=0.0):
    all_results = []
    image_paths = submissions[0]['image_id'].unique()

    for sub in submissions:
        sub['class'] = sub['class'].apply(lambda x: l2n[x])

    height, width = 0,0
    for path in tqdm(image_paths, total=len(image_paths)):
        all_boxes, conf, labels = [],[],[]
        for sub in submissions:
            df = sub[sub['image_id'] == path]
            # if no height then no prediction
            try:
                height = df['height'].values[0]
                width = df['width'].values[0]
            except:
                continue
            boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
            boxes[:, :3:2] /= width
            boxes[:, 1::2] /= height
            conf.append(df['score'].values.astype(np.float32))
            labels.append(df['class'].values.astype(np.int32))
            all_boxes.append(boxes)
        boxes, conf, labels = weighted_boxes_fusion(all_boxes, conf, labels, iou_thr=iou_thr, 
                                                    skip_box_thr=skip_box_thr)
        all_results.append({
            'image_id':[path]*len(boxes),
            'class': [number_2_labels[l] for l in labels],
            'score': np.round(conf,2).astype(str),
            'xmin': (boxes[:,0]*width).astype(np.int32),
            'ymin': (boxes[:,1]*height).astype(np.int32),
            'xmax': (boxes[:,2]*width).astype(np.int32),
            'ymax': (boxes[:,3]*height).astype(np.int32),
            'height': [height]*len(boxes),
            'width': [width]*len(boxes),
        })

    return all_results


def save_csv(all_results, submission_name):
    append_data=[]
    for result in all_results:
        df = pd.DataFrame(result, columns = ['image_id','class','score','xmin','ymin','xmax','ymax','height', 'width'])
        append_data.append(df)
    final_results=pd.concat(append_data)
    final_results.to_csv(f'{submission_name}', index=False)
    print('df shape', final_results.shape)
    print(f'Saved {submission_name}')
    return final_results




def merge_csv(submission_weight_list, method='nms', iou_thr=0.55, skip_box_thr=0.0, 
              sigma=0.1, l2n=labels_2_number, n2l=number_2_labels, submission_name='submission.csv',
              gt_name='gt_csv/ground_truth_test2.csv'):
    '''
    submission_weight_dict : [
        (path1, weight1),
        (path2, weight2),
        ...
    ],
    method: nms, snms, wbf, nmw
    '''

    submissions = []
    sub_weights = []
    all_results = []

    for path, weight in submission_weight_list:
        print(path, weight)
        submissions.append(pd.read_csv(path))
        sub_weights.append(weight)

    
    image_paths = submissions[0]['image_id'].unique()

    for sub in submissions:
        sub['class'] = sub['class'].apply(lambda x: l2n[x])

    height, width = 0,0
    for path in tqdm(image_paths, total=len(image_paths)):
        all_boxes, conf, labels = [],[],[]
        weights = []
        for i in range(len(submissions)):
            sub = submissions[i]
            df = sub[sub['image_id'] == path]
            # if no height then no prediction
            try:
                height = df['height'].values[0]
                width = df['width'].values[0]
                weights.append(sub_weights[i])
            except:
                continue
            boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
            boxes[:, :3:2] /= width
            boxes[:, 1::2] /= height
            conf.append(df['score'].values.astype(np.float32))
            labels.append(df['class'].values.astype(np.int32))
            all_boxes.append(boxes)
        

        if method == 'nms':
            boxes, conf, labels = nms(all_boxes, conf, labels, weights=weights,
                                    iou_thr=iou_thr)
        elif method == 'snms':
            boxes, conf, labels = soft_nms(all_boxes, conf, labels, iou_thr=iou_thr, weights=weights,
                                    thresh=skip_box_thr, sigma=sigma)
        elif method == 'wbf':
            boxes, conf, labels = weighted_boxes_fusion(all_boxes, conf, labels, weights=weights,
                                    iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif method  == 'nmw':
            boxes, conf, labels = non_maximum_weighted(all_boxes, conf, labels, weights=weights,
                                    iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        
    
        all_results.append({
            'image_id':[path]*len(boxes),
            'class': [number_2_labels[l] for l in labels],
            'score': np.round(conf,2).astype(str),
            'xmin': (boxes[:,0]*width).astype(np.int32),
            'ymin': (boxes[:,1]*height).astype(np.int32),
            'xmax': (boxes[:,2]*width).astype(np.int32),
            'ymax': (boxes[:,3]*height).astype(np.int32),
            'height': [height]*len(boxes),
            'width': [width]*len(boxes),
        })

    pred_df = save_csv(all_results, submission_name)


    # gt_df = pd.read_csv(gt_name)
    # score(pred_df = pred_df, gt_df = gt_df)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='merge.py')
    parser.add_argument('--iou-thres', type=float, default=0.55, help='for wbf')
    parser.add_argument('--skipbox-thres', type=float, default=0.0, help='for wbf')
    parser.add_argument('--csv-directory', type=str, default='0', help='directory of all submission csvs to merge')
    parser.add_argument('--output-name', type=str, default='submission_merged.csv', help='name of output csv')
    
    opt = parser.parse_args()
    print(opt)

    if opt.csv_directory != 0:
        all_submissions = get_all_csv(opt.csv_directory)
        if len(all_submissions) != 0:
            all_results = merge_data(all_submissions, iou_thr=opt.iou_thres, skip_box_thr=opt.skipbox_thres)
            save_csv(all_results, opt.output_name)
    else:
        print('Please recheck directory')
