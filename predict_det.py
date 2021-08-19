# check functions
def checked():
    print(u'[\u2713]')
def failed():
    print(u"[\u2715]")
    
print('### IMPORTING LIBRARIES', end=' ')
import argparse
import numpy as np, pandas as pd
from glob import glob
import os, shutil
import torch
import os
from tqdm import tqdm
tqdm.pandas()
import math, re
import json
checked()


CLASS_LABELS  = ['0', '1', '2', '3']

# params for geometric mean
## bbox
ALPHA = 0.7 # det
BETA  = 0.2 # 4cls
GAMMA = 0.1 # 2cls

## none
BETA2  = 0.8 # 4cls
GAMMA2 = 0.2 # 2cls

## negative
BETA3  = 1.0 # 4cls
GAMMA3 = 0.0 # 2cls

## bbox-filter
BBOX_FILTER = True


# detection args
NMS_CONF = 0.001 # 0.001 - chris
NMS_IOU  = 0.5
MAX_DET  = 1000 # max bbox per img
DIM      = 512


NAME2LABEL = { 
    'negative': 0,
    'indeterminate': 1,
    'atypical': 2,
    'typical': 3}
LABEL2NAME  = {v:k for k, v in NAME2LABEL.items()}


def voc2yolo(image_height, image_width, bboxes):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]/ image_height
    
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    
    bboxes[..., 0] = bboxes[..., 0] + w/2
    bboxes[..., 1] = bboxes[..., 1] + h/2
    bboxes[..., 2] = w
    bboxes[..., 3] = h
    
    return bboxes

def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height
    
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
    
    return bboxes

def coco2yolo(image_height, image_width, bboxes):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # normolizinig
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]/ image_height
    
    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]/2
    
    return bboxes

def yolo2coco(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # denormalizing
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]* image_height
    
    # converstion (xmid, ymid) => (xmin, ymin) 
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    
    return bboxes

def fix_labels(row):
    image_id = row['id']
    prob_b = img_cls_df.query("image_id==@image_id")["0"].values[0] # study-level classifier
    prob_g = 1-img_cls_df.query("image_id==@image_id")["opacity"].values[0] # 2cls opacity classifier
    prob   = np.power(prob_b, BETA2)*np.power(prob_g, GAMMA2)
    if row['PredictionString']!="none 1 0 0 1 1":
        row['PredictionString'] = (row['PredictionString']+' '+f"none {prob} 0 0 1 1").strip(' ')
    return row


def get_PredictionString(row, thr=0):
    string = ''
    for idx in range(4):
        conf =  row[str(idx)]
        if conf>thr:
            string+=f'{LABEL2NAME[idx]} {conf} 0 0 1 1 '
    if len(string)==0:
        string = 'negative 1.0 0 0 1 1'
    string = string.strip()
    return string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='fast', 
                        help='use provided weights in `fast` mode else newly trained weights`full` mode')
    parser.add_argument('--debug', type=int, default=0, help="infer on only first 100 images")   
    opt = parser.parse_args()
    
    # getting args
    DEBUG = opt.debug
    MODE  = opt.mode
    
    # settings.json
    print('### LOADING SETTINGS.json', end=' ')
    cfg = json.load(open('SETTINGS.json', 'r'))
    checked(); print()
    
    if MODE=='fast':
        CHECKPOINT_DIR = os.path.abspath(cfg['CHECKPOINT_DIR'])
    elif MODE=='full':
        CHECKPOINT_DIR = os.path.abspath(cfg['MODEL_DIR'])
    else:
        raise ValueError('mode is neither `fast` nor `full`')
    
    # test.csv
    TEST_CSV = cfg['TEST_CSV_PATH']
    print(f'### READING {TEST_CSV}', end=' ')
    test_df = pd.read_csv(TEST_CSV)
    checked()
    
    # meta data dir
    SUB_DIR = os.path.abspath(cfg['SUBMISSION_DIR'])
    
    # test_directory
    DATA_DIR = os.path.abspath(cfg['TEST_DATA_CLEAN_PATH'])
    
    #-------------------------------------------
    ### Detection
    #------------------------------------------
    os.chdir('yolov5')
    backbones = sorted(glob(os.path.join(CHECKPOINT_DIR,'det/*/')))
    backbones = backbones[:(2 if DEBUG else len(backbones))]
    folds = [0, 1] if DEBUG else [0,1,2,3,4] 
    for k,backbone in enumerate(backbones):
        for fold in folds:
            print('#'*25)
            print('### Backbone =',k,', Fold =',fold)
            print('#'*25)
            model = backbone + f'/fold-{fold}'
            command=f"""python detect.py --weights_dirs {model}\
                --img {DIM}\
                --conf {NMS_CONF}\
                --iou {NMS_IOU}\
                --source {DATA_DIR}\
                --save-txt\
                --save-conf\
                --exist-ok\
                --save-img {0}\
                --augment\
                --project runs/detect_b{k}_f{fold}\
                --max-det {MAX_DET}"""
            os.system(command)
    os.chdir('..') 
    
    #------------------------------------------
    ### WBF
    #------------------------------------------
    iou_thr = 0.625

    wbf_files = []; final = []
    for bb in range(len(backbones)):
        model = []
        for ff in range(len(folds)):
            model.append(os.path.join(SUB_DIR, f'test_b{bb}_f{ff}.csv'))
        wbf_files.append(model)
        final.append(os.path.join(SUB_DIR, f'test_b{bb}.csv'))
    wbf_files.append(final); print('== wbf_files ==\n', wbf_files)
    
    # Detection
    print('### CONVERTING TXT TO PREDSTRINGS...')
    POST_PROCESS = False
    # WE DON'T DO DETECTION POST PROCESS HERE. WE DO BELOW AFTER WBF
    files_to_remove = []
    for bb in range(len(backbones)):
        for ff in range(len(folds)):
            total = 0
            image_ids = []
            PredictionStrings = []
            if POST_PROCESS:
                img_cls_df = pd.read_csv(os.path.join(SUB_DIR,'image_cls.csv'))
            for file_path in tqdm(glob(os.path.join(os.getcwd(), f'yolov5/runs/detect_b{bb}_f{ff}/exp/labels/*txt'))):
                image_id = file_path.split('/')[-1].split('.')[0]+'_image'
                w, h = test_df.loc[test_df.image_id==image_id,['width', 'height']].values[0]
                f = open(file_path, 'r')
                data = np.array(f.read().replace('\n', ' ').strip().split(' ')).astype(np.float32).reshape(-1, 6)                
                data = data[:, [0, 5, 1, 2, 3, 4]]
                total+=data.shape[0]
                bboxes = list(np.concatenate((data[:, :2], np.round(yolo2voc(h, w, data[:, 2:]))), axis =1).reshape(-1).astype(str))
                if POST_PROCESS:
                    prob_b = 1-img_cls_df.query("image_id==@image_id")["0"].values[0]
                    prob_g = img_cls_df.query("image_id==@image_id")["opacity"].values[0]
                for idx in range(len(bboxes)):
                    bboxes[idx] = str(int(float(bboxes[idx]))) if idx%6!=1 else bboxes[idx]
                    bboxes[idx] = 'opacity' if idx%6==0 else bboxes[idx]
                    if (idx%6==1)&(POST_PROCESS):
                        bboxes[idx] = str(np.power(float(bboxes[idx]), ALPHA)*np.power(prob_b, BETA)*np.power(prob_g, GAMMA)) # geometric mean (x^alpha)*(y^beta)
                    elif (idx%6==1):
                        bboxes[idx] = f'{float(bboxes[idx]):.9}'
                image_ids.append(image_id)
                PredictionStrings.append((' '.join(bboxes)))
            print('Total BBox:',total)
            pred_img_df = pd.DataFrame({'image_id':image_ids,
                                       'PredictionString':PredictionStrings})
            pred_img_df.to_csv(os.path.join(SUB_DIR, f'test_b{bb}_f{ff}.csv'),index=False)
            files_to_remove.append(os.path.join(SUB_DIR, f'test_b{bb}_f{ff}.csv'))
    print('### CONVERTING TXT TO PREDSTRINGS', end=' '); checked()
            
    # Fusing Boxes
    import warnings
    warnings.filterwarnings("ignore") #wbf
    print('\n### FUSING BOXES USING WBF...')
    for wbf_num,files in enumerate(wbf_files):
        print('#'*25)
        print('### Backbone',wbf_num)
        print('#'*25)
        print()

        # FILLNA, REMOVE WORD OPACITY, REMOVE WORD IMAGE FROM ID, REMOVE NONE PREDICTIONS
        preds = [pd.read_csv(file).fillna('').rename({'id':'image_id'},axis=1) for file in files]
        for p in preds:
            p.PredictionString = p.PredictionString.str.replace('opacity','0')
            p.image_id = p.image_id.map(lambda x: x.split('_')[0])
            for index,row in p.iterrows():
                text = ''
                s = row.PredictionString.split(' ')
                if len(s)%6!=0: print('ERROR')
                for k in range(len(s)//6):
                    if s[k*6]=='none': continue
                    text += f'{s[k*6]} {s[k*6+1]} {s[k*6+2]} {s[k*6+3]} {s[k*6+4]} {s[k*6+5]} '
                row.PredictionString = text[:-1]

        # GET TEST WIDTHS AND HEIGHTS
        df_test = test_df.copy()
        df_test['image_id'] = df_test['image_id'].map(lambda x: x.split('_')[0])
        df_test = df_test[['image_id','width','height']].set_index('image_id')
        if DEBUG:
            df_test = df_test.iloc[:100]
        df_test.width = df_test.width.astype('int32')
        df_test.height = df_test.height.astype('int32')

        # CONVERT PREDS AS STRING TO DATAFRAME
        print('Converting',len(preds),'dataframe of string to dataframe of numbers...')
        for i, pred in enumerate(preds):
            print(i,', ',end='')
            new_pred = []
            for index, row in pred.iterrows():
                #if index%50==0: print(index,', ',end='')
                if row.PredictionString == '': continue
                try:
                    data_flat = np.array(row.PredictionString.split(' '))
                except:
                    print('###',row.PredictionString,'###')
                data_matrix = data_flat[:len(data_flat) // 6 * 6].reshape(-1, 6)

                df = pd.DataFrame( {
                    'image_id' : np.repeat(row.image_id,len(data_matrix)), 
                    'score' : data_matrix[:,1].astype(float),
                    'x_min' : data_matrix[:,2].astype(int),
                    'y_min' : data_matrix[:,3].astype(int),
                    'x_max' : data_matrix[:,4].astype(int),
                    'y_max' : data_matrix[:,5].astype(int),
                    'class_id' : data_matrix[:,0].astype(int)})
                new_pred.append(df)
            preds[i] = pd.concat(new_pred).join(df_test, on=['image_id'])
        print(); print()

        for i, sub_name in enumerate(files):
            print(sub_name,'has box count', len(preds[i]))
        print()

        # NORMALIZE BOXES
        for i, pred in enumerate(preds):
            pred['x_min'] = pred['x_min'] / pred['width']
            pred['y_min'] = pred['y_min'] / pred['height']
            pred['x_max'] = pred['x_max'] / pred['width']
            pred['y_max'] = pred['y_max'] / pred['height']

        print('Here are preds',files[0],'preds...')
        print( preds[0].sort_values(['score']).head(2) )
        print()

        from ensemble_boxes import weighted_boxes_fusion, non_maximum_weighted, soft_nms, nms

        sub_results = []
        label_dict = {0: 'opacity'}

        # processing of other classes
        print('Processing boxes with WBF...')
        for jj,image_id in enumerate(preds[0].image_id.unique()):
            if jj%100==0: print(jj,', ',end='')

            boxes_list, labels_list, scores_list = [], [], []

            for i in range(len(preds)):
                sub_df = preds[i][preds[i].image_id == image_id].sort_values(['score'])
                boxes_list.append(sub_df[['x_min', 'y_min', 'x_max', 'y_max']].values)
                labels_list.append(sub_df['class_id'].values)
                scores_list.append(sub_df['score'].values)

            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr)

            sub_df_weighted = pd.DataFrame(boxes, columns=['x_min', 'y_min', 'x_max', 'y_max'])
            if len(sub_df)>0:
                sub_df_weighted['image_id'] = image_id
                sub_df_weighted['x_min'] = (sub_df_weighted['x_min'] * sub_df.width.values[0]).astype(int)
                sub_df_weighted['x_max'] = (sub_df_weighted['x_max'] * sub_df.width.values[0]).astype(int)
                sub_df_weighted['y_min'] = (sub_df_weighted['y_min'] * sub_df.height.values[0]).astype(int)
                sub_df_weighted['y_max'] = (sub_df_weighted['y_max'] * sub_df.height.values[0]).astype(int)
                sub_df_weighted['height']= (sub_df.height.values[0]).astype(int)
                sub_df_weighted['width'] = (sub_df.width.values[0]).astype(int)
                sub_df_weighted['score'] = scores
                sub_df_weighted['class_id'] = labels.astype(int)
                sub_df_weighted['class_name'] = sub_df_weighted['class_id'].apply(lambda s : label_dict[s])

                sub_results.append(sub_df_weighted.copy(deep=True))

        preds_test_weight = pd.concat(sub_results)
        preds_test_weight = preds_test_weight[preds[0].columns]
        print(); print()

        print('Done. We now have',len(preds_test_weight),'bboxes\n')

        subs = pd.DataFrame(columns=['image_id','PredictionString'])

        print('Converting dataframe of numbers to dataframe of strings...')
        for jj,(image_id, sub_df) in enumerate(preds_test_weight.groupby('image_id')):
            if jj%50==0: print(jj,', ',end='')

            predsxx = ''
            for index, row in sub_df.iterrows():
                predsxx += f'{int(row.class_id)} {row.score} {int(row.x_min)} {int(row.y_min)} {int(row.x_max)} {int(row.y_max)} '

            subs.loc[len(subs)] = (image_id, predsxx[:-1])


        if (wbf_num+1) != len(wbf_files):
            subs.to_csv(os.path.join(SUB_DIR, f'test_b{wbf_num}.csv'),index=False)
            print('Wrote to',os.path.join(SUB_DIR,f'test_b{wbf_num}.csv\n'))
            files_to_remove.append(os.path.join(SUB_DIR,f'test_b{wbf_num}.csv'))
        else:
            subs.to_csv(os.path.join(SUB_DIR,'test_wbf.csv'),index=False)
            print('Wrote to',os.path.join('test_wbf.csv\n'))
            files_to_remove.append(os.path.join(SUB_DIR,'test_wbf.csv'))
            
    pred_img_df = pd.read_csv(os.path.join(SUB_DIR,'test_wbf.csv'))
    for f in files_to_remove: os.system(f'rm {f}')
    img_cls_df = pd.read_csv(os.path.join(SUB_DIR,'image_cls.csv'))

    for index,row in pred_img_df.iterrows():
        row.image_id = row.image_id+'_image'

        # DETECTION SCORE POST PROCESS
        image_id = row.image_id
        prob_b = 1-img_cls_df.query("image_id==@image_id")["0"].values[0]
        prob_g = img_cls_df.query("image_id==@image_id")["opacity"].values[0]

        # WRITE DETECTION PREDICTION STRINGS
        p = row.PredictionString.split(' ')
        if len(p)%6!=0: print('ERROR')
        text = ''
        for k in range(len(p)//6):
            pr = str(np.power(float(p[k*6+1]), ALPHA)*np.power(prob_b, BETA)*np.power(prob_g, GAMMA)) # geometric mean (x^alpha)*(y^beta)
            text += f'opacity {pr} {p[k*6+2]} {p[k*6+3]} {p[k*6+4]} {p[k*6+5]} '
        row.PredictionString = text[:-1]
        
    image_df = pd.merge(test_df[['image_id']], pred_img_df, on='image_id', how='left').fillna("none 1 0 0 1 1")
    image_df = image_df.rename(columns={'image_id':'id'})
    
    # utilize 4cls & 2cls for `none`
    image_df   = image_df.apply(fix_labels, axis=1)
    print('### PROCESSING OF DETECTION', end=' '); checked();
    img_cls_csv = os.path.join(SUB_DIR,'image_cls.csv')
    os.system(f'rm {img_cls_csv}')
    
    # bbox-filter
    if BBOX_FILTER:
        print('### BBOX-FILTER...')
        image_df = image_df.merge(test_df.rename(columns={'image_id':'id',
                                                         'width':'Width',
                                                         'height':'Height'})[['id', 'Width', 'Height']])
        nofilter_csv = os.path.join(SUB_DIR, 'image-lvl-nofilter.csv')
        filter_csv   = os.path.join(SUB_DIR, 'image-lvl-filter.csv')
        image_df.to_csv(nofilter_csv,index=False)
        
        os.chdir('bbox')
        command = f"""python bbox_filter.py\
        --sub-csv {nofilter_csv}\
        --save-csv {filter_csv}"""
        os.system(command)
        os.chdir('..')
        image_df = pd.read_csv(filter_csv)
        print('### FILTERING BBOX', end=' '); checked()
        os.system(f'rm {filter_csv}')
        os.system(f'rm {nofilter_csv}')
    
    # use 2cls result for 4cls
    prob_g = 1-img_cls_df["opacity"].values[:, None] # 2cls-opacity

    prob_0  = img_cls_df[CLASS_LABELS[0:1]].values # none
    prob_0  = np.power(prob_0, BETA3)*np.power(prob_g, GAMMA3)

    prob_1  = img_cls_df[CLASS_LABELS[1:]].values # typical,atypical,indeterminate
    prob_1  = np.power(prob_1, BETA3)*np.power(1-prob_g, GAMMA3)

    img_cls_df.loc[:, CLASS_LABELS] = np.concatenate([prob_0, prob_1], axis=1).tolist()
    
    # grouping by max
    study_df = img_cls_df.groupby(['study_id'])[CLASS_LABELS].max().reset_index()
    study_df.rename(columns={'study_id':'id'}, inplace=True)

    #------------------------
    # Submission csv  
    #------------------------
    study_df['PredictionString'] = study_df.progress_apply(get_PredictionString, axis=1)
    study_df = study_df.drop(CLASS_LABELS, axis=1)
    
    # concat image & study dataframe
    sub_df = pd.concat([image_df, study_df])
    sub_df.to_csv(os.path.join(SUB_DIR,'submission.csv'),index=False)
    
    
    print('\n### FULL PREDICTION IS DONE!','\U0001F603\n')