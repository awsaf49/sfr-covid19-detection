import argparse
import json
import yaml

import numpy as np, pandas as pd
from glob import glob
import shutil, os
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
import seaborn as sns
import sklearn
tqdm.pandas()

def extract_json_info(json_file):
    '''
    returns dictionary containing file paths
    '''
    PATHS = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
    PATHS['ROOT_DIR'] = os. getcwd()
    PATHS['YOLO_REPO_PATH'] = data['']
    PATHS['TRAIN_CSV_PATH'] = data['']
    PATHS['TEST_CSV_PATH'] = data['']
    PATHS['DET_TRAIN_IMAGES_PATH'] = data['']
    PATHS['DET_TRAIN_LABELS_PATH'] = data['']
    PATHS['DET_TEST_IMAGES_PATH'] = data['']

    # EXTERNAL RSNA DATA
    PATHS['RSNA_IMAGES_PATH'] = data['']
    PATHS['RSNA_LABELS_PATH'] = data['']
    PATHS['RSNA_METADATA_CSV'] = data['']

    # DEFINE YOLO DATA PATH
    os.makedirs(f'{os. getcwd()}/Dataset/scd/images', exist_ok = True)
    os.makedirs(f'{os. getcwd()}/Dataset/scd/labels', exist_ok = True)

    PATHS['YOLO_IMAGES_PATH'] = f'{os. getcwd()}/Dataset/scd/images/main/'
    PATHS['YOLO_LABELS_PATH'] = f'{os. getcwd()}/Dataset/scd/labels/main/'
    PATHS['YOLO_RSNA_IMAGES_PATH'] = f'{os. getcwd()}/Dataset/scd/images/rsna-pdc/'
    PATHS['YOLO_RSNA_LABELS_PATH'] = f'{os. getcwd()}/Dataset/scd/labels/rsna-pdc/'
    

    PATHS['META_DATA_DIR'] = data['']
    return PATHS
    
def hyperparams():
    hyp = """lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
smoothing: 0.01 # label smoothing for bce
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 2.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.5  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.25  # image mixup (probability)"""
    return hyp

# FIX DATA
def get_fix(grp):
    if grp.loc[grp.label!='none 1 0 0 1 1'].shape[0]!=0: # remove from those groups where there is img with bbox
        grp.loc[grp.label=='none 1 0 0 1 1', 'fix'] = 1
    return grp

# CONVERT
def convert1(label_dir = new_label_dir, save=False):
    label_paths = glob(os.path.join(label_dir,'**/*txt'), recursive=True)
    img_cnt=0
    for label_path in tqdm(label_paths):
        string = open(label_path, 'r').read()
        string = string.replace('\n','').strip(' ').split(' ')
        change=False
        for idx in range(len(string)):
            if idx%5==0:
                if string[idx]!='0':
                    change=True
                    string[idx]='0'
                else:
                    string=''
                    change=True
                    break
        img_cnt+=change*1           
        if save:
            f = open(label_path, 'w')
        if len(string)==0 and save:
            f.write(string)
            f.close()
            continue
        try:
            bboxes = np.array(string).reshape(-1, 5)
        except:
            print(label_path)
            print(open(label_path, 'r').read())
            continue
        for bbox_idx, bbox in enumerate(bboxes):
            annot = ' '.join(bbox) + (' \n' if len(bboxes)!=(bbox_idx+1) else '')
            if save:
                f.write(annot.strip(' '))
        if save:
            f.close()
    print(f'image changed: {img_cnt}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings-path', type=str, default='SETTINGS.json', help='image size to create')
    opt = parser.parse_args()

    # EXTRACT PATHS FROM SETTINGS.JSON
    PATHS = extract_json_info(opt.settings_path)

    # LOAD HYPERPARAMETERS
    HYP = hyperparams()
    hyp_path = PATHS['YOLO_REPO_PATH'] + '/hyp.yaml'
    yaml.dump(yaml.load(HYP), open(hyp_path, 'w'))
    yaml.load(open(hyp_path, 'r'), yaml.FullLoader)

    # LOAD META DATA
    train_df = pd.read_csv(PATHS['TRAIN_CSV_PATH'])
    test_df  = pd.read_csv(PATHS['TEST_CSV_PATH'])
    train_df['image_path'] = PATHS['DET_TRAIN_IMAGES_PATH'] + train_df.image_id + '.png'
    test_df['image_path']  = PATHS['DET_TEST_IMAGES_PATH'] + test_df.image_id + '.png'
    print('Checking train.csv shape: ',train_df.shape)

    # FIX DATA
    train_df['fix'] = 0
    train_df = train_df.groupby(['StudyInstanceUID']).progress_apply(get_fix)
    print('Fixed train.csv shape: ',train_df.shape, ' and count: ', train_df.fix.value_counts())
    
    # CLASS TO LABEL MAPPING
    name2label = {
        'Negative for Pneumonia': 0,
        'Indeterminate Appearance': 1,
        'Atypical Appearance': 2,
        'Typical Appearance': 3,
    }
    class_names  = list(name2label.keys())
    class_labels = list(name2label.values())
    label2name = {v:k for k, v in name2label.items()}
    train_df['class_name']  = train_df.progress_apply(lambda row:row[class_names].iloc[[row[class_names].values.argmax()]].index.tolist()[0], axis=1)
    train_df['class_label'] = train_df.class_name.map(name2label)
    

    # SPLIT
    fold_df = pd.read_csv(f"{PATHS['META_DATA_DIR']}/scd_fold.csv")
    fold_df['StudyInstanceUID'] = fold_df.image_id.map(dict(train_df[['image_id','StudyInstanceUID']].values))
    study2fold = dict(fold_df[['StudyInstanceUID', 'fold']].values)
    train_df['fold'] = train_df['StudyInstanceUID'].map(study2fold)
    print('FOLDWISE train.csv count: ', train_df.fold.value_counts())

    # TRANSFER MAIN DATA TO YOLO FORMAT DATA
    print('Processing Main Dataset :', end='')
    shutil.copytree(PATHS['DET_TRAIN_IMAGES_PATH'], PATHS['YOLO_IMAGES_PATH'])
    shutil.copytree(PATHS['DET_TRAIN_LABELS_PATH'], PATHS['YOLO_LABELS_PATH'])
    print(u'[\u2713]')

    #################### EXTERNAL DATA
    # RSNA
    print('Processing RSNA Dataset :', end='')
    shutil.copytree(PATHS['RSNA_IMAGES_PATH'], PATHS['YOLO_RSNA_IMAGES_PATH'])
    shutil.copytree(PATHS['RSNA_LABELS_PATH'], PATHS['YOLO_RSNA_LABELS_PATH'])
    print(u'[\u2713]')
    rsna_df = pd.read_csv(PATHS['RSNA_METADATA_CSV']).drop_duplicates()
    rsna_df['image_path'] = PATHS['YOLO_RSNA_IMAGES_PATH'] + rsna_df.image_id + '.png'
    rsna_df['label_path'] = PATHS['YOLO_RSNA_LABELS_PATH'] + rsna_df.image_id + '.txt'
    
    # CONVERSION
    convert1(PATHS['YOLO_LABELS_PATH'], save=True)
    print('before conversion:')
    print(open(sorted(glob(PATHS['DET_TRAIN_LABELS_PATH']+'*'))[10], 'r').read())
    print(open(sorted(glob(PATHS['DET_TRAIN_LABELS_PATH']+'*'))[100], 'r').read())
    print('after conversion:')
    print(open(sorted(glob(PATHS['YOLO_IMAGES_PATH']+'*'))[10], 'r').read())
    print(open(sorted(glob(PATHS['YOLO_LABELS_PATH']+'*'))[100], 'r').read())

    ####################  FILTER
    print('===== Filter =====')
    print('Before filter:',train_df.shape[0])

    # FIX_DATA: # take only images with bbox
    train_df = train_df[train_df.fix!=1]
    print('After fix:',train_df.shape[0])
    
    # REMOVE_DUP:  # remove duplicates
    dup_0 = train_df.query("dup_id==0") # take all from non-duplicates
    dup_1 = train_df.query("dup_id>0").groupby("StudyInstanceUID").head(1) # take one from duplicates
    train_df = pd.concat((dup_0, dup_1), axis=0)
    print('After removal:',train_df.shape[0])

    # save df
    main_csv_path = PATHS['YOLO_REPO_PATH'] + '/main.csv'
    rsna_csv_path = PATHS['YOLO_REPO_PATH'] + '/rsna.csv'
    print('Saving DataFrames :', end='')
    train_df.to_csv(main_csv_path, index=False)
    rsna_df.to_csv(rsna_csv_path, index=False)
    print(u'[\u2713]')

    print('Completed processing yolo format datasets')
    