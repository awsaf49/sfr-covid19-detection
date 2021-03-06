import argparse
import json
import yaml

import numpy as np, pandas as pd
from glob import glob
import shutil, os
from os import listdir
from os.path import isfile, join
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
    PATHS['YOLO_REPO_PATH'] = 'yolov5'
    PATHS['META_DATA_DIR'] = data['META_DATA_DIR']
    PATHS['TRAIN_CSV_PATH'] = join(data['META_DATA_DIR'], 'train_duplicate.csv')
    PATHS['DET_TRAIN_IMAGES_PATH'] = data['TRAIN_DATA_CLEAN_PATH']
    PATHS['DET_TRAIN_LABELS_PATH'] = data['LABEL_DIR']

    # EXTERNAL RSNA DATA
    PATHS['RSNA_IMAGES_PATH'] = join(data['RSNA_CLEAN_DIR'], 'images')
    PATHS['RSNA_LABELS_PATH'] = join(data['RSNA_CLEAN_DIR'], 'labels')
    PATHS['RSNA_METADATA_CSV'] = join(data['META_DATA_DIR'], 'rsna.csv')

    # DEFINE YOLO DATA PATH
    os.makedirs('yolov5/DETDataset/scd/images', exist_ok = True)
    os.makedirs('yolov5/DETDataset/scd/labels', exist_ok = True)

    PATHS['YOLO_IMAGES_PATH'] = 'yolov5/DETDataset/scd/images/main/'
    PATHS['YOLO_LABELS_PATH'] = 'yolov5/DETDataset/scd/labels/main/'
    PATHS['YOLO_RSNA_IMAGES_PATH'] = 'yolov5/DETDataset/scd/images/rsna-pdc/'
    PATHS['YOLO_RSNA_LABELS_PATH'] = 'yolov5/DETDataset/scd/labels/rsna-pdc/'
    return PATHS


def extract_model_params(model_name):
    '''
    returns model specific params based on model_name
    '''
    # TODO: recheck freeze_point
    if model_name == 'yolov5x-tr':
        WEIGHTS = 'yolov5x.pt'
        MODEL_CONFIG = 'yolov5/models/yolov5x-tr.yaml'
        FREEZE_POINT = 10

    elif model_name == 'yolov5x6':
        WEIGHTS = 'yolov5x6.pt'
        MODEL_CONFIG = 'yolov5/models/yolov5x6.yaml'
        FREEZE_POINT = 12

    elif model_name == 'yolov3-spp':
        WEIGHTS = 'yolov3-spp.pt'
        MODEL_CONFIG = 'yolov5/models/yolov3-spp.yaml'
        FREEZE_POINT = 11

    else:
        print('Wrong model name. Accepted names are : yolov5x-tr, yolov5x6, yolov3-spp')
        WEIGHTS = ''
        MODEL_CONFIG = ''
        FREEZE_POINT = 0

    return WEIGHTS, MODEL_CONFIG, FREEZE_POINT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings-path', type=str, default='SETTINGS.json', help='image size to create')
    parser.add_argument('--pretrained-backbone', type=str, default='', help='chexpert pretrained backbones')
    parser.add_argument('--model', type=str, default='yolov5x-tr', help='name of model to train')
    parser.add_argument('--save-dir', type=str, default='det/fold/best.pt', help='where to save best.pt')
    
    parser.add_argument('--img-size', type=int, default=512, help='image size to create')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=60, help='no of epochs to train')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--debug', action='store_true', help='process only 100 images in debug mode')
    opt = parser.parse_args()
    print(opt)
    # EXTRACT PATHS FROM SETTINGS.JSON
    PATHS = extract_json_info(opt.settings_path)

    # LOAD PARAMS
    FOLD = opt.fold
    IMAGE_SIZE = opt.img_size
    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    DEBUG = opt.debug

    WEIGHTS, MODEL_CONFIG, FREEZE_POINT = extract_model_params(opt.model)
    HYP_PATH = join(PATHS['YOLO_REPO_PATH'],'hyp.yaml')
    notebook_cfg = {
        'dim':IMAGE_SIZE,
        'batch':BATCH_SIZE,
        'fold':FOLD,
        'epochs':EPOCHS,
        'model':opt.model
    }
    NOTEBOOK_CONFIG_PATH = join(PATHS['YOLO_REPO_PATH'], 'notebook_cfg.yaml')
    yaml.dump(notebook_cfg, open(NOTEBOOK_CONFIG_PATH, 'w'))
    TRAIN_NAME = f'{opt.model} img{IMAGE_SIZE} fold{FOLD}'.replace(' ', '_')    

    # LOAD DATAFRAMES
    main_csv_path = join(PATHS['YOLO_REPO_PATH'], 'main.csv')
    rsna_csv_path = join(PATHS['YOLO_REPO_PATH'], 'rsna.csv')
    train_df = pd.read_csv(main_csv_path)
    rsna_df = pd.read_csv(rsna_csv_path)
    ap1_df = rsna_df[(rsna_df.view=='AP')&(rsna_df.label==1)]
    pa1_df = rsna_df[(rsna_df.view=='PA')&(rsna_df.label==1)]

  
    # CLASS TO LABEL MAPPING
    name2label = {
        'Negative for Pneumonia': 0,
        'Indeterminate Appearance': 1,
        'Atypical Appearance': 2,
        'Typical Appearance': 3,
    }
    class_names = list(name2label.keys())
    class_labels = list(name2label.values())
    label2name = {v:k for k, v in name2label.items()}


    # DISTRIBUTE FOLDWISE DATA
    train_paths = []
    fold_paths = PATHS['YOLO_IMAGES_PATH']+train_df[train_df.fold!=FOLD].image_id+'.png'
    train_paths += fold_paths.tolist()
    val_paths = PATHS['YOLO_IMAGES_PATH']+train_df[train_df.fold==FOLD].image_id+'.png'
    
    rsna_paths = []
    rsna_paths += ap1_df.image_path.tolist()
    rsna_paths += pa1_df.image_path.tolist()
    train_paths += rsna_paths
    train_paths = np.unique(train_paths).tolist()
    val_paths = np.unique(val_paths).tolist()

    print('Datasets:')
    print(f'  fold{FOLD}    :',len(fold_paths))
    print('  RSNA     :',len(rsna_paths))
    # print('  VBD      :',len(vbd_paths))
    print('===== Summary =====')
    print('  Total    :',len(train_paths)+len(val_paths))
    print('  Train    :',len(train_paths))
    print('  Val      :',len(val_paths))


    # YOLOv5 CONFIG
    if DEBUG:
        # only process first 100 images if in debug mode
        print("### RUNNING IN DEBUG MODE")
        train_paths = train_paths[:100]
        val_paths = val_paths[:100]
        EPOCHS = 3

    with open(join(PATHS['YOLO_REPO_PATH'], 'train.txt'), 'w') as f:
        for path in train_paths:
            f.write(path+'\n')
                
    with open(join(PATHS['YOLO_REPO_PATH'], 'val.txt'), 'w') as f:
        for path in val_paths:
            f.write(path+'\n')

    names = ['opacity']
    data = dict(
        train =  join(PATHS['YOLO_REPO_PATH'], 'train.txt') ,
        val   =  join(PATHS['YOLO_REPO_PATH'], 'val.txt' ),
        nc    = 1,
        names = names,
        )

    CONFIG_FILE_PATH = join(PATHS['YOLO_REPO_PATH'], 'siim-covid-19.yaml')
    with open(CONFIG_FILE_PATH, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    f = open(CONFIG_FILE_PATH, 'r')
    print('\nyaml:')
    print(f.read())


    # TRAIN USING YOLOv5
    import torch
    print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))   
    train_script_path = join(PATHS['YOLO_REPO_PATH'], 'train.py')

    train_command = f'WANDB_MODE=\"dryrun\" python {train_script_path} --img {IMAGE_SIZE} --batch {BATCH_SIZE} --epochs {EPOCHS} '+ \
                    f'--data {CONFIG_FILE_PATH} ' +  \
                    f'--cfg {MODEL_CONFIG} --weights {WEIGHTS} ' + \
                    f'--name {TRAIN_NAME} ' + \
                    f'--notebook {NOTEBOOK_CONFIG_PATH} ' + \
                    f'--hyp {HYP_PATH} --exist-ok ' + \
                    f'--backbone-weights {opt.pretrained_backbone} --freeze --freeze-modelno {FREEZE_POINT}'

    print("Executing train command : ", train_command)
    os.system(train_command)

    # SAVE MODEL
    try:
        print("Saving best model to ", opt.save_dir)
        model_path = f'runs/train/{TRAIN_NAME}/weights/best.pt'
        shutil.copy(
            model_path,
            opt.save_dir
        )
    except Exception as e: 
        print(e)
        print('Something went wrong')

    print('Training has completed.')


    

    