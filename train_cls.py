import shutil, os
from glob import glob
import json
import argparse


cfg2ep_2cls = {
    'efficientnet_b6':{
        '512':5,
        '640':5,
        '768':5
    },
    'efficientnet_b7':{
        '512':4,
        '640':4,
        '768':6
    }
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings-path', type=str, default='SETTINGS.json', help='image size to create')
    parser.add_argument('--bs-path', type=str, default='classification/cls_bs.json', help='classification batch size info for different models')
    parser.add_argument("--debug", action="store_true", help="train on only first 1000 images")
    opt = parser.parse_args()

    bs_params = json.load(open(opt.bs_path, "r"))
    SETTINGS=json.load(open(opt.settings_path, "r"))
    debug=opt.debug

    ROOT_CHEX_DIR=SETTINGS['ROOT_CHEXPERT_DIR']
    TRAIN_PATH=SETTINGS['TRAIN_DATA_CLEAN_PATH']
    RICORD_PATH=SETTINGS['RICORD_DIR']
    SAVE_DIR=SETTINGS['CLS_MODEL_DIR']

    save_loc_chex=os.path.join(SAVE_DIR,'chex_cls')
    save_loc_2cls=os.path.join(SAVE_DIR,'2cls')
    save_loc_4cls=os.path.join(SAVE_DIR,'4cls')

    os.makedirs(save_loc_chex,exist_ok=True)
    os.makedirs(save_loc_2cls,exist_ok=True)
    os.makedirs(save_loc_4cls,exist_ok=True)

    ## CHEXPERT TRAININGS

    saved_models_chex={}
    epoch=10
    print ('\n CHEXPERT TRAINING \n')
    for model in bs_params.keys():
        saved_models_chex[model]={}
        for img_size in bs_params[model].keys():
            bs=bs_params[model][img_size]

            name=model+'_'+str(img_size)
            name=os.path.join(save_loc_chex,name,name+'.h5')

            command= f'python classification/train_chex.py --TRAIN_DATA_CLEAN_PATH {ROOT_CHEX_DIR} ' +\
             f'--MODEL_DIR {name} --epochs {epoch} --dim {img_size} --model_name {model} --bs {bs} '

            if debug:
                command=command +'--debug'

            os.system(command)
            saved_models_chex[model][img_size]=name

    ## 4 CLS TRAININIGS

    model_dirs_4cls=[]
    epoch=8
    
    
    print ('\n 4 CLASS TRAINING \n')
    for model in bs_params.keys():
        print('\n## ',f'Model - {model}',' ##','\n')
        for img_size in bs_params[model].keys():
            print('## ',f'Image Size - {img_size}',' ##')

            save_epoch=cfg2ep_2cls[model][img_size]
            chex_path=saved_models_chex[model][img_size]
            bs=bs_params[model][img_size]

            name=model+'_'+str(img_size)
            name=os.path.join(save_loc_4cls,name,name+'.h5')

            command = f'python classification/train_4cls.py --TRAIN_DATA_CLEAN_PATH {TRAIN_PATH} --CHECKPOINT_DIR {chex_path} ' + \
                f'--MODEL_DIR {name} --epochs {epoch} --dim {img_size} --model_name {model} --bs {bs} --save_epoch {save_epoch} '

            if debug:
                command=command +'--debug'
            os.system(command)
            model_dirs_4cls.append(os.path.abspath(os.path.join(os.getcwd(),name)))

    ## 2 CLS TRAININIG
    print ('\n 2 CLASS TRAINING \n')
    model= 'efficientnet_b7'
    img_size=640
    epoch = 8
    chex_path=saved_models_chex[model][img_size]
    name=model+'_'+str(img_size)
    name=os.path.join(save_loc_2cls,name)
    model_dirs_2cls=[]

    command = f'python classification/train_2cls_fold.py --TRAIN_DATA_CLEAN_PATH {TRAIN_PATH} --RICORD_PATH {RICORD_PATH} ' + \
     f'--CHECKPOINT_DIR {chex_path}  --MODEL_DIR {name} --epochs {epoch} --dim {img_size} --model_name {model} --bs {bs} '

    if debug:
        command=command +'--debug'
    os.system(command)

    model_dirs_2cls.extend(glob(os.path.abspath(os.path.join(os.getcwd(),name,'*.h5'))))
