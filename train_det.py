import shutil, os
from os import listdir
from os.path import isfile, join
import json
import argparse

def train_model(model_name, opt):
    with open(opt.settings_path, 'r') as f:
        data = json.load(f)

    with open(opt.bs_path, 'r') as f:
        BS_DICT = json.load(f)

    ####################### CHEXPERT PRETRAIN
    save_chex_dir = f"./chex_det_models/{model_name}"
    os.makedirs(save_chex_dir, exist_ok = True)
    save_chex_dir += '/best.pt'
    command = f"python detection/chexpert_detection.py --CONFIG {model_name} --save-dir {save_chex_dir}"
    if opt.debug:
            command += " --DEBUG"
    print(command)
    os.system(command)
    backbone_path = save_chex_dir

    ####################### 5 FOLD TRAINING
    BATCH_SIZE = BS_DICT[model_name]['det']
    for fold in range(5):
        print('\n\n')
        print('#'*100)
        print('#'*30, model_name.upper(), ' - FOLD ', fold, '#'*50)
        print('#'*100)
        print('\n\n')

        print("Using Batch Size : ", BATCH_SIZE)
        save_dir = join(data['DET_MODEL_DIR'], f'{model_name}/fold-{fold}')
        os.makedirs(save_dir, exist_ok = True)
        save_dir += '/best.pt'
        command = f"python detection/train_det_1fold.py --settings-path {opt.settings_path} " + \
                  f"--pretrained-backbone {backbone_path} --model {model_name} " + \
                  f"--save-dir {save_dir} --img-size 512 --epochs 50 --fold {fold} --batch-size {BATCH_SIZE}"
        if opt.debug:
            command += " --debug"
        print(command)
        os.system(command)

    shutil.rmtree('runs') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings-path', type=str, default='SETTINGS.json', help='image size to create')
    parser.add_argument('--bs-path', type=str, default='detection/det_bs.json', help='detection batch size info for different models')
    parser.add_argument('--debug', action='store_true', help='process only 100 images in debug mode')
    opt = parser.parse_args()

    ##################### PREPARE DATA
    command = f"python detection/prepare_yolo_format.py --settings-path {opt.settings_path}"
    print(command)
    os.system(command)

    train_model("yolov5x-tr", opt)
    train_model("yolov5x6", opt)
    train_model("yolov3-spp", opt)




