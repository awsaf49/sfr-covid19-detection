import shutil, os
from os import listdir
from os.path import isfile, join
import json
import argparse

def train_model(model_name, opt):
    ####################### CHEXPERT PRETRAIN
    os.makedirs( f"./chex_det_models/{model_name}", exist_ok = True)
    save_chex_dir = f"./chex_det_models/{model_name}/best.pt"
    command = f"python detection/chexpert_detection.py --CONFIG {model_name} --save-dir {save_chex_dir}"
    print(command)
    # os.system(command)
    # TODO : get pretrained model path
    backbone_path = save_chex_dir

    ####################### 5 FOLD TRAINING
    for fold in range(5):
        print('\n\n')
        print('#'*100)
        print('#'*30, model_name.upper(), ' - FOLD ', fold, '#'*50)
        print('#'*100)
        print('\n\n')
        # TODO : fix this
        os.makedirs( f"./models/{model_name}/fold-{fold}", exist_ok = True)
        save_dir = f"./models/{model_name}/fold-{fold}/best.pt"
        command = f"python detection/train_det_1fold.py --settings-path {opt.settings_path} " + \
                  f"--pretrained-backbone {backbone_path} --model {model_name} " + \
                  f"--save-dir {save_dir} --img-size 512 --epochs 2 --debug --fold {fold}"
        print(command)
        os.system(command)

    shutil.rmtree('runs') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings-path', type=str, default='SETTINGS.json', help='image size to create')
    opt = parser.parse_args()

    ##################### PREPARE DATA
    command = f"python detection/prepare_yolo_format.py --settings-path {opt.settings_path}"
    print(command)
    os.system(command)

    train_model("yolov5x-tr", opt)
    train_model("yolov5x6", opt)
    train_model("yolov3-spp", opt)




