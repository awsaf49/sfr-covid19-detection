import shutil, os
from os import listdir
from os.path import isfile, join
import json
import argparse

def train_model(model_name, opt):
    ####################### CHEXPERT PRETRAIN
    command = f"python detection/chexpert_detection.py --CONFIG {model_name}"
    os.system(command)
    # TODO : get pretrained model path
    backbone_path = ""
    save_dir = ""

    ####################### 5 FOLD TRAINING
    for fold in range(5):
        command = f"python detection/train_det_1fold.py --settings-path {opt.settings.path} " + \
                  f"--pretrained-backbone {backbone_path} --model {model_name} " + \
                  f"--save-dir {save_dir} --img-size 512 --epochs 2 --debug --fold {fold}"
        os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings-path', type=str, default='SETTINGS.json', help='image size to create')
    opt = parser.parse_args()

    ##################### PREPARE DATA
    command = f"python detection/prepare_yolo_format.py --settings-path {opt.settings.path}"
    os.system(command)

    train_model("yolov5x-tr")
    train_model("yolov5x6")
    train_model("yolov3-spp")




