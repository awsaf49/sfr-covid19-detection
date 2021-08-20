import shutil, os
from os import listdir
from os.path import isfile, join
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings-path', type=str, default='SETTINGS.json', help='image size to create')
    parser.add_argument('--bs-path', type=str, default='classification/cls_bs.json', help='classification batch size info for different models')
    parser.add_argument('--debug', action='store_true', help='process only 100 images in debug mode')
    opt = parser.parse_args()

    print('\n\n')
    print('#'*100)
    print('#'*30, ' CLASSIFICATION ', '#'*52)
    print('#'*100)
    print('\n\n')

    command = f'python train_cls.py --settings-path {opt.settings_path} --bs-path {opt.bs_path}'
    if opt.debug:
        command += ' --debug'
    os.system(command)

    print('\n\n')
    print('#'*100)
    print('#'*30, ' DETECTION ', '#'*52)
    print('#'*100)
    print('\n\n')

    command = f'python train_det.py --settings-path {opt.settings_path}'
    if opt.debug:
        command += ' --debug'
    os.system(command)