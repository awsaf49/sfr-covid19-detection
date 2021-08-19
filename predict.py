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
import os
from tqdm import tqdm
tqdm.pandas()
import math, re
import json
checked()



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
    
    # test.csv
    TEST_CSV = cfg['TEST_CSV_PATH']
    print(f'### READING {TEST_CSV}', end=' ')
    test_df = pd.read_csv(TEST_CSV)
    checked()
    
    # test_directory
    DATA_DIR = cfg['TEST_DATA_CLEAN_PATH']
    
    #------------------------------------------
    ### Classification
    #------------------------------------------
    command=f"""python predict_cls.py\
    --debug {DEBUG}\
    --mode {MODE}"""
    os.system(command)
    
    #------------------------------------------
    ### Detection
    #------------------------------------------
    command=f"""python predict_det.py\
    --debug {DEBUG}\
    --mode {MODE}"""
    os.system(command)
