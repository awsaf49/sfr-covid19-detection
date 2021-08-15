# check functions
def checked():
    print(u'[\u2713]')
def failed():
    print(u"[\u2715]")
    
# installing libraries
print('### IMPORTING LIBRARIES', end=' ')
import os, shutil
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm; tqdm.pandas()
import cv2
import pydicom
import joblib
import json
from pydicom.pixel_data_handlers.util import apply_voi_lut
from joblib import Parallel, delayed
import argparse
checked()
    
# helpers
def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def resize_and_save(file_path, dim=256, aspect_ratio=False):
    split = 'train' if 'train' in file_path else 'test'
    base_dir = os.path.join(CLEAN_DATA_DIR, split)
    img  = read_xray(file_path)
    h, w = img.shape[:2]  # orig hw
    if dim!=-1:
        if aspect_ratio:
            r = dim / max(h, w)  # resize image to img_size
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            if r != 1:  # always resize down, only resize up if training with augmentation
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)
        else:
            img = cv2.resize(img, (dim, dim), cv2.INTER_AREA)
    filename = file_path.split('/')[-1].split('.')[0]
    cv2.imwrite(os.path.join(base_dir, f'{filename}.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return filename.replace('dcm','')+'_image',w, h

def find_path(row):
    row['filepath'] = glob(os.path.join(train_directory, row['StudyInstanceUID'] +f"/*/{row.image_id}.dcm"))[0]
    return row

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=1024, help='image size to create')
    parser.add_argument('--debug', type=int, default=0, help='process only 100 images in debug mode')
    opt = parser.parse_args()
    
    print('### LOADING SETTINGS.json', end=' ')
    cfg = json.load(open('SETTINGS.json', 'r'))
    checked(); print()

    RAW_DATA_DIR   = cfg["RAW_DATA_DIR"]
    train_image_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "train_image_level.csv"))
    train_study_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "train_study_level.csv"))

    train_directory = os.path.join(RAW_DATA_DIR, "train")
    test_directory  = os.path.join(RAW_DATA_DIR, "test")

    train_study_df['StudyInstanceUID'] = train_study_df['id'].apply(lambda x: x.replace('_study', ''))
    train_image_df['image_id']         = train_image_df['id'].map(lambda x: x.replace('_image', ''))

    del train_study_df['id'], train_image_df['id']

    train_df = train_image_df.merge(train_study_df, on='StudyInstanceUID')

    print(f'### TRAIN IMAGES FOUND: {train_df.shape[0]}/6334')
    print('### IMAGE_PATH SEARCHING', end=' ')
    tqdm.pandas(desc='searching ')
    train_df = train_df.progress_apply(find_path, axis=1)
    checked()

    # image & study id
    train_df['image_id'] = train_df.filepath.map(lambda x: x.split(os.sep)[-1].split('.')[0]+'_image')
    train_df['study_id'] = train_df.StudyInstanceUID.map(lambda x: x+'_study')

    # making directories for clean data
    print('### MAKING DIRECTORIES FOR CLEAN_DATA', end=' ')

    CLEAN_DATA_DIR = cfg["CLEAN_DATA_DIR"]
    os.makedirs(os.path.join(CLEAN_DATA_DIR, 'train'), exist_ok = True)
    os.makedirs(os.path.join(CLEAN_DATA_DIR, 'test'), exist_ok = True)
    checked(); print()

    # writing train images
    train_paths = train_df.filepath.tolist()
    write_files = 100 if opt.debug else len(train_paths)
    print(f'### WRITING {write_files} TRAIN IMAGES', end=' ')
    info     = Parallel(n_jobs=-1,
                       verbose=0,
                       backend='threading')(delayed(resize_and_save)(file_path,dim=opt.img_size)\
                                            for file_path in tqdm(train_paths[:write_files],
                                                                  desc='writing '))
    image_id, width, height = list(zip(*info))
    tmp_df = pd.DataFrame({'image_id':image_id,
                           'width':width,
                           'height':height
                          })
    train_df = pd.merge(train_df, tmp_df, on = 'image_id', how = 'left')
    checked()
    print(f'### WRITING {CLEAN_DATA_DIR}/train.csv', end=' ')
    train_df.to_csv(os.path.join(CLEAN_DATA_DIR, 'train.csv'), index=False)
    checked(); print()

    # test data
    test_paths = glob(os.path.join(RAW_DATA_DIR,'test/**/*dcm'),recursive=True)
    test_df = pd.DataFrame({'filepath':test_paths,})
    test_df['image_id'] = test_df.filepath.map(lambda x: x.split('/')[-1].replace('.dcm', '')+'_image')
    test_df['study_id'] = test_df.filepath.map(lambda x: x.split('/')[-3].replace('.dcm', '')+'_study')
    print(f'### TEST IMAGES FOUND: {test_df.shape[0]}');

    write_files = 100 if opt.debug else len(test_paths)
    print(f'### WRITING {write_files} TEST IMAGES', end=' ')
    info = Parallel(n_jobs=-1, 
                    verbose=0, 
                    backend='threading')(delayed(resize_and_save)(file_path,dim=opt.img_size)\
                                         for file_path in tqdm(test_paths[:write_files], 
                                                               desc='writing '))
    image_id, width, height = list(zip(*info))
    tmp_df = pd.DataFrame({'image_id':image_id,
                       'width':width,
                       'height':height})
    test_df = pd.merge(test_df, tmp_df, on = 'image_id', how = 'left')
    checked()
    print(f'### WRITING {CLEAN_DATA_DIR}/test.csv', end=' ')
    test_df.to_csv(os.path.join(CLEAN_DATA_DIR, 'test.csv'),index=False)
    checked()
    
    # all done
    print('\n### CLEAN DATA IS READY!', '\U0001F603')