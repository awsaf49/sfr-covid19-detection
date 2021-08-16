# check functions
def checked():
    print(u'[\u2713]')
def failed():
    print(u"[\u2715]")

# import os
# os.system("pip install -q efficientnet")

print('### IMPORTING LIBRARIES', end=' ')
import argparse
import numpy as np, pandas as pd
from glob import glob
import os, shutil
import efficientnet.tfkeras as efn
import tensorflow as tf, gc
import os
from tqdm import tqdm
tqdm.pandas()
import math, re
import tensorflow.keras.backend as K
import json
checked()

# Augmentation
IMG_SIZES = [[512, 512]]
DIM=IMG_SIZES[0]


sat  = (0.7, 1.3)
cont = (0.8, 1.2)
bri  =  0.1
ROT_    = 0.0
SHR_    = 2.0
HZOOM_  = 8.0
WZOOM_  = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, DIM=IMG_SIZES[0]):    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    
    # fixed for non-square image thanks to Chris Deotte
    
    if DIM[0]!=DIM[1]:
        pad = (DIM[0]-DIM[1])//2
        image = tf.pad(image, [[0, 0], [pad, pad+1],[0, 0]])
        
    NEW_DIM = DIM[0]
    
    XDIM = NEW_DIM%2 #fix for size 331
    
    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(NEW_DIM//2, -NEW_DIM//2,-1), NEW_DIM)
    y   = tf.tile(tf.range(-NEW_DIM//2, NEW_DIM//2), [NEW_DIM])
    z   = tf.ones([NEW_DIM*NEW_DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -NEW_DIM//2+XDIM+1, NEW_DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([NEW_DIM//2-idx2[0,], NEW_DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
    
    if DIM[0]!=DIM[1]:
        image = tf.reshape(d,[NEW_DIM, NEW_DIM,3])
        image = image[:, pad:DIM[1]+pad,:]
    image = tf.reshape(image, [*DIM, 3])
        
    return image


# device selector
def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    
    return strategy

# decoder
def build_decoder(with_labels=True, target_size=(300, 300), ext='png'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, target_size, method='area')
#         img = tf.image.resize(img, target_size)
        img = img/255.0

        return img
    
    def decode_with_labels(path, label):
        return decode(path), label
    
    return decode_with_labels if with_labels else decode

## augmenter
def build_augmenter(with_labels=True, DIM=[512, 512]):
    def augment(img):
        img = transform(img, DIM = DIM)
        img = tf.image.random_flip_left_right(img)
#         img = tf.image.random_flip_up_down(img)
        img = tf.image.random_saturation(img, sat[0], sat[1])
        img = tf.image.random_contrast(img, cont[0], cont[1])
        img = tf.image.random_brightness(img, bri)
        return img
    
    def augment_with_labels(img, label):
        return augment(img), label
    
    return augment_with_labels if with_labels else augment

# data loader
def build_dataset(paths, labels=None, bsize=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)
    
    return dset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='fast', 
                        help='use provided weights in `fast` mode else newly trained weights`full` mode')
    parser.add_argument('--debug', type=int, default=0, help="infer on only first 100 images")
    parser.add_argument('--tta', type=int, default=1)
    opt = parser.parse_args()
    
    # getting args
    DEBUG = opt.debug
    TTA   = opt.tta
    MODE  = opt.mode
    
    # settings.json
    print('### LOADING SETTINGS.json', end=' ')
    cfg = json.load(open('SETTINGS.json', 'r'))
    checked(); print()
    
    if MODE=='fast':
        CHECKPOINT_DIR = cfg['CHECKPOINT_DIR']
    elif MODE=='full':
        CHECKPOINT_DIR = cfg['MODEL_DIR']
    else:
        raise ValueError('mode is neither `fast` nor `full`')
        
    MODEL_DIRS4 = os.path.join(CHECKPOINT_DIR, '4cls', '04')
    print(F'### 4 CLASS MODELS DIRECTORY: {MODEL_DIRS4}')
    MODEL_DIRS2 = os.path.join(CHECKPOINT_DIR, '2cls')
    print(F'### 2 CLASS MODELS DIRECTORY: {MODEL_DIRS2}')
    print()
    
    # test.csv
    TEST_CSV = cfg['TEST_CSV_PATH']
    print(f'### READING {TEST_CSV}', end=' ')
    test_df = pd.read_csv(TEST_CSV)
    checked()
    
    # test_directory
    DATA_DIR = cfg['TEST_DATA_CLEAN_PATH']
    
    # set zeros to class_labels as dummy values
    CLASS_LABELS  = ['0', '1', '2', '3']
    CLASS_LABELS2 = ['opacity']
    test_df.loc[:,CLASS_LABELS]  = 0
    test_df.loc[:,CLASS_LABELS2] = 0
    
    # select accelerator
    strategy   = auto_select_accelerator()
    print('### ACCELERATOR SELECTION', end=' ')
    checked()
    
    # test image paths
    test_paths = test_df.image_id.map(lambda x: os.path.join(DATA_DIR, x.replace('_image','')+'.png'))
    test_paths = test_paths.tolist()
    test_paths = test_paths if not DEBUG else test_paths[:100]
    
    # predict 4 class
    print('\n### PREDICTING [4 CLASS] :')
    preds=[]
    for model_dir in [MODEL_DIRS4]:
        model_paths = glob(os.path.join(model_dir, '**/*h5'), recursive=True)
        for model_path in tqdm(model_paths,):
            # loading model
            with strategy.scope():
                model = tf.keras.models.load_model(model_path, compile=False)
            # image-paths
            dim = np.array(model.input.shape[1:])[0]
            # batch-size
            if 'aux' in model_paths[0]:
                BATCH_SIZE = strategy.num_replicas_in_sync * 16
            elif dim>=768:
                BATCH_SIZE = strategy.num_replicas_in_sync * 24
            elif dim>=640:
                BATCH_SIZE = strategy.num_replicas_in_sync * 32
            else:
                BATCH_SIZE = strategy.num_replicas_in_sync * 64
            # dataset
            dtest = build_dataset(
                test_paths, 
                bsize=BATCH_SIZE, repeat=True, 
                shuffle=False, augment=True if TTA>1 else False, cache=False,
                decode_fn=build_decoder(with_labels=False, target_size=[dim,dim]),
                augment_fn=build_augmenter(with_labels=False, DIM=[dim, dim])
            )
            pred = model.predict(dtest, steps = TTA*len(test_paths)/BATCH_SIZE, verbose=0)
            pred = pred['label'] if isinstance(pred, dict) else pred # for aux_loss
            pred = pred[:TTA*len(test_paths),:]
            pred = np.mean(pred.reshape(TTA, len(test_paths), -1), axis=0)
            preds.append(pred)
    preds = np.mean(preds, axis=0)
    
    # predict 2 class
    print()
    print('### PREDICTING [2 CLASS] :')
    preds2=[]
    for model_dir in [MODEL_DIRS2]:
        model_paths = glob(os.path.join(model_dir, '**/*h5'), recursive=True)
        for model_path in tqdm(model_paths,):
            with strategy.scope():
                model = tf.keras.models.load_model(model_path, compile=False)
            # batch-size
            if 'aux' in model_paths[0]:
                BATCH_SIZE = strategy.num_replicas_in_sync * 16
            elif dim>=768:
                BATCH_SIZE = strategy.num_replicas_in_sync * 24
            elif dim>=640:
                BATCH_SIZE = strategy.num_replicas_in_sync * 32
            else:
                BATCH_SIZE = strategy.num_replicas_in_sync * 64
            # dataset
            dtest = build_dataset(
                test_paths, 
                bsize=BATCH_SIZE, repeat=True, 
                shuffle=False, augment=True if TTA>1 else False, cache=False,
                decode_fn=build_decoder(with_labels=False, target_size=[dim,dim]),
                augment_fn=build_augmenter(with_labels=False, DIM=[dim, dim])
            )
            pred = model.predict(dtest, steps = TTA*len(test_paths)/BATCH_SIZE, verbose=0)[:TTA*len(test_paths),:]
            pred = np.mean(pred.reshape(TTA, len(test_paths), -1), axis=0)
            preds2.append(pred)
    preds2 = np.mean(preds2, axis=0)
    
    # replacing zero with actual prediction
    test_df.loc[:99 if DEBUG else test_df.shape[0],CLASS_LABELS]  = preds
    test_df.loc[:99 if DEBUG else test_df.shape[0],CLASS_LABELS2] = preds2
    
    # saving prediction
    SUB_DIR  = cfg['SUBMISSION_DIR']
    os.makedirs(SUB_DIR, exist_ok=True)
    PRED_CSV = os.path.join(SUB_DIR, 'image_cls.csv')
    print(f'### SAVING {PRED_CSV}', end=' ')
    test_df.to_csv(PRED_CSV,index=False)
    checked()
    
    print('\n### STUDY PREDICTION IS DONE!','\U0001F603')