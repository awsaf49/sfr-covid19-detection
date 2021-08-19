
import sys,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse,shutil
import pandas as pd, numpy as np, random,os, shutil
from glob import glob
from tqdm import tqdm
import json
tqdm.pandas()
from kaggle_datasets import KaggleDatasets
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import gc

## global parameters

class_names = ["Negative for Pneumonia",
              "Indeterminate Appearance",
              "Atypical Appearance",
              "Typical Appearance"]
class_labels= [0, 1, 2, 3]
name2label = dict(zip(class_names, class_labels))
label2name = {v:k for k, v in name2label.items()}


#-----------------------------------------	
# VISUALIZATION
#-----------------------------------------

def display_batch(batch, save_path,size=2):
    imgs, tars = batch
    plt.figure(figsize=(size*5, 5))
    for img_idx in range(size):
        plt.subplot(1, size, img_idx+1)
        plt.title(f'class: {label2name[tars[img_idx].numpy().argmax()]}', fontsize=15)
        plt.imshow(imgs[img_idx,:, :, :])
        plt.xticks([])
        plt.yticks([])
    plt.savefig(save_path)
    #plt.show() 
    
#-----------------------------------------
# SEEDING AND ACCELERATOR 
#-----------------------------------------


def seeding(SEED,tpu=True):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    if tpu:
        os.environ["TF_CUDNN_DETERMINISTIC"] = str(SEED)
    tf.random.set_seed(SEED)
    print("seeding done!!!")

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
        device="TPU"
    except ValueError:
        strategy = tf.distribute.get_strategy()
        device="GPU"
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy,device



#-----------------------------------------
# AUGMENTATIONS
#-----------------------------------------



def get_mat(shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
        
    # CONVERT DEGREES TO RADIANS
    #rotation = math.pi * rotation / 180.
    shear    = math.pi * shear / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX

    one  = tf.constant([1],dtype="float32")
    zero = tf.constant([0],dtype="float32")
      
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
    

    return  K.dot(shear_matrix,K.dot(zoom_matrix, shift_matrix))             

def transform(image, DIM=[512,512]):
    if DIM[0]!=DIM[1]:
        pad = (DIM[0]-DIM[1])//2
        image = tf.pad(image, [[0, 0], [pad, pad+1],[0, 0]])
        
    NEW_DIM = DIM[0]
    
    rot = params["ROT_"] * tf.random.normal([1], dtype="float32")
    shr = params["SHR_"] * tf.random.normal([1], dtype="float32") 
    h_zoom = 1.0 + tf.random.normal([1], dtype="float32") / params["HZOOM_"]
    w_zoom = 1.0 + tf.random.normal([1], dtype="float32") / params["WZOOM_"]
    h_shift = params["HSHIFT_"] * tf.random.normal([1], dtype="float32") 
    w_shift = params["WSHIFT_"] * tf.random.normal([1], dtype="float32") 
    
    transformation_matrix=tf.linalg.inv(get_mat(shr,h_zoom,w_zoom,h_shift,w_shift))
    
    flat_tensor=tfa.image.transform_ops.matrices_to_flat_transforms(transformation_matrix)
    
    image=tfa.image.transform(image,flat_tensor, fill_mode=params["FILL_MODE"])
    
    rotation = math.pi * rot / 180.
    
    image=tfa.image.rotate(image,-rotation, fill_mode=params["FILL_MODE"])
    
    if DIM[0]!=DIM[1]:
        image=tf.reshape(image, [NEW_DIM, NEW_DIM,3])
        image = image[:, pad:DIM[1]+pad,:]
    image = tf.reshape(image, [*DIM, 3])    
    return image

def dropout(image,DIM=[512,612], PROBABILITY = 0.6, CT = 5, SZ = 0.1):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image with CT squares of side size SZ*DIM removed
    
    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast( tf.random.uniform([],0,1)<PROBABILITY, tf.int32)
    if (P==0)|(CT==0)|(SZ==0): 
        return image
    
    for k in range(CT):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM[1]),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM[0]),tf.int32)
        # COMPUTE SQUARE 
        WIDTH = tf.cast( SZ*min(DIM),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM[0],y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM[1],x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        two = tf.zeros([yb-ya,xb-xa,3], dtype = image.dtype) 
        three = image[ya:yb,xb:DIM[1],:]
        middle = tf.concat([one,two,three],axis=1)
        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM[0],:,:]],axis=0)
        image = tf.reshape(image,[*DIM,3])

#     image = tf.reshape(image,[*DIM,3])
    return image



#-----------------------------------------
# DATASET
#-----------------------------------------


def build_decoder(with_labels=True, target_size=[512,612], ext="png"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        elif ext == "npy":
            dtype = tf.float64
            img = tf.io.decode_raw(file_bytes, out_type=dtype)
            img = img[:1024//int(str(dtype)[14:16])]
            img = tf.reshape(img, target_size)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        
        img=tf.image.resize(img,target_size)
        
        img = tf.reshape(img, [*target_size, 3])

        return img
    
    def decode_with_labels(path, label):
        return decode(path), tf.cast(label, tf.float32)
    
    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True, dim=[512,612]):
    def augment(img, dim=dim):
        img = transform(img,DIM=dim) if params["TRANSFORM"] else img
        img = tf.image.random_flip_left_right(img) if params["H_FLIP"] else img
        img = tf.image.random_flip_up_down(img) if params["V_FLIP"] else img
        img = tf.image.random_saturation(img, params["sat"][0], params["sat"][1])
        img = tf.image.random_contrast(img, params['cont'][0], params['cont'][1])
        img = tf.image.random_brightness(img, params["bri"])
        img = dropout(img, DIM=dim, PROBABILITY = params["PROBABILITY"],CT = params["CT"], SZ = params["SZ"])
        img = tf.clip_by_value(img, 0, 1)  if params["CLIP"] else img         
        img = tf.reshape(img, [*dim, 3])
        return img
    
    def augment_with_labels(img, label):    
        return augment(img), label
    
    return augment_with_labels if with_labels else augment

def onehot(image,label):
    return image,tf.one_hot(label,NUM_CLASS)

def build_dataset(paths, labels=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir="", drop_remainder=False):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None,target_size=params["IMG_SIZES"][0])
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None,dim=params["IMG_SIZES"][0])
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=params["SEED"])
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTO)
    return ds


#-----------------------------------------
# MODEL
#-----------------------------------------


import efficientnet.tfkeras as efn

name2effnet = {
    'efficientnet_b0': efn.EfficientNetB0,
    'efficientnet_b1': efn.EfficientNetB1,
    'efficientnet_b2': efn.EfficientNetB2,
    'efficientnet_b3': efn.EfficientNetB3,
    'efficientnet_b4': efn.EfficientNetB4,
    'efficientnet_b5': efn.EfficientNetB5,
    'efficientnet_b6': efn.EfficientNetB6,
    'efficientnet_b7': efn.EfficientNetB7,
}

def build_model(dim=[512,512], model_name='efficientnet_b0',weights="imagenet",compile_model=True,
                loss="CategoricalCrossentropy",ls=0.01):
    inp = tf.keras.layers.Input(shape=(*dim,3))
    if '.h5' not in weights:
        base = name2effnet[model_name](input_shape=(*dim,3),weights=weights,include_top=False)
    else:
        base=tf.keras.models.load_model(weights, compile=False).layers[1]
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation = "relu")(x)
    x = tf.keras.layers.Dense(NUM_CLASS,activation="softmax")(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    if compile_model:
        #optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        #loss
        loss = getattr(tf.keras.losses, loss)(label_smoothing=ls)
        #metric
        auc = tf.keras.metrics.AUC(curve="ROC", 
                                   multi_label=True,)
        acc = tf.keras.metrics.CategoricalAccuracy()
        f1  = tfa.metrics.F1Score(num_classes=NUM_CLASS,average="macro",threshold=None)

        model.compile(optimizer=opt,loss=loss,metrics=[acc, auc, f1])
    return model


#-----------------------------------------
# Callback
#-----------------------------------------
class EpochSave(tf.keras.callbacks.Callback):
    def __init__(self,save_epoch,filepath,save_weights_only=False,verbose=1):
        self.save_epoch=save_epoch-1
        self.verbose=verbose
        self.filepath=filepath
        self.save_weights_only=save_weights_only
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch==self.save_epoch:
            if self.save_weights_only:
                self.model.save_weights(self.filepath)
            else:
                self.model.save(self.filepath)
            if self.verbose==1:
                print(f'Saved model at {self.filepath}')


def get_lr_callback(batch_size=8, plot=False):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * REPLICAS * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--TRAIN_DATA_CLEAN_PATH", type=str, default="/tmp/Dataset/", help="directory of images")
    parser.add_argument("--CHECKPOINT_DIR", type=str, default="imagenet", help="location of checkpoint file")
    parser.add_argument("--MODEL_DIR", type=str, default="classification/runs_4cls/weight", help="where to save model after training")
    parser.add_argument("--cfg", type=str,default="classification/4cls_cfg.json" ,help="config path")
    parser.add_argument("--debug", action="store_true", help="train on only first 1000 images")
    parser.add_argument("--dim", type=int,default=512, help="img_dimension")
    parser.add_argument("--model_name", type=str,default='efficientnet_b0', help="model name")
    parser.add_argument("--epochs", type=int,default=10, help="how many epochs to run")
    parser.add_argument("--save_epoch", type=int,default=-1, help="which epoch to save as")
    parser.add_argument("--bs", type=int,default=32, help="batch size per replica")
    opt = parser.parse_args()
    
    
    DEBUG=opt.debug
    BATCH_SIZES=[opt.bs]
    NUM_CLASS=4 #opt.num_class
    if '.h5' in opt.MODEL_DIR:
        MODEL_DIR=os.path.dirname(opt.MODEL_DIR)
        MODEL_SAVENAME=os.path.basename(opt.MODEL_DIR)
    else:
        MODEL_DIR=opt.MODEL_DIR
        MODEL_SAVENAME='model.h5'
    PRETRAIN=opt.CHECKPOINT_DIR
    TRAIN_PATH=opt.TRAIN_DATA_CLEAN_PATH
    MODEL_NAME=opt.model_name
    SAVE_EPOCH=opt.save_epoch
    
    params = json.load(open(opt.cfg, "r"))
    params["IMG_SIZES"]=[[opt.dim,opt.dim]]
    params["EPOCHS"]= [opt.epochs]
    DIM=params["IMG_SIZES"][0]
    
    
    print('##',' ','Setting up seed and accelerator',' ','##\n')
    strategy,device = auto_select_accelerator()
    seeding(params["SEED"],device=='TPU')
    AUTO     = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    DISPLAY_PLOT=True
    print()
    
    
    #-----------------------------------------
    #-----------------------------------------
    ## DATASET FIXUP
    #-----------------------------------------
    #-----------------------------------------

    
    #df = #pd.read_csv('./data/meta/train_dupicate.csv')
    df = pd.read_csv(os.path.join('.','data','meta','train_duplicate.csv'))
    fold_df = pd.read_csv(os.path.join('.','data','meta','scd_fold.csv'))
    fold_df['StudyInstanceUID'] = fold_df['image_id'].map(dict(df[['image_id', 'StudyInstanceUID']].values))
    study2fold = dict(fold_df[['StudyInstanceUID', 'fold']].values)
    df['fold'] = df['StudyInstanceUID'].map(study2fold)
    df['study_id'] = df['StudyInstanceUID']
    
    print('##',' ','Modifying dataframe',' ', '##','\n')
    print('Before size:',df.shape[0])

    # remove duplicates
    dup_0 = df.query("dup_id==0") # take all from non-duplicates
    dup_1 = df.query("dup_id>0").groupby("StudyInstanceUID").head(1) # take one from duplicates
    df    = pd.concat((dup_0, dup_1), axis=0)
    print('After removal size:',df.shape[0])
    
    df["image_path"]       = TRAIN_PATH+os.sep+ "train" + os.sep+df.image_id+".png"
    tqdm.pandas(desc="Mapping labels  ")
    df["class_name"]  = df.progress_apply(lambda row:row[class_names].iloc[[row[class_names].values.argmax()]].index.tolist()[0], 
                                 axis=1)
    df["class_label"] = df.class_name.map(name2label)
    print()
    
    
    #-----------------------------------------
    #-----------------------------------------
    ## Save a sample
    #-----------------------------------------
    #-----------------------------------------

    temp_df = df[:1000]
    paths  = temp_df.image_path.tolist()
    labels = temp_df[class_names].values
    ds = build_dataset(paths, labels, cache=False, batch_size=BATCH_SIZES[0]*REPLICAS,
                       repeat=True, shuffle=False, augment=True)
    ds = ds.unbatch().batch(20)
    batch = next(iter(ds))
    save_loc=os.path.join(MODEL_DIR,"sample_image.png")
    display_batch(batch, os.path.join("sample_image.png"),5)
    del batch,ds,temp_df,labels,paths,save_loc
    z=gc.collect()
    
    
    #-----------------------------------------
    #-----------------------------------------
    ## Training
    #-----------------------------------------
    #-----------------------------------------
    
    fold=0              
    train_df = df
     
    if DEBUG:
        train_df  = train_df.iloc[:1000]

    train_paths = train_df.image_path.values; train_labels = train_df[class_names].values.astype(np.float32)
       
    index = np.arange(len(train_paths))
    np.random.shuffle(index)
    train_paths  = train_paths[index]
    train_labels = train_labels[index]
    

    print("## Train image Size: (%i, %i) | batch_size %i | num_images %i | model_name %s | weights %s ##\n"%
          (params["IMG_SIZES"][fold][0],params["IMG_SIZES"][fold][1],BATCH_SIZES[fold]*REPLICAS,len(train_paths),MODEL_NAME, PRETRAIN))
  
    
    K.clear_session()
    with strategy.scope():
        model = build_model(model_name=MODEL_NAME,dim=DIM, weights=PRETRAIN,compile_model=True)
        
    train_ds = build_dataset(train_paths, train_labels, cache= True, batch_size=BATCH_SIZES[fold]*REPLICAS,
                   repeat=True, shuffle=True, augment=params["AUGMENT"])
    
    
    
    os.makedirs(MODEL_DIR,exist_ok=True)
    sv = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR,"model-%s_epoch-{epoch:02d}.h5"%(MODEL_NAME)),
            monitor="auc", verbose=0, save_best_only=False,
            save_weights_only=False, mode="max", save_freq="epoch",)
    
    callbacks = [sv,get_lr_callback(BATCH_SIZES[fold])]
    if SAVE_EPOCH!=-1:
        save= EpochSave(save_epoch=SAVE_EPOCH,filepath=os.path.join(MODEL_DIR,MODEL_SAVENAME))
        callbacks.append(save)
        
    print()
    history = model.fit(
        train_ds, 
        epochs=params["EPOCHS"][fold] if not DEBUG else 2, 
        callbacks = callbacks, 
        steps_per_epoch=len(train_paths)/BATCH_SIZES[fold]/REPLICAS,
        verbose=params["VERBOSE"]
    )
    if SAVE_EPOCH==-1:
        print('Saving final model...')
        model.save(os.path.join(MODEL_DIR,MODEL_SAVENAME))
    
    if DISPLAY_PLOT:
        plt.figure(figsize=(15,5))
        plt.plot(np.arange(len(history.history["auc"])),history.history["auc"],"-o",label="Train auc",color="#ff7f0e")
        x = np.argmax( history.history["auc"] ); y = np.max( history.history["auc"] )
        xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x,y,s=200,color="#1f77b4"); plt.text(x-0.03*xdist,y-0.13*ydist,"max auc\n%.2f"%y,size=14)
        plt.ylabel("auc",size=14); plt.xlabel("Epoch",size=14)
        plt.legend(loc=2)
        plt2 = plt.gca().twinx()
        plt2.plot(np.arange(len(history.history["auc"])),history.history["loss"],"-o",label="Train Loss",color="#2ca02c")
        x = np.argmin( history.history["loss"] ); y = np.min( history.history["loss"] )
        ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x,y,s=200,color="#d62728"); plt.text(x-0.03*xdist,y+0.05*ydist,"min loss",size=14)
        plt.ylabel("Loss",size=14)
        plt.legend(loc=3)
        plt.savefig(os.path.join(MODEL_DIR,"loss_plot_4cls.png"))
        #plt.show()
    
