




import timm
from timm.models.layers import SelectAdaptivePool2d
import numpy as np
import pandas as pd
import os, shutil
import sys
import json
import matplotlib.pyplot as plt 
import random
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import time
from datetime import datetime
from tqdm.autonotebook import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import argparse
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
import torch.nn.functional as F
from glob import glob
import warnings
warnings.filterwarnings("ignore") 
warnings.filterwarnings("ignore", category=DeprecationWarning)
           
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--CONFIG", type=str, default="yolov5x-tr.yaml", help="config of yolo model")
    parser.add_argument("--DEBUG", action='store_true', help="is debug")
    
    
    
    opt = parser.parse_args()
    if opt.CONFIG == "yolov5x-tr.yaml":
        CONFIG = './yolov5/models/yolov5x-tr.yaml'
        MODEL = 'yolov5x'
        WEIGHTS = MODEL + ".pt"
        END_MODEL_POINT = 10
    elif opt.CONFIG == "yolov3-spp.yaml":
        CONFIG = './yolov5/models/yolov3-spp.yaml'
        MODEL = 'yolov3-spp'
        WEIGHTS = MODEL + ".pt"
        END_MODEL_POINT = 11
    elif opt.CONFIG == "yolov5x6.yaml":
        CONFIG = './yolov5/models/yolov5x6.yaml'
        MODEL = 'yolov5x'
        WEIGHTS = MODEL + ".pt"
        END_MODEL_POINT = 12
    else:
        print("Invalid config")
        
    PRETRAIN = 'imagenet'
    DATA = 'chexpert_14'
    PRETRAINED = True
    FLEXIBLE = False
    EXTRA_FC = True
    FOLD = -1
    BATCH_SIZE = 8
    AC_STEP = 8
    IMAGE_SIZE = 512    
    EPOCHS = 30
    TRAIN_TIME = 420 # mins
    NUM_CLASSES = 14
    
    from_previous_checkpoint = False
    checkpoint_path = None
    REPLACE_SILU = False
    
    config = {
        "weight" : PRETRAIN,
        "architecture" : MODEL,
        "optimizer" : "Adam",
        "lr" : 5e-4,
        "scheduler" : "chris",
        "epochs" : EPOCHS,
        "augs" : "simple",
        "batch_size" : BATCH_SIZE,
        "loss_function" : "bce",
        'image_size' : IMAGE_SIZE,
        "fold" : FOLD,
        "config": CONFIG
    }

    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    seed_everything(42)

    
#     shutil.copytree("/kaggle/input/srf-covid-19-detection/sfr-covid19-detection/yolov5", "./yolo/yolov5")
#     os.chdir('/kaggle/working/yolo/yolov5')
#     pip install -qr requirements.txt # install dependencies
#     os.chdir('/kaggle/working')
    
    sys.path.append('./yolov5')
    from models.yolo import Model
    from od import Model_flexible
    from utils.google_utils import attempt_download
    from utils.torch_utils import intersect_dicts, torch_distributed_zero_first
    
    
    
    def get_pretrain(model, weights, device):
        rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        exclude = ['anchor']
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        print("matching key length : ", len(state_dict))
        model.load_state_dict(state_dict, strict=False)  # load
        return model
    
    
    
    
    def get_model():
        if FLEXIBLE:
            model = Model_flexible(CONFIG, pretrain=PRETRAINED)
            model.fpn = nn.Identity()
            model.pan = nn.Identity()
            model.detection = nn.Identity()
        else:
            model = Model(cfg=CONFIG, ch=3, nc=4, anchors=True)
            if PRETRAINED:
                model = get_pretrain(model, WEIGHTS, device)
            model.model = model.model[0:END_MODEL_POINT] # 10 for v5, 12 for v5x6, 11 for v3-spp
        return model
    
    
    
    class Mish_func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, i):
            result = i * torch.tanh(F.softplus(i))
            ctx.save_for_backward(i)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            i = ctx.saved_tensors[0]

            v = 1. + i.exp()
            h = v.log() 
            grad_gh = 1./h.cosh().pow_(2) 

            # Note that grad_hv * grad_vx = sigmoid(x)
            #grad_hv = 1./v  
            #grad_vx = i.exp()

            grad_hx = i.sigmoid()

            grad_gx = grad_gh *  grad_hx #grad_hv * grad_vx 

            grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx 

            return grad_output * grad_f 


    class Mish(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

            print("Mish initialized")
            pass
        def forward(self, input_tensor):
            return Mish_func.apply(input_tensor)


        
        
    def replace_activations(model, existing_layer, new_layer):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                model._modules[name] = replace_activations(module, existing_layer, new_layer)

            if type(module) == existing_layer:
                layer_old = module
                layer_new = new_layer
                model._modules[name] = layer_new
        return model
    
    
    
    class Model_Classifier(nn.Module):
        def __init__(self,
                     n_classes,
                     model_name='efficientnet_b0',
                     extra_fc=False,
                     pretrained=True,
                     replace_silu=False,
                     flexible=FLEXIBLE):

            super(Model_Classifier, self).__init__()
            print('Building Model Backbone for {} model'.format(model_name))

            self.model = get_model()
            if flexible:
                LAST_LAYER_OUT = self.model(torch.randn(1, 3, 256, 256))[-1].shape[1]
            else:
                LAST_LAYER_OUT = self.model(torch.randn(1, 3, 256, 256)).shape[1]

            out_fc1 = 512 if extra_fc else n_classes
            self.extra_fc = extra_fc

            self.pooling = SelectAdaptivePool2d(pool_type='avg', flatten=True)
            self.out1 = nn.Linear(LAST_LAYER_OUT, out_fc1, bias=True)

            # handle out fc
            if extra_fc:
                print("Using extra fc : ", out_fc1)
                self.relu = nn.ReLU(inplace=True)
                self.out2 = nn.Linear(out_fc1, n_classes, bias=True)

            # replace silu
            if replace_silu:
                print("replacing silu with mish activations")
                existing_layer = torch.nn.SiLU
                new_layer = Mish()
                self.model = replace_activations(self.backbone, existing_layer, new_layer) # in eca_nfnet_l0 SiLU() is used, but it will be replace by Mish()

        def forward(self, x):
            if FLEXIBLE:
                out = self.model(x)[-1]
            else:
                out = self.model(x)

            out = self.out1(self.pooling(out))
            if self.extra_fc:
                out = self.out2(self.relu(out))
            return out
        
    
#     model = Model_Classifier(
#         n_classes = NUM_CLASSES,
#         model_name = MODEL,
#         extra_fc = EXTRA_FC,
#         pretrained = PRETRAINED,
#         replace_silu = REPLACE_SILU
#     )

    
    with open("./SETTINGS.json") as f:
        data = json.load(f)
    
    train_df = pd.read_csv(f"{data['ROOT_CHEXPERT_DIR']}/train.csv")
    train_df = train_df.fillna(0)
    val_df = pd.read_csv(f"{data['ROOT_CHEXPERT_DIR']}/valid.csv")
    val_df = val_df.fillna(0)
    df = train_df.append(val_df)
    df['Path'] = df['Path'].apply(lambda x : f"{data['ROOT_CHEXPERT_DIR']}/" + '/'.join(x.split('/')[1:]))
    df = df[~df[df.columns[3]].str.contains("Lateral")]
    df = df.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)
    df = df.replace(-1,1)
    df['patient_id'] = df.Path.map(lambda x: x.split('/')[6])
    df['study'] = df.Path.map(lambda x: x.split('/')[7])
    target_cols = df.columns[1:-2].tolist()
    gkf = GroupKFold(n_splits = 5)
    df.reset_index(drop = True, inplace = True)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df, groups = df.patient_id.tolist())):
        df.loc[val_idx, 'fold'] = fold
    
    SSR_PROB = 0.7
    SHIFT_LIMIT = 0.2
    SCALE_LIMIT = 0.2
    ROTATE_LIMIT = 0

    def get_train_transform():
        return A.Compose([              
                    A.HorizontalFlip(p=0.5),
                   A.ShiftScaleRotate(
                       shift_limit=SHIFT_LIMIT, scale_limit=SCALE_LIMIT, rotate_limit=ROTATE_LIMIT, border_mode=cv2.BORDER_CONSTANT,p=SSR_PROB
                    ),
                   A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=8, val_shift_limit=8, p=0.7),
                   A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                  A.Cutout(max_h_size=int(IMAGE_SIZE * 0.1), max_w_size=int(IMAGE_SIZE * 0.1), num_holes=5, p=0.5),
                  ToTensorV2(p=1.0),
            ])


    def get_test_transform():
        return A.Compose([              
                    ToTensorV2(p=1.0),
            ])


    class RANZERDataset(Dataset):
        def __init__(self, df, transforms=None):
            self.df = df.reset_index(drop=True)
            self.transforms = transforms
            self.labels = df[target_cols].values

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            row = self.df.loc[index]
            img = cv2.imread(row.Path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


            if self.transforms is not None:
                augmented = self.transforms(image=img)
                img = augmented['image']

            img = img.float()
            img /= 255.0
            label = torch.tensor(self.labels[index]).float()
            return img, label  
    
    
    if opt.DEBUG:
        df = df.sample(frac=0.001)

    if FOLD != -1:
        df_train_this = df[df['fold'] != FOLD]
        df_valid_this = df[df['fold'] == FOLD]
    else:
        df_train_this = df
        df_valid_this = df[df['fold'] == 0]

    train_dataset = RANZERDataset(
        df_train_this, 
        transforms=get_train_transform()
    )

    val_dataset = RANZERDataset(
        df_valid_this, 
        transforms=get_test_transform()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=2,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=2
    )


    class CustomLRScheduler(_LRScheduler):
        def __init__(self, optimizer, lr_start=5e-6, lr_max=1e-5,
                     lr_min=1e-6, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.8,
                     last_epoch=-1):
            self.lr_start = lr_start
            self.lr_max = lr_max
            self.lr_min = lr_min
            self.lr_ramp_ep = lr_ramp_ep
            self.lr_sus_ep = lr_sus_ep
            self.lr_decay = lr_decay
            super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            if not self._get_lr_called_within_step:
                warnings.warn("To get the last learning rate computed by the scheduler, "
                              "please use `get_last_lr()`.", UserWarning)

            if self.last_epoch == 0:
                self.last_epoch += 1
                return [self.lr_start for _ in self.optimizer.param_groups]

            lr = self._compute_lr_from_epoch()
            self.last_epoch += 1

            return [lr for _ in self.optimizer.param_groups]

        def _get_closed_form_lr(self):
            return self.base_lrs

        def _compute_lr_from_epoch(self):
            if self.last_epoch < self.lr_ramp_ep:
                lr = ((self.lr_max - self.lr_start) / 
                      self.lr_ramp_ep * self.last_epoch + 
                      self.lr_start)

            elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
                lr = self.lr_max

            else:
                lr = ((self.lr_max - self.lr_min) * self.lr_decay**
                      (self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep) + 
                      self.lr_min)
            return lr

        def set_epoch(self, epoch):
            self.last_epoch = epoch
            
            
    class Train_Config:
        batch_size = config['batch_size']
        n_epochs = config['epochs']
        lr =  config['lr']
        accumulation = AC_STEP

        folder = 'Output'
        verbose = True


        ########## SCHEDULER ##############

        step_scheduler = False # do scheduler.step after optimizer.step ( for one cycle )
        validation_scheduler = False  # do scheduler.step after validation stage loss ( for reduce on plateau )
        custom_step_scheduler = False # for customDecay takes epoch as input

        if config['scheduler'] == 'onecycle':
            SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
            scheduler_params = dict(
                max_lr=0.001,
                epochs=n_epochs,
                steps_per_epoch=int(len(train_dataset) / batch_size),
                pct_start=0.1,
                anneal_strategy='cos', 
                final_div_factor=10**5
            )
            step_scheduler = True

        if config['scheduler'] == 'reduce':
            SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
            scheduler_params = dict(
                mode='min',
                factor=0.5,
                patience=2,
                verbose=False, 
                threshold=0.0001,
                threshold_mode='abs',
                cooldown=0, 
                min_lr=1e-8,
                eps=1e-08
            )
            validation_scheduler = True


        if config['scheduler'] == 'chris':
            SchedulerClass = CustomLRScheduler
            scheduler_params = {
                "lr_start": 1e-5,
                "lr_max": 1e-5 * 32, # BATCH_SIZE
                "lr_min": 1e-6,
                "lr_ramp_ep": 5,
                "lr_sus_ep": 0,
                "lr_decay": 0.8,
            }
            custom_step_scheduler = True
            
            
    class AverageMeter(object):
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count



    def macro_multilabel_auc(label, pred):
        aucs = []
        for i in range(len(target_cols)):
            try:
                val = roc_auc_score(label[:, i], pred[:, i])
            except:
                print('No val')
                val = 0
            aucs.append(val)
        return np.mean(aucs)
    
    
    
    
    
    class Fitter:
        def __init__(self, model, device, config):
            self.accumulation_steps = config.accumulation
            self.config = config
            self.epoch = 0

            self.base_dir = f'./{config.folder}'
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)

            self.log_path = f'{self.base_dir}/log.txt'
            self.best_summary_loss = 10**5
            self.best_score = 0

            self.model = model
            self.device = device

            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ] 

            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
            self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
            self.log(f'Fitter prepared. Device is {self.device}')
            self.criterion = nn.BCEWithLogitsLoss()
            self.model.to(device)

        def fit(self, train_loader, validation_loader):
            TRAINING_START = time.time()

            best_epoch = 0
            for e in range(self.config.n_epochs):
                if (time.time() - TRAINING_START)/60 > TRAIN_TIME : 
                    self.log('Time limit exceeded')
                    break


                lr = self.optimizer.param_groups[0]['lr']
                self.log(f'LR: {lr}')

                train_loss, train_auc, _ = self.train_one_epoch(train_loader)


                self.log(f'Train => Epoch: {self.epoch}, summary_loss: {train_loss.avg:.5f}, train_score: {train_auc: .5f}')
                self.save(f'{self.base_dir}/last-checkpoint.bin', f'{self.base_dir}/last-checkpoint.pt')

                val_loss, val_auc = self.validation(validation_loader)

                self.log(f'Valid => Epoch: {self.epoch}, summary_loss: {val_loss.avg:.5f}, val_score: {val_auc: .5f}')

                if val_auc > self.best_score:
                    best_epoch = e
                    self.best_score = val_auc
                    self.best_summary_loss = val_loss.avg
                    self.model.eval()
                    self.log(f'Saving in {self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                    self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin', f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.pt')
                    for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                        os.remove(path)

                if self.config.validation_scheduler:
                    self.scheduler.step(metrics=val_loss.avg)

                self.epoch += 1

            self.log(f'Best score : {self.best_score}, Best_score_loss : {self.best_summary_loss}, Best Epoch : {best_epoch}')

        def validation(self, val_loader):
            self.model.eval()
            summary_loss = AverageMeter()
            accuracy = AverageMeter()
            TARGETS = []
            PREDS = []

            tqloader = tqdm(val_loader, total=len(val_loader))

            for step, (images, targets) in enumerate(tqloader):

                with torch.no_grad():
                    images, targets = images.to(device), targets.to(device)
                    batch_size = images.shape[0]

                    logits = self.model(images)
                    loss = self.criterion(logits,targets)
                    logits = logits.sigmoid()

                    PREDS += [logits.detach().cpu()]
                    TARGETS += [targets.detach().cpu()]
                    summary_loss.update(loss.detach().item(), batch_size)

            PREDS = torch.cat(PREDS).cpu().numpy()
            TARGETS = torch.cat(TARGETS).cpu().numpy()
            roc_auc = macro_multilabel_auc(TARGETS, PREDS)
            return summary_loss, roc_auc


        def train_one_epoch(self, train_loader):
            self.model.train()
            self.model.zero_grad() #accumulation

            summary_loss = AverageMeter()
            tqloader = tqdm(train_loader, total = len(train_loader))

            TARGETS = []
            PREDS = []
            for step, (images, targets) in enumerate(tqloader):

                images, targets = images.to(device), targets.to(device)
                batch_size = images.shape[0]

                # self.optimizer.zero_grad()

                logits = self.model.forward(images) # forward propogation
                loss = self.criterion(logits,targets)
                logits = logits.sigmoid()
                PREDS += [logits.detach().cpu()]
                TARGETS += [targets.detach().cpu()]

                loss = loss / self.accumulation_steps
                loss.backward()

                if (step+1) % self.accumulation_steps == 0: # Wait for several backward steps
                    self.optimizer.step()
                    self.model.zero_grad() 

                if self.config.step_scheduler:
                    self.scheduler.step()

                summary_loss.update(loss.detach().item(), batch_size)

            if self.config.custom_step_scheduler:
                self.scheduler.step()

            self.model.zero_grad()
            PREDS = torch.cat(PREDS).cpu().numpy()
            TARGETS = torch.cat(TARGETS).cpu().numpy()
            roc_auc = macro_multilabel_auc(TARGETS, PREDS)

            PREDS = np.where(PREDS>=0.5, 1, 0)
            f1 = f1_score(TARGETS, PREDS, average='weighted')
            return summary_loss, roc_auc, f1


        def save(self, path, backbone_path):
            self.model.eval()
            # self.log(f'Model saved in {path}')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_summary_loss': self.best_summary_loss,
                'best_score':self.best_score,
                'epoch': self.epoch,
            }, path)
            torch.save(self.model.model, backbone_path)

        def load(self, path):
            checkpoint = torch.load(path)
            print(f'{path} loaded')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_summary_loss = checkpoint['best_summary_loss']
            self.best_score = checkpoint['best_score']
            self.epoch = checkpoint['epoch'] + 1
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        def log(self, message):
            if self.config.verbose:
                print(message)
            with open(self.log_path, 'a+') as logger:
                logger.write(f'{message}\n')


        def fit_all_data(self, train_loader):
            TRAINING_START = time.time()
    #         if self.wconfig: wandb.watch(self.model, log="all", log_freq=10)

            best_epoch = 0
            for e in range(self.config.n_epochs):
                if (time.time() - TRAINING_START)/60 > TRAIN_TIME : 
                    self.log('Time limit exceeded')
                    break


                lr = self.optimizer.param_groups[0]['lr']
                self.log(f'LR: {lr}')

                train_loss, train_auc, f1 = self.train_one_epoch(train_loader)


                self.log(f'Train => Epoch: {self.epoch}, summary_loss: {train_loss.avg:.5f}, train_score: {train_auc: .5f}')
                self.log(f'Saving in {self.base_dir}/checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                self.save(f'{self.base_dir}/checkpoint-{str(self.epoch).zfill(3)}epoch.bin', f'{self.base_dir}/checkpoint-{str(self.epoch).zfill(3)}epoch.pt' )

                if train_auc > self.best_score:
                    best_epoch = self.epoch
                    self.best_score = train_auc
                    self.best_summary_loss = train_loss.avg


                if self.config.validation_scheduler:
                    self.scheduler.step(metrics=val_loss.avg)

                self.epoch += 1

            self.log(f'Best score : {self.best_score}, Best_score_loss : {self.best_summary_loss}, Best Epoch : {best_epoch}')



    model = Model_Classifier(
        n_classes = NUM_CLASSES,
        model_name = MODEL,
        extra_fc = EXTRA_FC,
        pretrained = PRETRAINED,
        replace_silu = REPLACE_SILU
    )

    fitter = Fitter(
        model=model, 
        device=device,
        config=Train_Config
    )

    if from_previous_checkpoint:
        fitter.load(checkpoint_path)

    if FOLD == -1:
        fitter.fit_all_data(train_loader)
    else:
        fitter.fit(train_loader, val_loader)
        
