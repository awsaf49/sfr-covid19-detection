<h1 align="center">Best Student Team & 4th Place Solution of SIIM-FISABIO-RSNA COVID-19 Detection</h1>
<p align="center">Identify and localize COVID-19 abnormalities on chest radiographs</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/36858976/119347450-5c669680-bcbd-11eb-97bd-f6d666b59cda.png">
</p>

<p align="center">
This is a collaboration between <a href="https://www.buet.ac.bd/web/">BUET</a> <img src="https://i.ibb.co/LkL8TQ8/1200px-BUET-LOGO-svg.png" width="50"> and <a href="https://www.nvidia.com/en-us/">NVIDIA</a> <img src="https://i.ibb.co/y0JSH0H/Capture.jpg" height="40">
</p>

<h2 align="center">Team Members</h2>

<div align="center">
<table>
  <tr>
    <th>Name</th>
    <th>Affiliation</th>
    <th>Country</th>
    <th>Position</th>
  </tr>
  <tr>
    <td>Md Awsafur Rahman</td>
    <td>Dept. of EEE, BUET</td>
    <td>🇧🇩</td>
    <td>Undergrad Student</td>
  </tr>
  <tr>
    <td>Bishmoy Paul</td>
    <td>Dept. of EEE, BUET</td>
    <td>🇧🇩</td>
    <td>Undergrad Student</td>
  </tr>
  <tr>
    <td>Najibul Haque Sarker</td>
    <td>Dept. of CSE, BUET</td>
    <td>🇧🇩</td>
    <td>Undergrad Student</td>
  </tr>
  <tr>
    <td>Zaber Ibn Abdul Hakim</td>
    <td>Dept. of CSE, BUET</td>
    <td>🇧🇩</td>
    <td>Undergrad Student</td>
  </tr>
  <tr>
    <td>Chris Deotte</td>
    <td>Nvidia</td>
    <td>🇺🇸</td>
    <td>Senior Data Scientist</td>
  </tr>
</table>
</div>


<h2 align="center">Solution Reproduction</h2>
<p align="center">Below you can find an outline of how to reproduce our solution.</p>
<p align="center">If you run into any trouble with the setup/code or have any questions please contact me at <a href="mailto:awsaf49@gmail.com">awsaf49@gmail.com</a></p>


## 0. Video Summary on YouTube
[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/awsaf49/sfr-covid19-detection)
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/Rqal21kT-zA)

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Rqal21kT-zA" target="_blank">
 <img src="https://i.ibb.co/LxdzdDs/chris.jpg" alt="Watch the video" width="600" border="10" />
</a>

## 1. Requirements:

### 1.1 Hardware:
* GPU : 4x Tesla V100
* GPU Memory : 4x32 GiB
* CUDA Version : 11.0
* Driver Version : 450.119.04
* CPU RAM : 16 GiB
* DISK : 2 TB

### 1.2 Libraries:

- python-gdcm==3.0.9.1
- pydicom==2.1.2
- joblib==1.0.1
- tensorflow==2.4.1
- torch==1.7.0
- torchvision==0.8.1
- numpy==1.19.5
- pandas==1.2.4
- matplotlib==3.4.2
- opencv-python==4.5.2.54
- opencv-python-headless==4.5.2.54
- Pillow==8.2.0
- PyYAML>=5.3.1
- scipy==1.6.3
- tqdm==4.61.1
- tensorboard==2.4.1
- seaborn==0.11.1
- ensemble_boxes==1.0.6
- albumentations==1.0.1
- thop==0.0.31.post2005241907
- Cython==0.29.23  
- pycocotools==2.0
- addict==2.4.0
- timm==0.4.12
- efficientnet==1.1.1

## 2. External Packages
External Packages with version number are listed on [requirements.txt](https://github.com/awsaf49/sfr-covid19-detection/blob/main/requirements.txt)

```
! pip install -qr requirements.txt
```

## 3. Data Preparation
### 3.1 Description
- Download [competition data](https://www.kaggle.com/c/siim-covid19-detection/data) and extract it to `./data/siim-covid19-detection`
- Download [chexpert dataset](https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0.zip&h=bb5d97db389ae3d2a319d4f78d1f3205c97a22ddb7f4ed59d0dc08db0a4383a2&v=1&xid=da3b9def78&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed) and extract to `./data/chexpert`
- Download [RSNA competion data](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) and extract it to `./data/rsna-pneumonia-detection-challenge`
- Download [Ricord dataset](https://www.kaggle.com/raddar/ricord-covid19-xray-positive-tests) and extract it to `./data/ricord`

After this `./data` directory should look something like this.
```
.
├── data
│   ├── chexpert
│   │   ├── train
│   │   ├── train.csv
│   │   ├── valid
│   │   └── valid.csv
│   ├── ricord
│   │   ├── MIDRC-RICORD
│   │   └── MIDRC-RICORD-meta.csv
│   ├── rsna-pneumonia-detection-challenge
│   │   ├── GCP Credits Request Link - RSNA.txt
│   │   ├── stage_2_detailed_class_info.csv
│   │   ├── stage_2_sample_submission.csv
│   │   ├── stage_2_test_images
│   │   ├── stage_2_train_images
│   │   └── stage_2_train_labels.csv
│   └── siim-covid19-detection
│       ├── sample_submission.csv
│       ├── test
│       ├── train
│       ├── train_image_level.csv
│       └── train_study_level.csv
```

In case you are wondering to have a look at complete directory structure, you can see it in [data_structure.txt](https://github.com/awsaf49/sfr-covid19-detection/blob/main/data_structure.txt)

After this run `prepare_data.py`. It does the following
- Read training data from RAW_DATA_DIR (specified in SETTINGS.json)
- Run any preprocessing steps
- Save the cleaned data to CLEAN_DATA_DIR (specified in SETTINGS.json)
### 3.2 Script
### prepare_data.py
- **--img-size** image size in which we want our cleaned to to be
- **--debug** if given 1, it will only process 100 images
```
! python prepare_data.py 
```


## 4. Training
### 4.1 Description
Simply run the `train.py` script. It does the following
- Read training data from TRAIN_DATA_CLEAN_PATH (specified in SETTINGS.json)
- Pretrains classification and detection backbones in chexpert data.
- Finetunes them on competition data and external data.
- Save model to MODEL_DIR (specified in SETTINGS.json)

### 4.2 Script
### train. py
- **--settings-path** path to SETTINGS.json. Default value uses the correct path.
- **--clsbs-path** path to json file containing necessary batch sizes for different classification models. Default value uses the correct path.
- **--detbs-path** path to json file containing necessary batch sizes for different detection models. Default value uses the correct path.
- **--debug** will process only 100 images
```
! python train.py
```

## 5. Prediction
### 5.1 Description
Before proceeding download this [already trained checkpoints](https://www.kaggle.com/dataset/d5bb61630644beae6821f07e4be4b9cfb643550521f9ddec244b6c41c742a053) and unzip them into the path specified in **CHECKPOINT_DIR** in SETTINGS.json.

`./checkpoints` then should look like
```
.
├── checkpoints
│   ├── 2cls
│   ├── 4cls
│   ├── det
```

For predicting on test data run `predict.py`. It does the following
- Read test data from TEST_DATA_CLEAN_PATH (specified in SETTINGS.json)
- Loads models from MODEL_DIR(specified in SETTINGS.json) when everything is trained from scratch or CHECKPOINT_DIR(specified in SETTINGS.json) when predicting from our previously trained checkpoints.
- Use our models to make predictions on new samples
- Save our predictions to SUBMISSION_DIR (specified in SETTINGS.json)

### 5.2 Script
### predict. py 
- **--mode** if used "full", then it will use the weights saved in [MODEL_DIR](https://github.com/awsaf49/sfr-covid19-detection/tree/main/models) (which was saved after training from scratch) and when used "fast" it will use the weights saved in [CHECKPOINT_DIR](https://github.com/awsaf49/sfr-covid19-detection/tree/main/checkpoints) (already trained checkpoints)
- **--debug** if given 1, it will infer on only first 100 images

```
!python predict.py --mode "fast"
```
or
```
! python predict.py --mode "full"
``` 



<!-- 
## External Dataset:
* chexpert : [CheXpert-v1.0 Original (~439G)](https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0.zip&h=bb5d97db389ae3d2a319d4f78d1f3205c97a22ddb7f4ed59d0dc08db0a4383a2&v=1&xid=da3b9def78&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed)
* rsna     : [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
* ricord   : [RICORD COVID-19 X-ray positive tests](https://www.kaggle.com/raddar/ricord-covid19-xray-positive-tests) -->

## Acknowledgement ❤️
* [Weights & Biases](https://wandb.ai/) for tracking **training**.
* [efficientnet](https://github.com/qubvel/efficientnet) for **efficientnet** model.
