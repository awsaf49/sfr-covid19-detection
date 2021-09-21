First of all, I would like to thank Kaggle, SIIM, FISABIO & RSNA for this amazing competition and also to this amazing community for constantly sharing. Congratulations to all the winners, medalists, and amazing people of this community.

Thanks to all the members of our team @cdeotte @zaber666 @nexh98 @artemenon. This would not be possible without you. Special thanks to @cdeotte for teaming up with us. We've learned a lot from you. No wonder you're one of the best **kagglers**.

## Data Processing:
* **Data-Cleaning**: We removed duplicates manually for both classification and detection. Also for detection, instead of taking one image per patient we only removed unannotated images. For this reason, our **CV** was low comparing most people but we had a good **lb** correlation.
* **Cross-Validation**: StratifiedGroupKFold.

## Study-Level:
* **Pretraining**: We pretrained our models on chexpert. So, all our models use it except(Chris's aux-loss model).
* **Models**: EfficientnetB6 & EfficientnetB7. (Others models didn't do well for us not even `effnetv2`, `resnet200d`, `nfnet`, `vit`).
* **Loss**: CategoricalCrossEntropy.
* **Label-Smoothing**: 0.01
* **Augmentation**: HorizontalFlip, VerticalFlip, RandomRotation, CoarseDropout, RandomShift, RandomZoom, Random(Brightness, Contrast), Cutmix-Mixup.
* **Scheduler**: WarmpupExponentialDecay.
* **Image-Size**: 512, 640, 768.
* **Pseudo-Labeling**: BIMCV + RICORD + RSNA.
* **Knowledge-Distillation(KD)**: We used around 5 of our best models to generate **soft-labels** for it and one of our submissions uses 3 KD models.
* **Aux-Loss**: One of our submissions uses **aux-loss** model by @cdeotte which uses FPN and EfficientnetB4 as backbone. You can check out the post [here][1] for more details. We also used **aux-loss** from @hengck23 's discussion and used it for generating **soft-labels** for KD model. 
* **Post-Processing:** We used **geometric-mean** to re-rank our confidence score(inspired from @cdeotte's solution of VinBigData [here][2]). Our **4cls** models were very dominating that this had very little impact on the score. So finally we decided not to use it even though it did improve the **CV** a little bit.
<img src="https://i.ibb.co/WtPVw7x/eq1.png" alt="eq1" border="0">
    
## 2cls Model:
We used only one **2cls** model. Including more didn't have that much impact.
* **Model**: EfficientNetb7
* **Image-Size**: 640
* **External Data**: We used **RICORD** dataset and their labels posted by @raddar from [here](https://www.kaggle.com/c/siim-covid19-detection/discussion/240187). We simply took **max-voting** to get the labels from different annotators.

## Image-Level:
* **Pretraining**: We pretrained all backbones of detection models.
* **External Data**: RSNA - `opacity` only (6k+).
* **Models**: yolov5x-transformer(thanks to @hengck23 for posting it), yolov5x6, yolov3-spp.
* **Augmentation**: HorizontalFlip, VerticalFlip, Random(Brightness, Contrast), Mosaic-Mixup.
* **Scheduler**: WarmpupCosineDecay.
* **Image-Size**: 512. (Increasing image size worsen our result cuz our backbones were pretrained on **512** image size. We didn't have the time to pretrain on large image-size)
* **Ensemble**: We used **WBF** to merge boxes from different models. (thanks for @cdeotte for making it work).
* **Post-Processing:** We used **geometric-mean** to re-rank our confidence score(inspired from @cdeotte's solution of VinBigData [here][2]). We used both **4cls** and **2cls** models' prediction here.
 <img src="https://i.ibb.co/jR3RXBN/eq2.png" alt="eq2" border="0">
 
* **BBox-Filter**: We filtered out abnormal boxes based on their features. This has little effect on the both **CV** and **LB** still we kept it.

## Final Submissions:
We tried to keep our two final submissions as different as possible. So two of our submissions use different classification models. When we looked back we noticed our best **CV** submission doesn't use any pseudo so we tried to incorporate them in our best **LB** submission. But as we had very little time we went for full data training with **BIMCV** + **RICORD** + **RSNA** pseudo. But we didn't want to take too much risk as we had very few submission left so ensembled full data models with our **LB 0.652** submission. For both submissions, 2cls model & detection models are kept the same.
* Best **CV**:
    * b6 | 512 | KD
    * b6 | 640 | KD
    * b7 | 512 | KD
    * FPN | b4 | 512 | aux_loss
* Best **LB**:
    * (b6, b7) x (512, 640, 768) - 6 models | BIMCV+RICORD+RSNA pseudo | Full Data
    * b6 | 512 | KD
    * b6 | 512 | BIMCV+RICORD pseudo
    * b6 | 512
    
## Result:

* **CV**:

| none  | opacity | study | final |
| ----- | ------- | ----- | ----- |
| 0.823 |  0.562  | 0.607 | 0.635 |


* **LB**:

| public/private | none  | opacity | study | final |
| :--------------: | :-----: |  -----  | ----- | ----- |
| public | 0.810 |  0.594  | 0.628 | 0.654 | 
| private  | 0.822 |  0.594  | 0.592 | 0.631 | 


[1]: https://www.kaggle.com/c/siim-covid19-detection/discussion/263676

[2]: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229637

**Inference Code:** https://www.kaggle.com/awsaf49/siim-covid19-detection-final-infer

## Thank you for reading such a long post :) 
