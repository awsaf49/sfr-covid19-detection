# Classification Training Scripts

## train_4cls.py
* **--TRAIN_DATA_CLEAN_PATH** takes the path where the train images are located. The script expects the images in 'TRAIN_DATA_CLEAN_PATH/train/*.png' format

* **--CHECKPOINT_DIR** if there is a pretrained checkpoint, add its location here. The default checkpoint is Imagenet weights

* **--MODEL_DIR** where to save our trained model. Input can be a directory like 'weight1' or weight path like 'weight1/model.h5'.

* **--cfg** location of config json file
* **--dim** image size of training
* **--model_name** only supports efficientnet models. Sample Input format 'efficientnet_b0' for efficientnetb0
* **--epochs** how many epochs to train
* **--save_epoch** which epoch model to save, Default is -1, which saves the last epoch 
* **--bs** batch size

## train_chex.py
* train_chex takes in all the arguments of train_4cls, with some minor changes

* **--TRAIN_DATA_CLEAN_PATH** takes the path where the images are located. The script expects the images in 'TRAIN_DATA_CLEAN_PATH/train/*.png' and 'TRAIN_DATA_CLEAN_PATH/valid/*.png' format

## train_2cls_fold.py
* train_2cls takes in all the arguments of train_4cls, with some additional arguments and changes

* **--MODEL_DIR** where to save our trained model. Input is a directory like 'weight1' where the 5 fold models would be saved.

* **--RICORD_PATH** takes the path where the ricord images are located. The script expects the images in 'RICORD_PATH/images/*.png' format

