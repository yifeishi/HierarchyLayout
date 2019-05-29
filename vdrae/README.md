# VDRAE
These are the code for VDRAE.

### Dependancy
The code depends on Pytorch 3. [pytorch-tools][2] should be installed.

### Pretrained model
You can download the pretrained models for S3DIS and Matterport3D [here](https://www.dropbox.com/s/53fyv743lcduul0/pretrained_model.zip?dl=0).

### Data preparation
You need to process the raw data before feed it to VDRAE, please see [preprocess](https://github.com/yifeishi/HierarchyLayout/tree/master/preprocess) for details. You can also download the preprocessed data [here](https://www.dropbox.com/s/kdty4kn7tlwsusv/processed_data.zip?dl=0).

### Training
Once you have the preprocessed data, you can train a model by run:
~~~~ 
python3 train.py --data_path PROCESSED_DATA_PATH
~~~~ 

Arguments: 
```
'--epochs' (number of epochs; default=5000)
'--batch_size' (batch size; default=1)
'--save_snapshot' (save snapshots of trained model; default=True)
'--save_snapshot_every' (save training log for every X frames; default=100)
'--no_cuda' (use cpu only; default=False)
'--gpu' (device id of GPU to run cuda; default=0)
'--data_path' (path of the preprocessed data)
'--save_path' (trained model path; default='models')
```

### Testing
To perform the inference, run:
~~~~ 
python3 test.py --data_path PROCESSED_DATA_PATH --g_path DATASET_PATH --pretrained_model PRETRAEINED_MODEL_PATH
~~~~ 

Arguments:
```
'--no_cuda' (use cpu only; default=False)
'--gpu' (device id of GPU to run cuda; default=0)
'--data_path' (path of the preprocessed data)
'--g_path' (path of the dataset)
'--pretrained_model' (pretrained model path; default='models')
```

### Evaluation
To evaluate the results , run:
~~~~ 
python evaluation.py --g_path DATASET_PATH --g_gt_path DATASET_GROUNDTRUTH_PATH --ap_category THE_CATEGORY_TO_BE_EVALUATED --IOU IOU
~~~~ 

Arguments:
```
'--g_path' (path of the dataset)
'--g_gt_path' (path of the groud-truth data)
'--ap_category' (the category of evaluation)
'--IOU' (iou to be used for evaluation; default=0.5)
```

### Testing and evaluation script
An example about how to run the code of testing and evaluation can be found in bat_test_eval.py

[1]:  https://arxiv.org/pdf/1903.03757.pdf "Hierarchy Denoising Recursive Autoencoders for 3D Scene Layout Prediction"
[2]:  https://github.com/nearai/torchfold "torchfold"
