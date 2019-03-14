This is the code repository for ["Hierarchy Denoising Recursive Autoencoders for 3D Scene Layout Prediction"][1] .

IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2019

Created by Yifei Shi, Angel Xuan Chang, Zhelun Wu, Manolis Savva and Kai Xu

![teaser](image/figure.JPG)

## Usage - VDRAE
### Dependancy
The code depends on Pytorch 3. [pytorch-tools][2] should be installed: 
~~~~ 
git clone https://github.com/nearai/pytorch-tools.git
python setup.py install
~~~~ 

### Pretrained model
You can download the pretrained models [here](https://www.dropbox.com/sh/wnlwvun49z0z3w3/AAAGXm7I-FBxO28dkL-rxrGYa?dl=0) for details.

### Data preparation
You need to process the data before feed it to VDRAE, please see [preprocess](https://github.com/yifeishi/HierarchyLayout/tree/master/preprocess) for details. You can also download the preprocessed data [here](https://www.dropbox.com/sh/xxxxxxxxxxxxxx). 

### Training
Once you have the preprocessed data, you can train a model by run:
~~~~ 
python train.py --data_path DATA_PATH
~~~~ 

Arguments: 
```
'--epochs' (number of epochs; default=5000)
'--batch_size' (batch size; default=1)
'--save_snapshot' (save snapshots of trained model; default=True)
'--save_snapshot_every' (save training log for every X frames; default=100)
'--no_cuda' (use cpu only; default=False)
'--gpu' (device id of GPU to run cuda; default=0)
'--data_path' (path of the pickle files)
'--save_path' (trained model path; default='models')
```

### Testing
To perform the inference, run:
~~~~ 
python test.py --data_path DATA_PATH --pretrained_model PRETRAEINED_MODEL_PATH
~~~~ 

Arguments:
```
'--no_cuda' (use cpu only; default=False)
'--gpu' (device id of GPU to run cuda; default=0)
'--data_path' (path of the pickle files)
'--pretrained_model' (pretrained model path; default='models')
```

### Evaluation
To evaluate the results , run:
~~~~ 
python evaluation.py --ap_category THE_CATEGORY_TO_BE_EVALUATED --IOU IOU
~~~~ 

Arguments:
```
'--ap_category' (the category of evaluation)
'--IOU' (iou to be used for evaluation; default=0.5)
```

## Citation
If you find the code is useful, please cite:
~~~~
@inproceedings{shi2019hierarchy, 
author = {Yifei Shi and Angel Xuan Chang and and Zhelun Wu and Manolis Savva and Kai Xu}, 
booktitle = {Proc. Computer Vision and Pattern Recognition (CVPR), IEEE}, 
title = {Hierarchy Denoising Recursive Autoencoders for 3D Scene Layout Prediction}, 
year = {2019}
}
~~~~

[1]:  https://arxiv.org/pdf/1903.03757.pdf "Hierarchy Denoising Recursive Autoencoders for 3D Scene Layout Prediction"
[2]:  https://github.com/nearai/torchfold "Data and model"

