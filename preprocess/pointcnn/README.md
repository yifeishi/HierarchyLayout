# PointCNN for feature extraction

These are the code to preprocess the data.

You should install matlab.engine and python 2.

The file format of our input are: 
scene.txt...overseg.txt...gt_info.txt
An example can be downloaded at...

Once you have transform your own data to the format, run:
~~~~ 
python bat_preprocess.py
~~~~ 
The code will compute pair-wise affinities, build segment hierarchies and output the *.pkl files which could be processed by VDRAE.

Please note that util_obb.py should be run by python 2.


This is the code repository for ["Hierarchy Denoising Recursive Autoencoders for 3D Scene Layout Prediction"][1] .

IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2019

Created by Yifei Shi, Angel Xuan Chang, Zhelun Wu, Manolis Savva and Kai Xu

![teaser](image/figure.PNG)

## Usage - VDRAE
### Dependancy
The code depends on Pytorch 3. ["pytorch-tools"][2] should be installed: 
~~~~ 
git clone https://github.com/nearai/pytorch-tools.git
python setup.py install
~~~~ 

### Data preparation
For S3DIS, ScanNet and Matterport3D, you can download the preprocessed data here. 
For your own scene, please see ./preprocess for details.

### Training
To train a model, run:
~~~~ 
python train.py --data_path DATA_PATH
~~~~ 

Arguments: 
```
'--epochs' (number of epochs; default=100)
'--batch_size' (batch size; default=16)
'--num_workers' (number of workers; default=8)
'--save_snapshot' (save snapshots of trained model)
'--save_snapshot_every' (save training log for every X frames; default=100)
'--lr' (initial learning rate; default=.001)
'--gpu' (device id of GPU to run cuda; default=0)
(TBA)
```

### Testing
To perform the inference, run:
~~~~ 
python test.py --data_path DATA_PATH --pretrained_model PRETRAEINED_MODEL_PATH
~~~~ 

Arguments:
```
(TBA)
```

### Evaluation
To evaluate the results , run:
~~~~ 
python evaluation.py --ap_category THE_CATEGORY_TO_BE_EVALUATED --IOU IOU_TO_BE_USED
~~~~ 

Arguments:
```
(TBA)
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

