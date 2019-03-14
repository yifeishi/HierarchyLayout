# PREPROCESS

These are the code to preprocess the data.
### Dependancy
You should install [MATLAB Engine API][1], [Open3D][2], [scikit-learn][3], [tensorflow][4].
### Input file format
**scene.txt**： Point cloud of the input scene. Each line provides the information of one point with:

    x y z r g b
    
    
**overseg.txt**： The over-segmentation of the input scene. Each line provides the over-segmentation label of one point:

    overseg_label
    
    
**gt_info.txt**： The information of ground-truth object. Each line provides the following information of a object:

    aabb_len_x aabb_len_x aabb_len_z aabb_cen_x aabb_cen_y aabb_cen_z obb_len_x obb_len_x obb_len_z obb_cen_x obb_cen_y obb_cen_z obb_rot_x obb_rot_y sem_label ins_label
    
### Process
Once you have transform your own data to the format, run:
~~~~ 
python bat_preprocess.py --g_path DATA_PATH --pkl_path PROCESSED_FILE_PATH --model_path PRETRAINED_POINTCNN_MODEL_PATH
~~~~ 
The code will compute pair-wise affinities, build segment hierarchies, extract PointCNN feature and output the *.pkl files which could be processed by VDRAE.

### Intermediate file format
（TBA）

### Details of pipeline
（TBA）


[1]:  https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html "MATLAB Engine API"
[2]:  https://github.com/IntelVCL/Open3D "Open3D"
[3]:  https://github.com/scikit-learn/scikit-learn "scikit-learn"
[4]:  https://github.com/tensorflow/tensorflow "tensorflow"
[5]:  https://github.com/ScanNet/ScanNet/tree/master/Segmentator "Over-segmentation"