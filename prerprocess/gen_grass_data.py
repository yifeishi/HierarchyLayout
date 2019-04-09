import os
import sys
import scipy.io as sio
import numpy as np
import random
import pickle
import util

def genGrassData(scene_dir,hier_name,node_info_name,all_box_dir):
    all_box = np.loadtxt(all_box_dir)
    predictions = all_box[:,9:]
    labels_pointcnn = np.zeros(predictions.shape[0])
    for i in range(predictions.shape[0]):
        prediction = predictions[i]
        pred_array = prediction[:]
        label = np.where(pred_array == np.max(pred_array))[0]
        labels_pointcnn[i] = label[0]
        if labels_pointcnn[i] == 99:
            labels_pointcnn[i] = 0

    obbFile = open(node_info_name)
    length = len(obbFile.readlines())
    obbs = np.zeros((length,8))
    ious = np.zeros(length)
    labels = np.zeros(length)
    offsets = np.zeros((length,8))
    valids = np.ones(length)
    feature_dir = []
    if predictions.shape[0] != length:
        print('error on PointCNN feature extraction',scene_dir)
    counts = np.zeros(5)
    obbFile = open(node_info_name)
    count = 0
    while 1:
        line = obbFile.readline()
        if not line:
            break
        L = line.split()
        obbs[count,0] = float(L[0])
        obbs[count,1] = float(L[1])
        obbs[count,2] = float(L[2])
        obbs[count,3] = float(L[3])
        obbs[count,4] = float(L[4])
        obbs[count,5] = float(L[5])
        obbs[count,6] = float(L[6])
        obbs[count,7] = float(L[7])
        
        if len(L) > 10:
            labels[count] = int(L[16])
            ious[count] = float(L[17])
            if ious[count] < 0.5:
                labels[count] = 0
            else:
                offsets[count,0] = float(L[8])
                offsets[count,1] = float(L[9])
                offsets[count,2] = float(L[10])
                offsets[count,3] = float(L[11])
                offsets[count,4] = float(L[12])
                offsets[count,5] = float(L[13])
                offsets[count,6] = float(L[14])
                offsets[count,7] = float(L[15])
        
        tmp1 = os.path.join(scene_dir,'data','merge_leaf_pts','objects_'+str(count)+'_pointcnn_feature.txt')
        tmp2 = os.path.join(scene_dir,'data','internal_pts','objects_'+str(count)+'_pointcnn_feature.txt')
        if os.path.exists(tmp1) and valids[count] == True:
            feature_dir.append(tmp1)
            valids[count] = True
            counts[0] += 1
        elif os.path.exists(tmp2) and valids[count] == True:  
            feature_dir.append(tmp2)
            valids[count] = True
            counts[0] += 1
        else:
            feature_dir.append('empty')
            labels[count] = 0
            counts[1] += 1
        count = count + 1
    
    hier = open(hier_name)
    mapFather = {}
    mapChild1 = {}
    mapChild2 = {}
    isLeaf = {}

    while 1:
        line = hier.readline()
        if not line:
            break
        L = line.split()
        father = int(L[0])
        child = int(L[1])
        mapFather[child] = father
        isLeaf[father] = 'False'
        if mapChild1.get(father, 'False') == 'False':
            mapChild1[father] = child
        else:
            mapChild2[father] = child
        if L[2] == "null":
            isLeaf[child] = 'False'
        else:
            isLeaf[child] = 'True'
    return(feature_dir, obbs, offsets, labels, labels_pointcnn, mapFather, mapChild1, mapChild2, isLeaf)


config = util.get_args()
g_path = config.g_path
scene_list_path = config.scene_list_path
pickle_path = config.pickle_path
if not os.path.exists(pickle_path):
    os.mkdir(pickle_path)

f = open(scene_list_path)
lines = f.readlines()
scene_list = []
for line in lines:
    scene_list.append(line.split()[0])

count = 0
scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    if scene_name.split('.')[1] not in scene_list:
        continue
    scene_dir = os.path.join(g_path,scene_name)
    hier_name = os.path.join(scene_dir,'data','files','hier_ncut.txt')
    node_info_name = os.path.join(scene_dir,'data','files','node_info.txt')
    all_box_dir = os.path.join(scene_dir,'data','files','pred_box_all_node.txt')

    if not os.path.exists(hier_name):
        print('hierarchy not exist',hier_name)
        continue
    (feature_dir,boxes,boxes_reg,category,category_pred,mapFather, mapChild1, mapChild2, isLeaf) = genGrassData(scene_dir,hier_name,node_info_name,all_box_dir)
    mdict={'scene_dir': scene_dir, 'feature_dir': feature_dir, 'boxes': boxes,'boxes_reg': boxes_reg, 'category_pred':category_pred, 'category':category, 'mapFather':mapFather, 'mapChild1':mapChild1, 'mapChild2':mapChild2, 'isLeaf':isLeaf}
    with open(os.path.join(pickle_path, 'test' + '_' + str(count) + '.pkl'), 'wb+') as f:
        pickle.dump(mdict,f)
    print(count)
    count = count + 1
