import os
import sys
import scipy.io as sio
import numpy as np
import random
import pickle

def genGrassData(scene_dir,hier_name,node_info_name):
    obbFile = open(node_info_name)
    length = len(obbFile.readlines())
    obbs = np.zeros((length,8))
    labels = np.zeros(length)
    ious = np.zeros(length)
    offsets = np.zeros((length,8))
    feature_dir = []

    obbFile = open(node_info_name)
    count = 0
    while 1:
        line = obbFile.readline()
        if not line:
            break
        L = line.split()
        if len(L) > 10:
            obbs[count,0] = float(L[0])
            obbs[count,1] = float(L[1])
            obbs[count,2] = float(L[2])
            obbs[count,3] = float(L[3])
            obbs[count,4] = float(L[4])
            obbs[count,5] = float(L[5])
            obbs[count,6] = float(L[6])
            obbs[count,7] = float(L[7])
            
            labels[count] = int(L[16])
            ious[count] = float(L[17])
            if labels[count] == -1 or labels[count] == 99 or ious[count] < 0.3:
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
        else:
            obbs[count,0] = float(L[0])
            obbs[count,1] = float(L[1])
            obbs[count,2] = float(L[2])
            obbs[count,3] = float(L[3])
            obbs[count,4] = float(L[4])
            obbs[count,5] = float(L[5])
            obbs[count,6] = float(L[6])
            obbs[count,7] = float(L[7])
        
        tmp1 = os.path.join(scene_dir,'data','leaf_pts','objects_'+str(count)+'_pointcnn_feature.txt')
        tmp2 = os.path.join(scene_dir,'data','internal_pts','objects_'+str(count)+'_pointcnn_feature.txt')
        if os.path.exists(tmp1):   
            feature_dir.append(tmp1)
        elif os.path.exists(tmp2):
            feature_dir.append(tmp2)
        else:
            feature_dir.append('empty')
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
    return(feature_dir, obbs, offsets, labels, mapFather, mapChild1, mapChild2, isLeaf)


config = util.get_args()
g_path = config.g_path
data_path = config.data_path
g_path = '/home/net663/Downloads/yifeis/S3DIS/data_release'
data_path = '/home/net663/Downloads/yifeis/S3DIS/region_feature'
if os.path.exists(data_path):
    os.mkdir(data_path)

count = 0
scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    hier_name = os.path.join(scene_dir,'data','files','hier_ncut.txt')
    node_info_name = os.path.join(scene_dir,'data','files','node_info.txt')
    
    if not os.path.exists(hier_name) or not os.path.exists(node_info_name):
        continue
    (feature_dir,boxes,boxes_reg,category,mapFather, mapChild1, mapChild2, isLeaf) = genGrassData(scene_dir,hier_name,node_info_name)
    if len(isLeaf) == 0:
        print(scene_name)
    if feature_dir == False:
        continue
    mdict={'scene_dir': scene_dir, 'feature_dir': feature_dir, 'boxes': boxes,'boxes_reg': boxes_reg, 'category':category, 'mapFather':mapFather, 'mapChild1':mapChild1, 'mapChild2':mapChild2, 'isLeaf':isLeaf}
    with open(data_path+ '_' + str(count).pickle', 'w') as f:
        pickle.dump(mdict,f)
    count = count + 1
print(count)