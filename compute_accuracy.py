import torch
from torch import nn
from torch.autograd import Variable
import util
import scipy.io as sio
from grassdata import GRASSDataset
from grassdata import GRASSDatasetTest
from grassmodel import GRASSEncoderDecoder
from grassdata import GRASSDataset
import grassmodel
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pickle
from sklearn.metrics import average_precision_score

config = util.get_args()
config.cuda = not config.no_cuda

if config.gpu < 0 and config.cuda:
    config.gpu = 0


# xMax,yMax,zMax,xMin,yMin,zMin to size,cen
def ObbFeatureTransformer(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[0] - obj_obb_fea_tmp[3]
    obj_obb_fea[1] = obj_obb_fea_tmp[1] - obj_obb_fea_tmp[4]
    obj_obb_fea[2] = obj_obb_fea_tmp[2] - obj_obb_fea_tmp[5]
    obj_obb_fea[3] = (obj_obb_fea_tmp[0] + obj_obb_fea_tmp[3])*0.5
    obj_obb_fea[4] = (obj_obb_fea_tmp[1] + obj_obb_fea_tmp[4])*0.5
    obj_obb_fea[5] = (obj_obb_fea_tmp[2] + obj_obb_fea_tmp[5])*0.5
    return obj_obb_fea

# size,cen to xMax,yMax,zMax,xMin,yMin,zMin
def ObbFeatureTransformerReverse(obj_obb_fea_tmp):
    obj_obb_fea = np.zeros(6)
    obj_obb_fea[0] = obj_obb_fea_tmp[3] + obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[1] = obj_obb_fea_tmp[4] + obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[2] = obj_obb_fea_tmp[5] + obj_obb_fea_tmp[2]*0.5
    obj_obb_fea[3] = obj_obb_fea_tmp[3] - obj_obb_fea_tmp[0]*0.5
    obj_obb_fea[4] = obj_obb_fea_tmp[4] - obj_obb_fea_tmp[1]*0.5
    obj_obb_fea[5] = obj_obb_fea_tmp[5] - obj_obb_fea_tmp[2]*0.5
    return obj_obb_fea

def computeIOU(boxA,boxB): # size,center
    boxA = ObbFeatureTransformerReverse(boxA)
    boxB = ObbFeatureTransformerReverse(boxB)
    xA = max(boxA[3], boxB[3])
    yA = max(boxA[4], boxB[4])
    zA = max(boxA[5], boxB[5])
    xB = min(boxA[0], boxB[0])
    yB = min(boxA[1], boxB[1])
    zB = min(boxA[2], boxB[2])

    # compute the area of intersection rectangle
    if xA - xB > 0 or yA - yB > 0 or zA - zB > 0:
        interArea = 0
        return 0
    else:
        interArea = (xB - xA) * (yB - yA) * (zB - zA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[0] - boxA[3]) * (boxA[1] - boxA[4]) * (boxA[2] - boxA[5])
    boxBArea = (boxB[0] - boxB[3]) * (boxB[1] - boxB[4]) * (boxB[2] - boxB[5])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou


def GetLeafNodeIDs(fatherNode, IDs, mapChild1, mapChild2, mapFather, isLeaf):
    childNode1 = mapChild1[fatherNode]
    childNode2 = mapChild2[fatherNode]
    if isLeaf[childNode1] == 'True':
        IDs.append(childNode1)
    else:
        GetLeafNodeIDs(childNode1, IDs, mapChild1, mapChild2, mapFather, isLeaf)
    
    if isLeaf[childNode2] == 'True':
        IDs.append(childNode2)
    else:
        GetLeafNodeIDs(childNode2, IDs, mapChild1, mapChild2, mapFather, isLeaf)
    return IDs

def computeAccIou(pred_seg_label_path,gt_seg_label_path,data_path,countGT,countPred,countRight):
    fg = open(gt_seg_label_path,'r')
    lines = fg.readlines()
    segNum = len(lines)
    segGtLabel = np.zeros(segNum)
    segPredLabel = np.zeros(segNum)

    fg = open(gt_seg_label_path,'r')
    while 1:
        line = fg.readline()
        if not line:
            break
        L = line.split()
        segID = int(L[1])
        label = 0
        if L[4].split('_')[0] == 'clutter':
            label = 0
        elif L[4].split('_')[0] == 'wall':
            label = 2
        elif L[4].split('_')[0] == 'door':
            label = 3
        elif L[4].split('_')[0] == 'ceiling':
            label = 4
        elif L[4].split('_')[0] == 'floor':
            label = 5
        elif L[4].split('_')[0] == 'chair':
            label = 6
        elif L[4].split('_')[0] == 'bookcase':
            label = 7
        elif L[4].split('_')[0] == 'board':
            label = 8
        elif L[4].split('_')[0] == 'table':
            label = 9
        elif L[4].split('_')[0] == 'beam':
            label = 10
        elif L[4].split('_')[0] == 'column':
            label = 11
        elif L[4].split('_')[0] == 'Icon':
            label = 12
        elif L[4].split('_')[0] == 'window':
            label = 13
        elif L[4].split('_')[0] == 'sofa':
            label = 14
        elif L[4].split('_')[0] == 'stairs':
            label = 15
        else:
            label = 0
        segGtLabel[segID] = label
    
    fp = open(pred_seg_label_path,'r')
    while 1:
        line = fp.readline()
        if not line:
            break
        L = line.split()
        segID = int(L[1])
        label = int(L[3])
        segPredLabel[segID] = label
    
    for j in range(15):#category
        countGtTemp = 0
        countPredTemp = 0
        countRightTemp = 0
        for i in range(segNum):
            pts_path = os.path.join(data_path, 'objects_'+str(i)+'.pts')
            fpts = open(pts_path,'r')
            lines = fpts.readlines()
            ptsNum = len(lines)

            if segGtLabel[i] == j:
                countGtTemp += ptsNum
            if segPredLabel[i] == j:
                countPredTemp += ptsNum
            if segGtLabel[i] == segPredLabel[i] and segGtLabel[i] == j:
                countRightTemp += ptsNum
        countGT[j] += countGtTemp
        countPred[j] += countPredTemp
        countRight[j] += countRightTemp

        if countGT[j]+countPred[j]-countRight[j] > 0:
            iou = float(countRight[j])/(countGT[j]+countPred[j]-countRight[j])
            
            print('j',j,'count',countRight[j],countGT[j],countPred[j],'iou',iou)
    return countGT,countPred,countRight

print("Loading data.")
grass_data = GRASSDatasetTest(config.data_path)
def my_collate(batch):
    return batch
test_iter = torch.utils.data.DataLoader(grass_data, batch_size=1, shuffle=False, collate_fn=my_collate)

g_box_path = '/home/net663/Downloads/yifeis/S3DIS'
g_bvh_path = '/home/net663/Downloads/yifeis/S3DIS/stanford_cluster/Stanford3dDataset_v1.2_Aligned_Version'


countGT = np.zeros(15)
countPred = np.zeros(15)
countRight = np.zeros(15)
for batch_idx, batch in enumerate(test_iter):
    L = batch[0].region_path.split('/')
    pred_seg_label_path = os.path.join(g_box_path, L[8], L[9], 'pred_segment_semantic_label.txt')
    gt_seg_label_path = os.path.join(g_bvh_path, L[8], L[9], 'data', 'segment_instance_label.txt')
    data_path = os.path.join(g_bvh_path, L[8], L[9], 'data')

    if not os.path.exists(pred_seg_label_path) or not os.path.exists(gt_seg_label_path):
        continue
    
    countGT,countPred,countRight = computeAccIou(pred_seg_label_path,gt_seg_label_path,data_path,countGT,countPred,countRight)
    