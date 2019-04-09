import os
import sys
import numpy as np
import random
import util
import shutil
from qhull_2d import *
from min_bounding_rect import *
import pickle
from sklearn.neighbors import KDTree
import time
from util_obb import *

def GetLeafNodeIDs(fatherNode, IDs):
    childNode1 = mapChild1[fatherNode]
    childNode2 = mapChild2[fatherNode]
    if isLeaf[childNode1] == 'True':
        IDs.append(childNode1)
    else:
        GetLeafNodeIDs(childNode1, IDs)
    
    if isLeaf[childNode2] == 'True':
        IDs.append(childNode2)
    else:
        GetLeafNodeIDs(childNode2, IDs)
    return IDs

def computePtsIou(pts0, pts1, obbA, obbB):
    iou = 0
    aabbA = obb2Aabb(obbA)
    aabbB = obb2Aabb(obbB)
    iou_aabb = computeAabbIou(aabbA,aabbB)
    if iou_aabb < 0.1:
        iou = 0
        return iou
    tree0 = KDTree(pts0[:,:3], leaf_size=2)
    tree1 = KDTree(pts1[:,:3], leaf_size=2)
    count_in0 = 0
    count_in1 = 0
    count_all0 = 0
    count_all1 = 0
    for i in range(pts0.shape[0]):
        if random.random() > 0.1:
            continue
        dist,ind = tree1.query(pts0[i:i+1,:3], k=1)
        if dist[0] < 0.1:
            count_in0 += 1
        count_all0 += 1
    for i in range(pts1.shape[0]):
        if random.random() > 0.1:
            continue
        dist,ind = tree0.query(pts1[i:i+1,:3], k=1)
        if dist[0] < 0.1:
            count_in1 += 1
        count_all1 += 1
    intersection = (count_in0 + count_in1) * 0.5
    union = count_all0 + count_all1 - intersection
    if union == 0:
        iou = 0
    else:
        iou = float(intersection)/union
    return iou

config = util.get_args()
g_path = config.g_path
g_gt_path = config.g_gt_path

scene_dir = os.path.join(g_path,'scene_normal.txt')
overseg_dir = os.path.join(g_path,'overseg.txt')
gt_info_dir = os.path.join(g_path,'gt_info.txt')
data_dir = os.path.join(g_path,'data')
leaf_dir = os.path.join(data_dir,'leaf_pts')
merge_leaf_dir = os.path.join(data_dir,'merge_leaf_pts')
internal_dir = os.path.join(data_dir,'internal_pts')
files_dir = os.path.join(data_dir,'files')
if os.path.exists(internal_dir):
    shutil.rmtree(internal_dir)
os.mkdir(internal_dir)

##########################################################################
## generate internal pts
hier_name = os.path.join(files_dir,'hier_ncut.txt')
hier = open(hier_name)
mapFather = {}
mapChild1 = {}
mapChild2 = {}
isLeaf = {}
maxNodeID = -1

while 1:
    line = hier.readline()
    if not line:
        break
    L = line.split()
    fatherNode = int(L[0])
    isLeaf[fatherNode] = 'False'
    childNode = int(L[1])
    if maxNodeID < fatherNode:
        maxNodeID = fatherNode
    mapFather[childNode] = fatherNode
    if mapChild1.get(fatherNode, 'False') == 'False':
        mapChild1[fatherNode] = childNode
    else:
        mapChild2[fatherNode] = childNode
    if L[2] == "null":
        isLeaf[childNode] = 'False'
    else:
        isLeaf[childNode] = 'True'

for i in range(0, maxNodeID+1):
    if isLeaf[i] == 'True':
        continue
    IDs = []
    IDs = GetLeafNodeIDs(i, IDs)
    IDs.sort()
    internalPtsName = os.path.join(internal_dir,'objects_'+str(i)+'.pts')
    fi = open(internalPtsName,'a')
    for id in IDs:
        ptsName = os.path.join(merge_leaf_dir,'objects_'+str(id)+'.pts')
        fpts = open(ptsName)
        while 1:
            line = fpts.readline()
            fi.write(line)
            if not line:
                break
        fpts.close()
    fi.close()

time1 = time.time()

##########################################################################
## compute node info
id_pts_dict = {}
pts_names = os.listdir(merge_leaf_dir)
pts_names.sort()
for pts_name in pts_names:
    if len(pts_name.split('.'))>1 and pts_name.split('.')[1]=='pts':
        id = int(pts_name.split('.')[0].split('_')[1])
        pts_file = os.path.join(merge_leaf_dir,pts_name)
        id_pts_dict[id] = pts_file
internal_pts_names = os.listdir(internal_dir)
internal_pts_names.sort()
for internal_pts_name in internal_pts_names:
    if len(internal_pts_name.split('.'))>1 and internal_pts_name.split('.')[1]=='pts':
        id = int(internal_pts_name.split('.')[0].split('_')[1])
        pts_file= os.path.join(internal_dir,internal_pts_name)
        id_pts_dict[id] = pts_file

node_aabb = np.zeros((len(id_pts_dict),6))
node_obb = np.zeros((len(id_pts_dict),8))

for i in range(0,node_obb.shape[0]):
    pts_name = id_pts_dict[i]
    pts = getPTS(pts_name)
    if pts.shape[0] > 10:
        node_aabb[i] = PTS2AABB(pts)
        node_obb[i] = PTS2OBB(pts)
    else:
        node_aabb[i] = np.zeros(6)
        node_obb[i] = np.zeros(8)
    
gtInfoFile = open(os.path.join(g_path,'gt_info.txt'),'r')
lines = gtInfoFile.readlines()
count = len(lines)
gt_aabb = np.zeros((count,6))
gt_obb = np.zeros((count,8))
gt_label = np.zeros(count)
gtInfoFile = open(os.path.join(g_path,'gt_info.txt'),'r')
count = 0
while 1:
    line = gtInfoFile.readline()
    if not line:
        break
    L = line.split()
    gt_aabb[count,0] = float(L[0])
    gt_aabb[count,1] = float(L[1])
    gt_aabb[count,2] = float(L[2])
    gt_aabb[count,3] = float(L[3])
    gt_aabb[count,4] = float(L[4])
    gt_aabb[count,5] = float(L[5])
    gt_obb[count,0] = float(L[6])
    gt_obb[count,1] = float(L[7])
    gt_obb[count,2] = float(L[8])
    gt_obb[count,3] = float(L[9])
    gt_obb[count,4] = float(L[10])
    gt_obb[count,5] = float(L[11])
    gt_obb[count,6] = float(L[12])
    gt_obb[count,7] = float(L[13])
    gt_label[count] = float(L[14])
    count=count+1


##########################################################################
## compute node info
node_offset = np.zeros((node_obb.shape[0],8))
node_label = np.zeros(node_obb.shape[0])
node_iou = np.zeros(node_obb.shape[0])
iou_matrix = np.zeros((gt_obb.shape[0],node_obb.shape[0]))

insidePointsGT = []
for i in range(0,gt_obb.shape[0]):
    insidePointsGT.append(samplePoints(gt_obb[i,:]))

for i in range(0,node_obb.shape[0]):
    insidePoints = samplePoints(node_obb[i,:])
    area_node = node_obb[i,0] * node_obb[i,1] * node_obb[i,2]
    for j in range(0,gt_obb.shape[0]): 
        iou = computeObbIou(node_obb[i,:],gt_obb[j,:],insidePoints,insidePointsGT[j])
        iou_matrix[j,i] = iou
        if iou > node_iou[i]:
            node_label[i] = gt_label[j]
            node_offset[i] = gt_obb[j,:] - node_obb[i,:]
            node_iou[i] = iou
        
output = open(os.path.join(files_dir,'node_info.txt'),'w')
for i in range(0,node_obb.shape[0]):
    if node_iou[i] > 0:
        output.write('%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f\n'\
        %(node_obb[i,0],node_obb[i,1],node_obb[i,2],node_obb[i,3],\
        node_obb[i,4],node_obb[i,5],node_obb[i,6],node_obb[i,7],
        node_offset[i,0],node_offset[i,1],node_offset[i,2],node_offset[i,3],\
        node_offset[i,4],node_offset[i,5],node_offset[i,6],node_offset[i,7],\
        node_label[i],node_iou[i]))
    else:
        output.write('%f %f %f %f %f %f %f %f\n'\
        %(node_obb[i,0],node_obb[i,1],node_obb[i,2],node_obb[i,3],\
        node_obb[i,4],node_obb[i,5],node_obb[i,6],node_obb[i,7]))
output.close()

pickle_dir = os.path.join(g_path,'data','files','iou_matrix.pkl')
mdict={'iou_matrix': iou_matrix}
with open(pickle_dir, 'wb+') as f:
	pickle.dump(mdict,f)

###########################################################################
## compute iou on points
segment_pts_list = []
segment_name_list = []
segment_id_list = []
trees = []

pts_names = os.listdir(merge_leaf_dir)
pts_names.sort()
for pts_name in pts_names:
    if len(pts_name.split('.'))>1 and pts_name.split('.')[1]=='pts':
        id = int(pts_name.split('.')[0].split('_')[1])
        pts = getPTS(os.path.join(merge_leaf_dir,pts_name))
        segment_pts_list.append(pts)
        segment_name_list.append(pts_name)
        segment_id_list.append(id)

internal_pts_names = os.listdir(internal_dir)
internal_pts_names.sort()
for internal_pts_name in internal_pts_names:
    if len(internal_pts_name.split('.'))>1 and internal_pts_name.split('.')[1]=='pts':
        id = int(internal_pts_name.split('.')[0].split('_')[1])
        pts = getPTS(os.path.join(internal_dir,internal_pts_name))
        segment_pts_list.append(pts)
        segment_name_list.append(pts_name)
        segment_id_list.append(id)

gtInfoFile = open(os.path.join(g_path,'gt_info.txt'),'r')
lines = gtInfoFile.readlines()
count = len(lines)
gt_label = np.zeros(count)
gtInfoFile = open(os.path.join(g_path,'gt_info.txt'),'r')
count = 0
while 1:
    line = gtInfoFile.readline()
    if not line:
        break
    L = line.split()
    gt_label[count] = float(L[14])
    count=count+1

gt_objects_path = os.path.join(g_gt_path,g_path.split('/')[-1].split('.')[0],g_path.split('/')[-1].split('.')[1],'Annotations')
gt_objects_names = os.listdir(gt_objects_path)
gt_objects_names.sort()
gt_pts_list = []
for i in range(len(gt_objects_names)):
    pts_file = os.path.join(gt_objects_path,'objects_'+str(i)+'.pts')
    pts = getPTS(pts_file,1)
    gt_pts_list.append(pts)


##########################################################################
## compute node info
node_label = np.zeros(node_obb.shape[0])
node_iou = np.zeros(node_obb.shape[0])
iou_matrix_on_points = np.zeros((gt_obb.shape[0],node_obb.shape[0]))

for i in range(len(segment_pts_list)):
    for j in range(len(gt_pts_list)):
        id = segment_id_list[i]
        iou = computePtsIou(segment_pts_list[i],gt_pts_list[j],node_obb[id,:],gt_obb[j,:])
        iou_matrix_on_points[j,id] = iou
        if iou > node_iou[id]:
            node_label[id] = gt_label[j]
            node_iou[id] = iou

output = open(os.path.join(files_dir,'node_info_on_points.txt'),'w')
for i in range(0,node_obb.shape[0]):
    if node_iou[i] > 0:
        output.write('%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f\n'\
        %(node_obb[i,0],node_obb[i,1],node_obb[i,2],node_obb[i,3],\
        node_obb[i,4],node_obb[i,5],node_obb[i,6],node_obb[i,7],
        node_offset[i,0],node_offset[i,1],node_offset[i,2],node_offset[i,3],\
        node_offset[i,4],node_offset[i,5],node_offset[i,6],node_offset[i,7],\
        node_label[i],node_iou[i]))
    else:
        output.write('%f %f %f %f %f %f %f %f\n'\
        %(node_obb[i,0],node_obb[i,1],node_obb[i,2],node_obb[i,3],\
        node_obb[i,4],node_obb[i,5],node_obb[i,6],node_obb[i,7]))
output.close()

pickle_dir = os.path.join(g_path,'data','files','iou_matrix_on_points.pkl')
mdict={'iou_matrix_on_points': iou_matrix_on_points}
with open(pickle_dir, 'wb+') as f:
	pickle.dump(mdict,f)