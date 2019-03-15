import os
import sys
import numpy as np
import random
import util
import shutil
from util_obb import *
from qhull_2d import *
from min_bounding_rect import *

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

def GetFileSize(filePath):
    filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize
    return round(fsize,2)

config = util.get_args()
g_path = config.g_path

scene_dir = os.path.join(g_path,'scene.txt')
overseg_dir = os.path.join(g_path,'overseg.txt')
gt_info_dir = os.path.join(g_path,'gt_info.txt')
data_dir = os.path.join(g_path,'data')
leaf_dir = os.path.join(data_dir,'leaf_pts')
internal_dir = os.path.join(data_dir,'internal_pts')
if os.path.exists(internal_dir):
    shutil.rmtree(internal_dir)
os.mkdir(internal_dir)
files_dir = os.path.join(data_dir,'files')

##########################################################################
## generate internal pts
print(".................generate internal pts.................")
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
        ptsName = os.path.join(leaf_dir,'objects_'+str(id)+'.pts')
        fpts = open(ptsName)
        while 1:
            line = fpts.readline()
            fi.write(line)
            if not line:
                break
        fpts.close()
    fi.close()
    
##########################################################################
## compute node info
print(".................compute node info.................SLOW")
print("compute aabb obb")
id_pts_dict = {}
pts_names = os.listdir(leaf_dir)
pts_names.sort()
for pts_name in pts_names:
    if len(pts_name.split('.'))>1 and pts_name.split('.')[1]=='pts':
        id = int(pts_name.split('.')[0].split('_')[1])
        pts_file = os.path.join(leaf_dir,pts_name)
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
    
print("read gt info")
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
print(".................compute node info.................")
node_offset = np.zeros((node_obb.shape[0],8))
node_label = np.zeros(node_obb.shape[0])
node_iou = np.zeros(node_obb.shape[0])
print("sample points for gt objects")
insidePointsGT = []
for i in range(0,gt_obb.shape[0]):
    insidePointsGT.append(samplePoints(gt_obb[i,:]))

print("compute iou")
for i in range(0,node_obb.shape[0]-1):
    insidePoints = samplePoints(node_obb[i,:])
    area_node = node_obb[i,0] * node_obb[i,1] * node_obb[i,2]
    for j in range(0,gt_obb.shape[0]): 
        area_gt = gt_obb[j,0] * gt_obb[j,1] * gt_obb[j,2]
        if (area_gt > 0) and (area_node > 0) and (area_node/area_gt > 10 or area_gt/area_node > 10):
            iou = 0
        elif len(insidePoints) < 10 or len(insidePointsGT[j]) < 10:
            iou = 0
        else:
            iou = computeObbIou(node_obb[i,:],gt_obb[j,:],insidePoints,insidePointsGT[j])

        if gt_label[j] == 8:
            aabbIou = computeAabbIOU(node_aabb[i,:],gt_aabb[j,:])
            iou = max(aabbIou, iou)

        if iou > node_iou[i]:
            node_label[i] = gt_label[j]
            node_offset[i] = gt_obb[j,:] - node_obb[i,:]
            node_iou[i] = iou
        
print("write info")
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






