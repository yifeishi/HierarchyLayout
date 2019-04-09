import json
import random
import numpy as np
import os
import util
from sklearn.neighbors import KDTree
from open3d import *
import math

def getPTS(pts_file,colorPara=1):
    fpts = open(pts_file)
    count = 0
    while 1:
        line = fpts.readline()
        if not line:
            break
        count = count + 1
    if count==0:
        return np.zeros(9)
    points = np.zeros((count,9))
    count = 0
    fpts = open(pts_file)
    while 1:
        line = fpts.readline()
        if not line:
            break
        L = line.split(' ')
        points[count,0] = float(L[0])
        points[count,1] = float(L[1])
        points[count,2] = float(L[2])
        points[count,3] = float(L[3])/colorPara
        points[count,4] = float(L[4])/colorPara
        points[count,5] = float(L[5])/colorPara
        points[count,6] = float(L[6])
        points[count,7] = float(L[7])
        points[count,8] = float(L[8])
        count = count + 1
    return points

def computeColorBin(pts):
    color_bin = np.zeros(180)#r 5,g 5,b 5
    for i in range(0,pts.shape[0]):
        id0 = int(pts[i,3]*255/52)
        id1 = int(pts[i,4]*255/52)
        id2 = int(pts[i,5]*255/52)
        id = id0*25+id1*5+id2
        color_bin[id] += 1
    return color_bin

def computeNormalAffinity(normal0,normal1,center0,center1):
    affinity = abs(np.dot(normal0,normal1))
    if normal0[0]*normal0[0]+normal0[1]*normal0[1]+normal0[2]*normal0[2] != 0:
        affinity = affinity/math.sqrt(normal0[0]*normal0[0]+normal0[1]*normal0[1]+normal0[2]*normal0[2])
    else:
        affinity = 0
    if normal1[0]*normal1[0]+normal1[1]*normal1[1]+normal1[2]*normal1[2] != 0:
        affinity = affinity/math.sqrt(normal1[0]*normal1[0]+normal1[1]*normal1[1]+normal1[2]*normal1[2])
    else:
        affinity = 0
    if np.dot(center0-center1,normal0) > 0:
        affinity *= affinity*0.1 + 0.9
    else:
        affinity = affinity
    return affinity

def chiSquareDis(bin0,bin1):
    dis = 0
    for i in range(bin0.shape[0]):
        if bin0[i] + bin1[i] > 0:
            dis += (bin0[i] - bin1[i])*(bin0[i] - bin1[i])/(bin0[i] + bin1[i])*0.5
    return dis

def computeColorAffinity(bin0,bin1):
    dis = chiSquareDis(bin0,bin1)
    if dis == 0:
        affinity = 5
    else:
        affinity = 1/chiSquareDis(bin0,bin1)*20
    return affinity

config = util.get_args()
g_path = config.g_path
scene_dir = os.path.join(g_path,'scene_normal.txt')
overseg_dir = os.path.join(g_path,'overseg.txt')
gt_info_dir = os.path.join(g_path,'gt_info.txt')
data_dir = os.path.join(g_path,'data')
leaf_dir = os.path.join(data_dir,'leaf_pts')
merge_leaf_dir = os.path.join(data_dir,'merge_leaf_pts')
internal_dir = os.path.join(data_dir,'internal_pts')
files_dir = os.path.join(data_dir,'files')

##########################################################################
## generate merge leaf pts
pts = getPTS(scene_dir)
z_max = pts[:,2].max()
z_min = pts[:,2].min()

pts_names = os.listdir(merge_leaf_dir)
pts_names.sort()
count=0
for pts_name in pts_names:
    if len(pts_name.split('.'))>1 and pts_name.split('.')[1]=='pts':
        count=count+1

segment_pts_list = []
segment_name_list = []
segment_id_list = []
segment_pts_type_list = [] #'other':0,'floor':1,'ceiling':2
trees = []
for pts_name in pts_names:
    if len(pts_name.split('.'))>1 and pts_name.split('.')[1]=='pts':
        id = int(pts_name.split('.')[0].split('_')[1])
        pts = getPTS(os.path.join(merge_leaf_dir,pts_name))
        if pts.shape[0] < 10:
            trees.append(None)
            segment_pts_list.append('None')
            segment_name_list.append(pts_name)
            segment_id_list.append(id)
            segment_pts_type_list.append(np.zeros(pts.shape[0]))
            continue
        tree = KDTree(pts[:,:3], leaf_size=2)
        trees.append(tree)
        segment_pts_list.append(pts)
        segment_name_list.append(pts_name)
        segment_id_list.append(id)
        segment_pts_type_list.append(np.zeros(pts.shape[0]))

for i in range(len(segment_pts_list)):
    segment_pts = segment_pts_list[i]
    if segment_pts is 'None':
        continue
    for j in range(segment_pts.shape[0]):
        if segment_pts[j,2] > z_min and segment_pts[j,2] < z_min + 0.2:
            segment_pts_type_list[i][j] = 1
        elif segment_pts[j,2] > z_max - 0.5 and segment_pts[j,2] < z_max:
            segment_pts_type_list[i][j] = 2

segment_type_list = []
for i in range(len(segment_pts_list)):
    segment_pts_type = segment_pts_type_list[i]
    type0 = np.where(segment_pts_type==0,1,0)
    type1 = np.where(segment_pts_type==1,1,0)
    type2 = np.where(segment_pts_type==2,1,0)
    num0 = type0.sum()
    num1 = type1.sum()
    num2 = type2.sum()
    if max(num0,num1,num2) == num0:
        segment_type_list.append(0)
    elif max(num0,num1,num2) == num1:
        segment_type_list.append(1)
    elif max(num0,num1,num2) == num2:
        segment_type_list.append(2)

##########################################################################
## compute affinity matrix
dis_matrix = np.zeros((len(trees),len(trees)))
for i in range(len(trees)):
    if segment_type_list[i] != 0:
        continue
    tree = trees[i]
    if tree == None:
        continue
    for j in range(len(segment_pts_list)):
        pts = segment_pts_list[j]
        if segment_type_list[j] != 0 or i >= j or pts is 'None':
            continue
        min_dis = 9999
        for k in range(pts.shape[0]):
            sample_prob = 0.1
            if random.random() > sample_prob:
                continue
            dist,ind = tree.query(pts[k:k+1,:3], k=1)
            if min_dis > dist[0]:
                min_dis = dist[0]
        dis_matrix[segment_id_list[i],segment_id_list[j]] = min_dis
        dis_matrix[segment_id_list[j],segment_id_list[i]] = min_dis

segment_size_list = []
segment_cen_list = []
for i in range(len(segment_pts_list)):
    if segment_pts_list[i] is 'None':
        segment_size_list.append(0)
        continue
    segment_size_list.append(segment_pts_list[i].shape[0])
    segment_cen_list.append(segment_pts_list[i].mean(0)[0:3])

segment_color_bin_list = []
for i in range(len(segment_pts_list)):
    if segment_pts_list[i] is 'None':
        segment_color_bin_list.append(np.zeros(180))
        continue
    color_bin = computeColorBin(segment_pts_list[i])
    segment_color_bin_list.append(color_bin)


affinity_matrix = np.zeros((len(segment_pts_list),len(segment_pts_list)))
for i in range(len(segment_pts_list)):
    for j in range(len(segment_pts_list)):
        id0 = segment_id_list[i]
        id1 = segment_id_list[j]
        if id0 >= id1:
            continue
        if segment_pts_list[i] is 'None' or segment_pts_list[j] is 'None':
            continue
        if dis_matrix[id0,id1] == 0:
            affinity_matrix[id0,id1] = 0
            affinity_matrix[id1,id0] = affinity_matrix[id0,id1]
            continue 
        affinity_matrix[id0,id1] = computeColorAffinity(segment_color_bin_list[id0],segment_color_bin_list[id1]) / dis_matrix[id0,id1] * 1000
        if segment_cen_list[id0][2] > 2:
            affinity_matrix[id0,id1] *= 0.5
        if segment_cen_list[id1][2] > 2:
            affinity_matrix[id0,id1] *= 0.5
        affinity_matrix[id1,id0] = affinity_matrix[id0,id1]

output = open(os.path.join(files_dir,'merge_segment_affinity_matrix.txt'),'w')
for i in range(0,affinity_matrix.shape[0]):
    for j in range(0,affinity_matrix.shape[1]):
        if affinity_matrix[i,j] == 0:
            output.write('1 ')
        else:
            output.write('%f '%(affinity_matrix[i,j]))
    output.write('\n')
output.close()