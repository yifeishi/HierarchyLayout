import json
import random
import numpy as np
import os
import util
from sklearn.neighbors import KDTree
from open3d import *
from util_obb import *
import math

def computeColorBin(pts):
    color_bin = np.zeros(180)#r 5,g 5,b 5
    for i in range(0,pts.shape[0]):
        id0 = int(pts[i,3]*255/52)
        id1 = int(pts[i,4]*255/52)
        id2 = int(pts[i,5]*255/52)
        id = id0*25+id1*5+id2
        color_bin[id] += 1
    return color_bin

def computeNormalAffinity(normal0,normal1):
    affinity = abs(np.dot(normal0,normal1))
    if normal0[0]*normal0[0]+normal0[1]*normal0[1]+normal0[2]*normal0[2] != 0:
        affinity = affinity/math.sqrt(normal0[0]*normal0[0]+normal0[1]*normal0[1]+normal0[2]*normal0[2])
    else:
        affinity = 0
    if normal1[0]*normal1[0]+normal1[1]*normal1[1]+normal1[2]*normal1[2] != 0:
        affinity = affinity/math.sqrt(normal1[0]*normal1[0]+normal1[1]*normal1[1]+normal1[2]*normal1[2])
    else:
        affinity = 0
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

scene_dir = os.path.join(g_path,'scene.txt')
overseg_dir = os.path.join(g_path,'overseg.txt')
gt_info_dir = os.path.join(g_path,'gt_info.txt')
data_dir = os.path.join(g_path,'data')
leaf_dir = os.path.join(data_dir,'leaf_pts')
internal_dir = os.path.join(data_dir,'internal_pts')
files_dir = os.path.join(data_dir,'files')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(leaf_dir):
    os.mkdir(leaf_dir)
if not os.path.exists(internal_dir):
    os.mkdir(internal_dir)
if not os.path.exists(files_dir):
    os.mkdir(files_dir)

##########################################################################
## generate leaf pts
print(".................generate leaf pts.................SLOW")
print("load scene txt")
pts = getPTS(scene_dir)

print("load label txt")
f = open(overseg_dir,'r')
line_num = len(f.readlines())
f.close()
labels = np.zeros(line_num)
f = open(overseg_dir,'r')
count = 0
while 1:
    line = f.readline()
    if not line:
        break
    labels[count] = line.strip()
    count+=1
max_label = np.max(labels)
min_label = np.min(labels)

print("write leaf pts")
for i in range(0,int(max_label)):
    label = i+1
    count = 0
    output = open(leaf_dir+'/objects_'+str(i)+'.pts','w')
    for j in range(0,pts.shape[0]):
        if labels[j] == label:
            output.write('%f %f %f %f %f %f\n'%(pts[j,0],pts[j,1],pts[j,2],\
            float(pts[j,3])/255,float(pts[j,4])/255,float(pts[j,5])/255))
            count+=1
    output.close()

    output = open(leaf_dir+'/objects_'+str(i)+'.ply','w')
    output.write("ply\n");
    output.write("format ascii 1.0\n");
    output.write("element vertex %d\n"%count);
    output.write("property float x\n");
    output.write("property float y\n");
    output.write("property float z\n");
    output.write("property uchar red\n");
    output.write("property uchar green\n");
    output.write("property uchar blue\n");
    output.write("property uchar alpha\n");
    output.write("end_header\n");
    for j in range(0,pts.shape[0]):
        if labels[j] == label:
            output.write('%f %f %f %d %d %d 0\n'%(pts[j,0],pts[j,1],pts[j,2],\
            int(pts[j,3]),int(pts[j,4]),int(pts[j,5])))
    output.close()


##########################################################################
## remove floor and ceiling
print(".................remove floor and ceiling.................")
pts_names = os.listdir(leaf_dir)
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
        pts = getPTS(os.path.join(leaf_dir,pts_name))
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

# compute zmax and zmin
z_max = -9999999
z_min = 9999999
pts_num = 0
for i in range(len(segment_pts_list)):
    segment_pts = segment_pts_list[i]
    if segment_pts is 'None':
        continue
    if z_max < segment_pts[:,2].max():
        z_max = segment_pts[:,2].max()
    if z_min > segment_pts[:,2].min():
        z_min =segment_pts[:,2].min()
#print('z_max',z_max)
#print('z_min',z_min)

# determine the labels for each patch
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
## compute adjacent matrix
print(".................compute adjacent matrix.................SLOW")
adj_matrix = np.zeros((len(trees),len(trees)))
count_adj = 0
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
        neighborFlag = False
        neighborCount = 0
        for k in range(pts.shape[0]):
            sample_prob = 0.2
            if random.random() > sample_prob:
                continue
            if k > pts.shape[0]*0.5*sample_prob and neighborCount == 0:
                break
            dist,ind = tree.query(pts[k:k+1,:3], k=1)
            if dist[0,0] < 0.1:
                neighborCount += 1
            if neighborCount >= 5:
                neighborFlag = True
                break
        if neighborFlag:
            adj_matrix[segment_id_list[i],segment_id_list[j]] = 1
            adj_matrix[segment_id_list[j],segment_id_list[i]] = 1
            count_adj += 1
print(float(count_adj)/len(trees),'connections per object')

output = open(os.path.join(files_dir,'segment_adjacent_matrix.txt'),'w')
for i in range(0,adj_matrix.shape[0]):
    for j in range(0,adj_matrix.shape[1]):
    	output.write('%d '%(adj_matrix[i,j]))
    output.write('\n')
output.close()


##########################################################################
## compute affinity matrix
print(".................compute affinity matrix.................")
affinity_matrix_dir = os.path.join(files_dir,'segment_affinity_matrix.txt')

print("compute normal")
segment_normal_list = []
for i in range(len(segment_pts_list)):
    if segment_pts_list[i] is 'None':
        segment_normal_list.append(np.zeros(3))
        continue
    pcd = read_point_cloud(os.path.join(leaf_dir,segment_name_list[i].split('.')[0]+'.ply'))
    estimate_normals(pcd, search_param = KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
    normals = np.asarray(pcd.normals)
    segment_normal_list.append(normals.mean(0))
print('get normal',len(segment_normal_list))

print("compute color bin")
segment_color_bin_list = []
for i in range(len(segment_pts_list)):
    if segment_pts_list[i] is 'None':
        segment_color_bin_list.append(np.zeros(180))
        continue
    color_bin = computeColorBin(segment_pts_list[i])
    segment_color_bin_list.append(color_bin)
print('get color bin',len(segment_color_bin_list))

print("compute affinity")
color_affinity_matrix = np.zeros((len(segment_pts_list),len(segment_pts_list)))
normal_affinity_matrix = np.zeros((len(segment_pts_list),len(segment_pts_list)))
affinity_matrix = np.zeros((len(segment_pts_list),len(segment_pts_list)))

for i in range(len(segment_pts_list)):
    for j in range(len(segment_pts_list)):
        id0 = segment_id_list[i]
        id1 = segment_id_list[j]
        if adj_matrix[id0,id1] == 0 or id0 >= id1:
            continue
        if segment_pts_list[i] is 'None' or segment_pts_list[j] is 'None':
            continue
        color_affinity_matrix[id0,id1] = computeColorAffinity(segment_color_bin_list[id0],segment_color_bin_list[id1])
        color_affinity_matrix[id1,id0] = color_affinity_matrix[id0,id1]
        normal_affinity_matrix[id0,id1] = computeNormalAffinity(segment_normal_list[id0],segment_normal_list[id1])
        normal_affinity_matrix[id1,id0] = normal_affinity_matrix[id0,id1]
        affinity_matrix[id0,id1] = color_affinity_matrix[id0,id1] * normal_affinity_matrix[id0,id1]
        affinity_matrix[id1,id0] = affinity_matrix[id0,id1]

output = open(os.path.join(files_dir,'segment_affinity_matrix.txt'),'w')
for i in range(0,affinity_matrix.shape[0]):
    for j in range(0,affinity_matrix.shape[1]):
        if affinity_matrix[i,j] == 0:
    	    output.write('%d '%(affinity_matrix[i,j]))
        else:
            output.write('%f '%(affinity_matrix[i,j]))
    output.write('\n')
output.close()
