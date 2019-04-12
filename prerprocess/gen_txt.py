import os
import numpy as np
import util

config = util.get_args()
g_path = config.g_path

data_dir = os.path.join(g_path,'data')
merge_leaf_dir = os.path.join(data_dir,'merge_leaf_pts')
internal_dir = os.path.join(data_dir,'internal_pts')
files_dir = os.path.join(data_dir,'files')
node_info_dir = os.path.join(files_dir,'node_info_on_points.txt')
pred_box_txt_dir = os.path.join(files_dir,'pred_box_all_node.txt')

id_pts_dict = {}
id_pts_feature_dict = {}
pts_names = os.listdir(merge_leaf_dir)
pts_names.sort()
for pts_name in pts_names:
    if len(pts_name.split('.'))>1 and pts_name.split('.')[1]=='pts':
        id = int(pts_name.split('.')[0].split('_')[1])
        pts_file = os.path.join(merge_leaf_dir,pts_name)
        pts_feature_file  = os.path.join(merge_leaf_dir,pts_name.split('.')[0]+'_pointcnn_feature.txt')
        id_pts_dict[id] = pts_file
        id_pts_feature_dict[id] = pts_feature_file
internal_pts_names = os.listdir(internal_dir)
internal_pts_names.sort()
for internal_pts_name in internal_pts_names:
    if len(internal_pts_name.split('.'))>1 and internal_pts_name.split('.')[1]=='pts':
        id = int(internal_pts_name.split('.')[0].split('_')[1])
        pts_file= os.path.join(internal_dir,internal_pts_name)
        pts_feature_file  = os.path.join(internal_dir,internal_pts_name.split('.')[0]+'_pointcnn_feature.txt')
        id_pts_dict[id] = pts_file
        id_pts_feature_dict[id] = pts_feature_file

output = open('Matterport_all_node_pointcnn_input.txt','a')
f = open(node_info_dir,'r')
lines = f.readlines()
for i in range(len(lines)):
    L = lines[i].split()
    for j in range(len(L)):
        L[j] = float(L[j])
    if len(L) > 8:
        output.write('%s %f %f %f %f %f %f %f %f %d %s %s\n'\
            %(id_pts_dict[i],L[0],L[1],L[2],L[3],\
            L[4],L[5],L[6],L[7],\
            L[16],pred_box_txt_dir,id_pts_feature_dict[i]))
    else:
        neg = float(99)
        output.write('%s %f %f %f %f %f %f %f %f %d %s %s\n'\
            %(id_pts_dict[i],L[0],L[1],L[2],L[3],\
            L[4],L[5],L[6],L[7],\
            neg,pred_box_txt_dir,id_pts_feature_dict[i]))

for i in range(64):
    L = lines[0].split()
    for j in range(len(L)):
        L[j] = float(L[j])
    neg = float(99)
    pred_box_txt_dir = 'tmp.txt'
    output.write('%s %f %f %f %f %f %f %f %f %d %s %s\n'\
            %(id_pts_dict[i],L[0],L[1],L[2],L[3],\
            L[4],L[5],L[6],L[7],\
            neg,pred_box_txt_dir,pred_box_txt_dir))
output.close()
