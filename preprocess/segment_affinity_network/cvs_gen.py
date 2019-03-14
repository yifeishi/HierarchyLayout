import os
import math
import numpy as np
import random

data_path = 'F:\project\sceneparsing\detection.v1\detections.v1'
data_path2 = 'F:\project\sceneparsing\matterport_data\matterport_data'
output_file = '.\siamese_feature.txt'

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

def computeLayout(obb1,obb2):
    layout = np.zeros(7)
    obb1_t = ObbFeatureTransformerReverse(obb1)
    obb2_t = ObbFeatureTransformerReverse(obb2)
    layout[0] = obb1_t[5] - obb2_t[5]
    layout[1] = obb1_t[5] - obb2_t[2]
    layout[2] = obb1_t[2] - obb2_t[5]
    layout[3] = math.sqrt(math.pow(obb1[3]-obb2[3],2) + math.pow(obb1[4]-obb2[4],2) + math.pow(obb1[5]-obb2[5],2))
    layout[4] = layout[3] # need to change
    S1 = obb1[0]*obb1[1]*obb1[2]
    S2 = obb2[0]*obb2[1]*obb2[2]
    SI= max(0, min(obb1_t[0], obb2_t[0]) - max(obb1_t[3], obb2_t[3])) * \
        max(0, min(obb1_t[1], obb2_t[1]) - max(obb1_t[4], obb2_t[4])) * \
        max(0, min(obb1_t[2], obb2_t[2]) - max(obb1_t[5], obb2_t[5]))
    layout[5] = SI/S1
    layout[6] = SI/S2
    return layout

def getLayout(dir1,id1,dir2,id2):
    obb_dir = os.path.join(dir1,'obb_0.txt')
    obb_file = open(obb_dir)
    L = obb_file.readlines()
    num = len(L)
    obbs = np.zeros((num,6))
    obb_file = open(obb_dir)
    count = 0
    while 1:
        line = obb_file.readline()
        if not line:
            break
        L = line.split()
        obbs[count,0] = float(L[3])
        obbs[count,1] = float(L[4])
        obbs[count,2] = float(L[5])
        obbs[count,3] = float(L[0])
        obbs[count,4] = float(L[1])
        obbs[count,5] = float(L[2])
        count = count + 1
    
    obb1 = obbs[id1,:]
    obb2 = obbs[id2,:]
    layout_ap = computeLayout(obb1,obb2)
    return layout_ap

def getLayoutNeg(dir1,id1,dir2,id2,obb3):
    obb_dir = os.path.join(dir1,'obb_0.txt')
    obb_file = open(obb_dir)
    L = obb_file.readlines()
    num = len(L)
    obbs = np.zeros((num,6))
    obb_file = open(obb_dir)
    count = 0
    while 1:
        line = obb_file.readline()
        if not line:
            break
        L = line.split()
        obbs[count,0] = float(L[3])
        obbs[count,1] = float(L[4])
        obbs[count,2] = float(L[5])
        obbs[count,3] = float(L[0])
        obbs[count,4] = float(L[1])
        obbs[count,5] = float(L[2])
        count = count + 1
    
    obb1 = obbs[id1,:]
    obb2 = obbs[id2,:]
    obb3[3:5] = obb2[3:5]
    layout_an = computeLayout(obb1,obb3)
    return layout_an

def randomlySelectNeg(room_names):
    room_id = int(random.random()*len(room_names))
    room_name = room_names[room_id]
    room_dir = os.path.join(data_path,room_name)
    
    file_names = os.listdir(room_dir)
    object_names = []
    object_ids = []
    for file_name in file_names:
        if file_name.split('_')[0] == 'object' and file_name.split('_')[2] == 'pointnet' and file_name.split('_')[3] == 'feature.txt':
            object_names.append(file_name)
            object_ids.append(int(file_name.split('_')[1]))

    obb_dir = os.path.join(room_dir,'obb_0.txt')
    obb_file = open(obb_dir)
    L = obb_file.readlines()
    num = len(L)
    obbs = np.zeros((num,6))
    obb_file = open(obb_dir)
    count = 0
    while 1:
        line = obb_file.readline()
        if not line:
            break
        L = line.split()
        obbs[count,0] = float(L[3])
        obbs[count,1] = float(L[4])
        obbs[count,2] = float(L[5])
        obbs[count,3] = float(L[0])
        obbs[count,4] = float(L[1])
        obbs[count,5] = float(L[2])
        count = count + 1

    object_id = int(random.random()*num)
    obb3 = obbs[object_id,:]
    object_dir = os.path.join(room_dir,object_names[object_id])
    return object_dir,room_dir,object_id,obb3

def writeFeaturesToFile(object_dir1,object_dir2,object_dir3,layout_ap,layout_an):
    file1 = open(object_dir1)
    L1 = file1.readline()
    file1.close()

    file2 = open(object_dir2)
    L2 = file2.readline()
    file2.close()

    file3 = open(object_dir3)
    L3 = file3.readline()
    file3.close()

    output = open(output_file, 'a')
    output.write('%s,%s,%s,'%(object_dir1.strip(),object_dir2.strip(),object_dir3.strip()))
    output.write('%f,%f,%f,%f,%f,%f,%f,'%(layout_ap[0],layout_ap[1],layout_ap[2],layout_ap[3],layout_ap[4],layout_ap[5],layout_ap[6]))
    output.write('%f,%f,%f,%f,%f,%f,%f,\n'%(layout_an[0],layout_an[1],layout_an[2],layout_an[3],layout_an[4],layout_an[5],layout_an[6]))
    output.close()
    return 0

# for folder in data_path
#   get object #
#   for each two object: compute the layout(obb_0.txt); find another object with different catgeory (F:\project\sceneparsing\matterport_data\matterport_data\1LXtFkjw3qL\house_features\region_0\gt_box.txt),compute the layout; write(object_dir_a,object_dir_p,object_dir_n,layout_ap,layout_an);

room_names = os.listdir(data_path)
for room_name in room_names:
    room_dir = os.path.join(data_path,room_name)
    file_names = os.listdir(room_dir)
    object_names = []
    object_ids = []
    for file_name in file_names:
        if file_name.split('_')[0] == 'object' and file_name.split('_')[2] == 'pointnet' and file_name.split('_')[3] == 'feature.txt':
            object_names.append(file_name)
            object_ids.append(int(file_name.split('_')[1]))
    print(room_name)
    
    count1 = 0
    for object_name1 in object_names:
        count2 = 0
        for object_name2 in object_names:
            if count1 == count2:
                continue
            object_dir1 = os.path.join(room_dir,object_name1)
            object_dir2 = os.path.join(room_dir,object_name2)
            layout_ap = getLayout(room_dir,object_ids[count1],room_dir,object_ids[count2])
            #object_category2 = getObjectCategory(os.path.join(data_path2,room_name.split('_')[0],'house_feature',room_name.split('_')[1],'gt_box.txt'),count2)#10
            object_dir3,room_dir3,count3,obb3 = randomlySelectNeg(room_names) # search other room, random select an object, get the 'object_feautre.txt' and object size
            layout_an = getLayoutNeg(room_dir,object_ids[count1],room_dir,object_ids[count2],obb3) # add object2 location
            
            writeFeaturesToFile(object_dir1,object_dir2,object_dir3,layout_ap,layout_an)
            #write to file object_dir1,object_dir2,object_dir3,layout_ap,layout_an
            count2 = count2 + 1
        count1 = count1 + 1