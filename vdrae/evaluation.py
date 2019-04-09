import util
import numpy as np
import os
from util_obb import * 
from sklearn.neighbors import KDTree

config = util.get_args()
g_path = config.g_path
g_gt_path = config.g_gt_path

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
        if random.random() > 0.99:
            continue
        dist,ind = tree1.query(pts0[i:i+1,:3], k=1)
        if dist[0] < 0.1:
            count_in0 += 1
        count_all0 += 1
    for i in range(pts1.shape[0]):
        if random.random() > 0.99:
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

def box2AP(gt_pts_dirs_list,node_pts_dirs_list,gt_info_path_list,pred_box_path_list):
    gt_obbs_list = []
    pred_obbs_list = []
    pred_probs_list = []
    valid_gt_pts_dirs_list = []
    valid_pred_pts_dirs_list = []
    prob_list = []
    for i in range(len(gt_info_path_list)):
        gt_info_path = gt_info_path_list[i]
        pred_box_path = pred_box_path_list[i]
        gt_pts_dirs = gt_pts_dirs_list[i]
        node_pts_dirs = node_pts_dirs_list[i]

        lines = open(gt_info_path,'r').readlines()
        gt_num = 0
        for line in lines:
            L = line.split()
            if int(L[14]) == config.ap_category:
               gt_num += 1

        lines = open(pred_box_path,'r').readlines()
        pred_num = 0
        for line in lines:
            L = line.split()
            if float(L[9+config.ap_category]) > 0.001:
                pred_num += 1
                prob_list.append(float(L[9+config.ap_category]))

        gt_obbs = np.zeros((gt_num,8))
        pred_obbs = np.zeros((pred_num,8))
        pred_probs = np.zeros(pred_num)
        valid_gt_pts_dirs = []#
        valid_pred_pts_dirs = []#

        lines = open(gt_info_path,'r').readlines()
        gt_num = 0
        count = 0
        for line in lines:
            L = line.split()
            if int(L[14]) == config.ap_category:
                gt_obbs[gt_num,0] = float(L[6])
                gt_obbs[gt_num,1] = float(L[7])
                gt_obbs[gt_num,2] = float(L[8])
                gt_obbs[gt_num,3] = float(L[9])
                gt_obbs[gt_num,4] = float(L[10])
                gt_obbs[gt_num,5] = float(L[11])
                gt_obbs[gt_num,6] = float(L[12])
                gt_obbs[gt_num,7] = float(L[13])
                valid_gt_pts_dirs.append(gt_pts_dirs[count])
                gt_num += 1
            count += 1

        lines = open(pred_box_path,'r').readlines()
        pred_num = 0
        count = 0
        for line in lines:
            L = line.split()
            if float(L[9+config.ap_category]) > 0.001:
                pred_obbs[pred_num,0] = float(L[0])
                pred_obbs[pred_num,1] = float(L[1])
                pred_obbs[pred_num,2] = float(L[2])
                pred_obbs[pred_num,3] = float(L[3])
                pred_obbs[pred_num,4] = float(L[4])
                pred_obbs[pred_num,5] = float(L[5])
                pred_obbs[pred_num,6] = float(L[6])
                pred_obbs[pred_num,7] = float(L[7])
                pred_probs[pred_num] = float(L[9+config.ap_category])
                valid_pred_pts_dirs.append(node_pts_dirs[count])
                pred_num += 1
            count += 1
        gt_obbs_list.append(gt_obbs)
        pred_obbs_list.append(pred_obbs)
        pred_probs_list.append(pred_probs)
        valid_gt_pts_dirs_list.append(valid_gt_pts_dirs)
        valid_pred_pts_dirs_list.append(valid_pred_pts_dirs)
    
    prob_list = np.array(prob_list)
    prob_list.sort()
    prob_list = np.unique(prob_list)

    # comput iou
    iou_matrix_list = []
    gt_num_all = 0
    for j in range(len(gt_obbs_list)):
        gt_obbs = gt_obbs_list[j]
        pred_obbs = pred_obbs_list[j]
        pred_probs = pred_probs_list[j]
        valid_gt_pts_dirs = valid_gt_pts_dirs_list[j]
        valid_pred_pts_dirs = valid_pred_pts_dirs_list[j]
        iou_matrix = np.zeros((gt_obbs.shape[0],pred_obbs.shape[0]))
        for k in range(gt_obbs.shape[0]):
            for m in range(pred_obbs.shape[0]):
                iou = computePtsIou(getPTS(valid_gt_pts_dirs[k]),getPTS(valid_pred_pts_dirs[m]),gt_obbs[k,:],pred_obbs[m,:])
                iou_matrix[k,m] = iou
        iou_matrix_list.append(iou_matrix)
        gt_num_all += len(gt_obbs)
        
    recall_list = []
    precision_list = []
    for i in range(-1,prob_list.shape[0]+1):
        TP = 0
        FP = 0
        FN = 0
        gt_box_num = 0
        pred_box_num = 0
        if i == -1:
            threshold = prob_list[i+1] * 0.5
        elif i == prob_list.shape[0]:
            threshold = (1-prob_list[i-1])*0.5+prob_list[i-1]
        else:
            threshold = prob_list[i]

        for j in range(len(gt_obbs_list)):
            iou_matrix = iou_matrix_list[j]
            gt_obbs = gt_obbs_list[j]
            pred_obbs = pred_obbs_list[j]
            pred_probs = pred_probs_list[j]
            gt_box_num += gt_obbs_list[j].shape[0]
            pred_box_num += pred_obbs_list[j].shape[0]
            
            # true-pos
            for k in range(gt_obbs.shape[0]):    
                for m in range(pred_obbs.shape[0]):
                    iou = iou_matrix[k,m]
                    if iou > config.IOU and pred_probs[m] >= threshold:
                        TP += 1
                        break

            # false-pos
            for m in range(pred_obbs.shape[0]):
                if pred_probs[m] < threshold:
                    continue
                FPflag = True
                for k in range(gt_obbs.shape[0]):
                    iou = iou_matrix[k,m]
                    if iou > config.IOU:
                        FPflag = False
                if FPflag:
                    FP += 1
        recall = float(TP)/gt_box_num
        if TP+FP>0:
            precision = float(TP)/(TP+FP)
            recall_list.append(recall)
            precision_list.append(precision)
    recall_list.append(0)
    precision_list.append(1)
    ap = 0
    length = 0
    for i in range(0, len(recall_list)-1):
        r = recall_list[i]
        p = precision_list[i]
        r_n = recall_list[i+1]
        p_n = precision_list[i+1]
        ap += abs(r - r_n)*p_n
        length += (r - r_n)
    print(config.ap_category,ap,gt_num_all)
    fw = open('ap.txt','a')
    fw.write('category:%d  iou:%f  ap:%f\n'%(config.ap_category,config.IOU,ap))
    fw.close()

scene_names = os.listdir(g_path)
scene_names.sort()
gt_info_path_list = []
pred_box_path_list = []

node_pts_dirs_list = []
gt_pts_dirs_list = []

f = open(scene_list_path)
lines = f.readlines()
scene_list = []
for line in lines:
    scene_list.append(line.split()[0])

for scene_name in scene_names:
    if scene_name.split('.')[0] not in scene_list:
        continue
    scene_dir = os.path.join(g_path,scene_name)
    box_dir = os.path.join(scene_dir,'data','files','pred_box_vdrae_for_eval.txt')
    merge_leaf_pts_dir = os.path.join(scene_dir,'data','merge_leaf_pts')
    internal_pts_dir = os.path.join(scene_dir,'data','internal_pts')
    gt_info_path = os.path.join(scene_dir,'gt_info.txt')
    if not os.path.exists(box_dir):
        continue
    scene_count += 1
    gt_info_path_list.append(gt_info_path)
    pred_box_path_list.append(box_dir)

    node_pts_dirs = []
    gt_pts_dirs = []
    node_pts_num = len(open(box_dir,'r').readlines())
    for line in open(box_dir,'r').readlines():
        L = line.split()
        if os.path.exists(os.path.join(merge_leaf_pts_dir,'objects_'+str(L[49])+'.pts')):
            node_pts_dirs.append(os.path.join(merge_leaf_pts_dir,'objects_'+str(L[49])+'.pts'))
        elif os.path.exists(os.path.join(internal_pts_dir,'objects_'+str(L[49])+'.pts')):
            node_pts_dirs.append(os.path.join(internal_pts_dir,'objects_'+str(L[49])+'.pts'))
        
    gt_pts_num = len(open(gt_info_path,'r').readlines())
    for i in range(gt_pts_num): 
        gt_pts_dirs.append(os.path.join(g_gt_path, scene_name.split('.')[0], scene_name.split('.')[1], 'Annotations', 'objects_'+str(i)+'.pts'))
    node_pts_dirs_list.append(node_pts_dirs)
    gt_pts_dirs_list.append(gt_pts_dirs)
box2AP(gt_pts_dirs_list,node_pts_dirs_list,gt_info_path_list,pred_box_path_list)
