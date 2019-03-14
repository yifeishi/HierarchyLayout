import torch
import util
from grassdata import GRASSDatasetTest
import numpy as np
import os
from util_obb import * 


def box2AP(gt_info_path_list,pred_box_path_list,category):
    # load data
    gt_obbs_list = []
    pred_obbs_list = []
    pred_probs_list = []
    for i in range(len(gt_info_path_list)):
        gt_info_path = gt_info_path_list[i]
        pred_box_path = pred_box_path_list[i]

        f1 = open(gt_info_path,'r')
        gt_num = 0
        while 1:
            line = f1.readline()
            if not line:
                break
            L = line.split()
            if int(L[14])==category:
               gt_num += 1
        f1.close()
        
        f2 = open(pred_box_path,'r')
        pred_num = 0
        while 1:
            line = f2.readline()
            if not line:
                break
            L = line.split()
            pred_num += 1
        f2.close()
        
        gt_obbs = np.zeros((gt_num,8))
        pred_obbs = np.zeros((pred_num,8))
        pred_probs = np.zeros(pred_num)
        f3 = open(gt_info_path,'r')
        gt_num = 0
        while 1:
            line = f3.readline()
            if not line:
                break
            L = line.split()
            if int(L[14])==category:
                gt_obbs[gt_num,0] = float(L[6])
                gt_obbs[gt_num,1] = float(L[7])
                gt_obbs[gt_num,2] = float(L[8])
                gt_obbs[gt_num,3] = float(L[9])
                gt_obbs[gt_num,4] = float(L[10])
                gt_obbs[gt_num,5] = float(L[11])
                gt_obbs[gt_num,6] = float(L[12])
                gt_obbs[gt_num,7] = float(L[13])
                gt_num += 1 
        f3.close()
        
        f4 = open(pred_box_path,'r')
        pred_num = 0
        while 1:
            line = f4.readline()
            if not line:
                break
            L = line.split()
            pred_obbs[pred_num,0] = float(L[0])
            pred_obbs[pred_num,1] = float(L[1])
            pred_obbs[pred_num,2] = float(L[2])
            pred_obbs[pred_num,3] = float(L[3])
            pred_obbs[pred_num,4] = float(L[4])
            pred_obbs[pred_num,5] = float(L[5])
            pred_obbs[pred_num,6] = float(L[6])
            pred_obbs[pred_num,7] = float(L[7])
            pred_probs[pred_num] = float(L[9+category])
            pred_num += 1
        f4.close()
       
        gt_obbs_list.append(gt_obbs)
        pred_obbs_list.append(pred_obbs)
        pred_probs_list.append(pred_probs)
    
    # comput iou
    iou_matrix_list = []
    for j in range(len(gt_obbs_list)):
        gt_obbs = gt_obbs_list[j]
        pred_obbs = pred_obbs_list[j]
        pred_probs = pred_probs_list[j]

        insidePoints_gt = []
        for k in range(gt_obbs.shape[0]): 
            insidePoints_gt.append(samplePoints(gt_obbs[k,:]))

        insidePoints_pred = []
        for k in range(pred_obbs.shape[0]):
            insidePoints_pred.append(samplePoints(pred_obbs[k,:]))

        iou_matrix = np.zeros((gt_obbs.shape[0],pred_obbs.shape[0]))
        for k in range(gt_obbs.shape[0]):    
            for m in range(pred_obbs.shape[0]):
                iou = computeObbIou(gt_obbs[k,:],pred_obbs[m,:],insidePoints_gt[k],insidePoints_pred[m])
                iou_matrix[k,m] = iou
        iou_matrix_list.append(iou_matrix)
        
    recall_list = []
    precision_list = []
    recall_list.append(1)
    precision_list.append(0)
    for i in range(0,100):
        TP = 0
        FP = 0
        FN = 0
        gt_box_num = 0
        pred_box_num = 0
        threshold = float(i)/100

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
        #print(TP,FP,FN,gt_box_num)
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
    print(category,ap)
    fw = open('ap.txt','a')
    fw.write('area:%s  category:%d  iou:%f  ap:%f\n'%(config.room_type, config.ap_category,config.IOU,ap))
    fw.close()

config = util.get_args()
grass_data = GRASSDatasetTest(config.data_path)
def my_collate(batch):
    return batch
test_iter = torch.utils.data.DataLoader(grass_data, batch_size=1, shuffle=False, collate_fn=my_collate)

g_path = config.g_path
category = config.ap_category
room_type = config.room_type
gt_info_path_list = []
pred_box_path_list = []
for batch_idx, batch in enumerate(test_iter):
    if room_type not in batch[0].scene_dir:
        continue
    gt_info_path = os.path.join(g_path,batch[0].scene_dir.split('/')[-1],'gt_info.txt')
    pred_box_path = os.path.join(g_path,batch[0].scene_dir.split('/')[-1],'data','files','pred_box.txt')
    gt_info_path_list.append(gt_info_path)
    pred_box_path_list.append(pred_box_path)
box2AP(gt_info_path_list,pred_box_path_list,category)
