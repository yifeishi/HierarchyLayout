import torch
import util
import grassmodel
from grassdata import GRASSDatasetTest
from grassmodel import GRASSEncoderDecoder
import os
import random
import pickle
from util_obb import * 
from sklearn.neighbors import KDTree


def NMS(predictions,obbs,labels):
    visited = np.zeros(len(obbs))
    flags = np.ones(len(obbs))
    
    insidePoints = []
    for i in range(len(obbs)):
        insidePoints.append(samplePoints(obbs[i]))
    
    while True:
        max_probs = np.zeros(len(obbs))
        for i in range(len(predictions)):
            prediction = predictions[i]
            label = np.where(prediction == np.max(prediction))[0]
            if visited[i] == 0:
                max_probs[i] = prediction[label]
        index = np.where(max_probs == np.max(max_probs))[0]
        index_one = -1
        for i in range(len(index)):
            if visited[index[i]] == 0:
                index_one = index[i]
                break
        for i in range(len(predictions)):
            if i == index_one:
                flags[i] = 1
                visited[i] = 1
                continue
            if visited[i] == 1:
                continue
            if labels[i] != labels[index_one]:
                continue
            iou = computeObbIou(obbs[index_one],obbs[i],insidePoints[index_one],insidePoints[i])
            IOU = 0.5
            if iou > IOU:
                flags[i] = 0
                visited[i] = 1
        stopFlag = 1
        num = 0
        for i in range(len(predictions)):
            if visited[i] == 0:
                stopFlag = 0
                num += 1
        if stopFlag:
            break
    return flags

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
        if random.random() > 0.05:
            continue
        dist,ind = tree1.query(pts0[i:i+1,:3], k=1)
        if dist[0] < 0.1:
            count_in0 += 1
        count_all0 += 1
    for i in range(pts1.shape[0]):
        if random.random() > 0.05:
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

def NMS_on_points(predictions,obbs,pts_dirs,labels):
    visited = np.zeros(len(obbs))
    flags = np.ones(len(obbs))
    
    countLabelCorrect = 0
    for i in range(len(labels)):
        prediction = predictions[i]
        validFlag = False
        #print('prediction',prediction)
        for category in categories:
            if prediction[category] > 0.01:
                validFlag = True
                break
        if validFlag == False:
            flags[i] = 0
            visited[i] = 1
        else:
            countLabelCorrect += 1
    print('node before NMS',countLabelCorrect)
    if len(obbs) == 0 or countLabelCorrect == 0:
        flags = np.zeros(len(obbs))
        return flags

    while True:
        #print('while')
        max_probs = np.zeros(len(obbs))
        for i in range(len(predictions)):
            prediction = predictions[i]
            label = int(labels[i])
            if visited[i] == 0:
                max_probs[i] = prediction[label]
        index = np.where(max_probs == np.max(max_probs))[0]
        index_one = -1
        for i in range(len(index)):
            if visited[index[i]] == 0:
                index_one = index[i]
                break
        for i in range(len(predictions)):
            if i == index_one:
                flags[i] = 1
                visited[i] = 1
                continue
            if visited[i] == 1:
                continue
            if labels[i] != labels[index_one]:
                continue
            #iou = computeObbIou(obbs[index_one],obbs[i],insidePoints[index_one],insidePoints[i])
            iou = computePtsIou(getPTS(pts_dirs[index_one]),getPTS(pts_dirs[i]),obbs[index_one],obbs[i])
            IOU = 0.5
            if iou > IOU:
                flags[i] = 0
                visited[i] = 1
        stopFlag = 1
        num = 0
        for i in range(len(predictions)):
            if visited[i] == 0:
                stopFlag = 0
                num += 1
        if stopFlag:
            break
    return flags

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

def pickle2predbox(pickle_dir,box_dir,merge_leaf_pts_dir,internal_pts_dir):
    fo = open(pickle_dir, 'rb+')
    data = pickle.load(fo,encoding = 'ios-8859-1')
    gts = data['gts']
    predictions = data['predictions']
    obbs = data['obbs']
    ids = data['ids']
    labels = data['labels']
    count = data['count']

    pts_dirs = []
    for i in range(count):
        if os.path.exists(os.path.join(merge_leaf_pts_dir,'objects_'+str(ids[i])+'.pts')):
            pts_dirs.append(os.path.join(merge_leaf_pts_dir,'objects_'+str(ids[i])+'.pts'))
        elif os.path.exists(os.path.join(internal_pts_dir,'objects_'+str(ids[i])+'.pts')):
            pts_dirs.append(os.path.join(internal_pts_dir,'objects_'+str(ids[i])+'.pts'))
    flags = NMS_on_points(predictions,obbs,pts_dirs,labels)
    print('node after NMS', int(flags.sum()))
    fw = open(box_dir,'w')
    correct = wrong = 0
    count1 = 0
    for i in range(count):
        if flags[i] == 0:
            continue
        prediction = predictions[i]
        segment_id = ids[i]
        pred_array = prediction[0:40]
        label = np.where(pred_array == np.max(pred_array))[0]
        obb = obbs[i]
        if label[0] == gts[i]:
            correct += 1
        else:
            wrong += 1
        fw.write('%f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n'\
        %(obb[0], obb[1], obb[2], obb[3], obb[4], obb[5], obb[6], obb[7], gts[i],\
        pred_array[0], pred_array[1], pred_array[2], pred_array[3], pred_array[4],
        pred_array[5], pred_array[6], pred_array[7], pred_array[8], pred_array[9],
        pred_array[10], pred_array[11], pred_array[12], pred_array[13], pred_array[14],
        pred_array[15], pred_array[16], pred_array[17], pred_array[18], pred_array[19],
        pred_array[20], pred_array[21], pred_array[22], pred_array[23], pred_array[24],
        pred_array[25], pred_array[26], pred_array[27], pred_array[28], pred_array[29],
        pred_array[30], pred_array[31], pred_array[32], pred_array[33], pred_array[34],
        pred_array[35], pred_array[36], pred_array[37], pred_array[38], pred_array[39], segment_id))
        count1 += 1
    

config = util.get_args()
dataset = config.dataset
if dataset == 'S3DIS':
    categories = [6,8,9,14] #chair,board,table,sofa
elif dataset == 'Matterport':
    categories = [3,5,7,8,11,15,18,22,25,28]
elif dataset == 'ScanNet':
    categories = [3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
    
g_path = config.g_path
config.cuda = not config.no_cuda
if config.gpu<0 and config.cuda:
    config.gpu = 0
if config.cuda:
    torch.cuda.set_device(config.gpu)
pretrained_model = config.pretrained_model
encoder_decoder = torch.load(pretrained_model)
if config.cuda:
    encoder_decoder.cuda(config.gpu)

grass_data = GRASSDatasetTest(config.data_path)
def my_collate(batch):
    return batch
test_iter = torch.utils.data.DataLoader(grass_data, batch_size=1, shuffle=False, collate_fn=my_collate)

for batch_idx, batch in enumerate(test_iter):
    print(os.path.join(g_path,batch[0].scene_dir.split('/')[-1],'data','files','pred.pickle'))
    pred_pickle_path = os.path.join(g_path,batch[0].scene_dir.split('/')[-1],'data','files','pred.pickle')
    gts, predictions, obbs, labels, ids, count = grassmodel.encode_decode_structure_eval(encoder_decoder, batch[0])
    mdict={'gts': gts, 'predictions': predictions, 'obbs': obbs, 'labels': labels,'ids':ids, 'count': count}
    with open(pred_pickle_path, 'wb+') as f:
        pickle.dump(mdict,f)
    pred_box_path = os.path.join(g_path,batch[0].scene_dir.split('/')[-1],'data','files','pred_box_vdrae_for_eval.txt')
    merge_leaf_pts_dir = os.path.join(g_path,batch[0].scene_dir.split('/')[-1],'data','merge_leaf_pts')
    internal_pts_dir = os.path.join(g_path,batch[0].scene_dir.split('/')[-1],'data','internal_pts')
    pickle2predbox(pred_pickle_path,pred_box_path,merge_leaf_pts_dir,internal_pts_dir)