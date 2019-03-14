import torch
import util
import grassmodel
from grassdata import GRASSDatasetTest
from grassmodel import GRASSEncoderDecoder
import os
import pickle
from util_obb import * 

def NMS(predictions,obbs,labels):
    visited = np.zeros(len(obbs))
    flags = np.ones(len(obbs))
    
    countLabelCorrect = 0
    for i in range(len(labels)):
        if labels[i] == 0 or labels[i] == 1:
            flags[i] = 0
            visited[i] = 1
        else:
            countLabelCorrect += 1
    if len(obbs) == 0 or countLabelCorrect == 0:
        return flags

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

def pickle2predbox(pickle_dir, box_dir):
    fo = open(pickle_dir, 'rb+')
    data = pickle.load(fo,encoding = 'ios-8859-1')
    gts = data['gts']
    predictions = data['predictions']
    obbs = data['obbs']
    ids = data['ids']
    labels = data['labels']
    count = data['count']

    flags = NMS(predictions,obbs,labels)
    fw = open(box_dir,'w')
    correct = wrong = 0
    for i in range(count):
        if flags[i] == 0:
            continue
        prediction = predictions[i]
        segment_id = ids[i]
        pred_array = prediction[0:20]
        label = np.where(pred_array == np.max(pred_array))[0]
        obb = obbs[i]
        if label == gts[i]:
            correct += 1
        else:
            wrong += 1
        fw.write('%f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n'\
        %(obb[0], obb[1], obb[2], obb[3], obb[4], obb[5], obb[6], obb[7], gts[i],\
        pred_array[0], pred_array[1], pred_array[2], pred_array[3], pred_array[4],
        pred_array[5], pred_array[6], pred_array[7], pred_array[8], pred_array[9],
        pred_array[10], pred_array[11], pred_array[12], pred_array[13], pred_array[14],
        pred_array[15], pred_array[16], pred_array[17], pred_array[18], pred_array[19], segment_id))

config = util.get_args()
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
    print(batch[0].scene_dir)
    pred_pickle_path = os.path.join(batch[0].scene_dir,'data','files','pred.pickle')
    gts, predictions, obbs, labels, ids, count = grassmodel.encode_decode_structure_eval(encoder_decoder, batch[0])
    mdict={'gts': gts, 'predictions': predictions, 'obbs': obbs, 'labels': labels,'ids':ids, 'count': count}
    with open(pred_pickle_path, 'wb+') as f:
        pickle.dump(mdict,f)
    pred_box_path = os.path.join(batch[0].scene_dir,'data','files','pred_box.txt')
    pickle2predbox(pred_pickle_path,pred_box_path)