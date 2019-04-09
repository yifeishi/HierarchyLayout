import os

dataset = 'Matterport'
data_path = '/home/net663/Downloads/yifeis/'
g_path = os.path.join(data_path,dataset,'data_release')
g_gt_path = os.path.join(data_path,dataset,'gt')
test_scene_list_path = os.path.join(data_path,dataset,'task','test.txt')
pretrained_model = config.pretrained_model
cmd = 'python3 test_for_eval.py  --g_path %s --pretrained_model %s --data_path %s'%(g_path, pretrained_model,'processed_data_test')
os.system('%s' %cmd)

if dataset == 'S3DIS':
    categories = [6,8,9,14] #chair,board,table,sofa
elif dataset == 'Matterport':
    categories = [3,5,7,8,11,15,18,22,25,28]
elif dataset == 'ScanNet':
    categories = [3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
IOUs = [0.5]
for IOU in IOUs:
    cmdList = []
    for category in categories:
        cmd = 'python evaluation.py --g_path %s --g_gt_path %s --ap_category %d --IOU %f --scene_list_path %s'%(g_path,g_gt_path,category,IOU,test_scene_list_path)
        cmdList.append(cmd)
        #os.system('%s' %cmd)
pool = Pool(len(categories)*len(IOUs))
pool.map(os.system, cmdList)
pool.close()
pool.join()