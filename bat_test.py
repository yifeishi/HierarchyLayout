import os
from multiprocessing import Pool 
import util

config = util.get_args()
g_path = config.g_path
pretrained_model = config.pretrained_model

cmd = 'python3 test.py  --g_path %s --pretrained_model %s --room_type %s'%(g_path, pretrained_model, 'Area_6')
os.system('%s' %cmd)

IOUs = [0.25,0.5]
categories = [6,8,9,14]
for IOU in IOUs:
    cmdList = []
    for category in categories:
        cmd = 'python3 evaluation.py --ap_category %d --IOU %f --room_type %s'%(category,IOU,'Area_6')
        cmdList.append(cmd)
    pool = Pool(4)
    pool.map(os.system, cmdList)
    pool.close()
    pool.join()