import os
from multiprocessing import Pool 

g_path = '/home/net663/Downloads/yifeis/S3DIS/data_release'


pretrained_net_path = './models/snapshots_2019-03-11_14-55-39_Area_6/encoder_decoder_model_epoch_90_loss_3.4855.pkl'
cmd = 'python3 test.py  --g_path %s --pretrained_model %s --gpu 1 --room_type %s'%(g_path, pretrained_net_path, 'Area_1')
os.system('%s' %cmd)

pretrained_net_path = './models/snapshots_2019-03-11_14-54-18_Area_1/encoder_decoder_model_epoch_90_loss_129.7515.pkl'
cmd = 'python3 test.py  --g_path %s --pretrained_model %s --gpu 1 --room_type %s'%(g_path, pretrained_net_path, 'Area_2')
os.system('%s' %cmd)

pretrained_net_path = './models/snapshots_2019-03-11_14-54-50_Area_2/encoder_decoder_model_epoch_60_loss_1126.3905.pkl'
cmd = 'python3 test.py  --g_path %s --pretrained_model %s --gpu 1 --room_type %s'%(g_path, pretrained_net_path, 'Area_3')
os.system('%s' %cmd)

pretrained_net_path = './models/snapshots_2019-03-11_14-55-00_Area_3/encoder_decoder_model_epoch_60_loss_185.9370.pkl'
cmd = 'python3 test.py  --g_path %s --pretrained_model %s --gpu 1 --room_type %s'%(g_path, pretrained_net_path, 'Area_4')
os.system('%s' %cmd)

pretrained_net_path = './models/snapshots_2019-03-11_14-55-16_Area_4/encoder_decoder_model_epoch_90_loss_1.0223.pkl'
cmd = 'python3 test.py  --g_path %s --pretrained_model %s --gpu 1 --room_type %s'%(g_path, pretrained_net_path, 'Area_5')
os.system('%s' %cmd)

pretrained_net_path = './models/snapshots_2019-03-11_14-55-29_Area_5/encoder_decoder_model_epoch_90_loss_32.5628.pkl'
cmd = 'python3 test.py  --g_path %s --pretrained_model %s --gpu 1 --room_type %s'%(g_path, pretrained_net_path, 'Area_6')
os.system('%s' %cmd)

IOUs = [0.3,0.4,0.5,0.6,0.7]
categories = [6,8,9,14]
for IOU in IOUs:
    cmdList = []
    for category in categories:
        cmd = 'python3 evaluation.py --ap_category %d --IOU %f'%(category,IOU)
        print(cmd)
        cmdList.append(cmd)
    pool = Pool(16)
    pool.map(os.system, cmdList)
    pool.close()
    pool.join()