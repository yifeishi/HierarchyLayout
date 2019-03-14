import os
from multiprocessing import Pool 

g_path = '/home/net663/Downloads/yifeis/S3DIS/stanford_cluster/Stanford3dDataset_v1.2_Aligned_Version'
g_box_path = '/home/net663/Downloads/yifeis/S3DIS'

cmdList = []
house_names = os.listdir(g_path)
house_names.sort()
for house_name in house_names:
    house_seg_dir = os.path.join(g_path,house_name)
    house_box_dir = os.path.join(g_box_path,house_name)
    room_names = os.listdir(house_seg_dir)
    room_names.sort()
    for room_name in room_names:
        room_dir = os.path.join(house_seg_dir, room_name, 'data')
        box_dir = os.path.join(house_box_dir, room_name, 'Annotations')
        cmd = 'python segment_affinity_graph_construction.py --g_path %s --g_box_path %s'%(room_dir, box_dir)
        print(cmd)
        #os.system('%s' %cmd)
        #xx=yy
        cmdList.append(cmd)
pool = Pool(8)
pool.map(os.system, cmdList)
pool.close()
pool.join()

