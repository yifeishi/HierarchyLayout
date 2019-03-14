import os
from multiprocessing import Pool
g_path = '/home/net663/Downloads/yifeis/S3DIS/data_release'
model_path = 'iter-2572'
pkl_path = '/home/net663/Downloads/yifeis/S3DIS/region_feature'

cmdList = []
scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    print(scene_dir)
    cmd = 'python preprocess_leaf.py --g_path %s'%(scene_dir)
    cmdList.append(cmd)
pool = Pool(16)
pool.map(os.system, cmdList)
pool.close()
pool.join()

import matlab 
import matlab.engine
os.chdir('./Ncut_9')
eng = matlab.engine.start_matlab()
scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    print(scene_dir)
    affinity_matrix_dir = os.path.join(scene_dir,'data','files','segment_affinity_matrix.txt')
    hier_dir = os.path.join(scene_dir,'data','files','hier_ncut.txt')
    eng.demoMatrix(affinity_matrix_dir,hier_dir)
os.chdir('..')

cmdList = []
scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    cmd = 'python preprocess_all.py --g_path %s'%(scene_dir)
    cmdList.append(cmd)
pool = Pool(16)
pool.map(os.system, cmdList)
pool.close()
pool.join()

cmd = 'python ./pointcnn/get_pts_feature_list.py --g_path %s'%(g_path)
os.system('%s' %cmd)
cmd = 'python ./pointcnn/gen_feature.py -t ./pointcnn/pts_feature_list.txt -s ./pointcnn/point2feature_weight -m pointcnn_cls -x example_x3_l4 -l %s'%(model_path)
os.system('%s' %cmd)
pickle_path = '../pickle'
cmd = 'python ./gen_grass_data.py --g_path %s --pickle_path %s'%(g_path, pickle_path)
os.system('%s' %cmd)