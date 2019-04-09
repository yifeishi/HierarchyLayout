import os
from multiprocessing import Pool

dataset = 'Matterport'
data_path = '/home/net663/Downloads/yifeis/'
g_path = os.path.join(data_path,dataset,'data_release')
g_gt_path = os.path.join(data_path,dataset,'gt')
train_scene_list_path = os.path.join(data_path,dataset,'task','train.txt')
test_scene_list_path = os.path.join(data_path,dataset,'task','test.txt')
val_scene_list_path = os.path.join(data_path,dataset,'task','val.txt')

cmdList = []
scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    break
    scene_dir = os.path.join(g_path,scene_name)
    print(scene_dir)
    cmd = 'python3 preprocess_leaf.py --g_path %s'%(scene_dir)
    #os.system('%s' %cmd)
    cmdList.append(cmd)
pool = Pool(30)
pool.map(os.system, cmdList)
pool.close()
pool.join()

cmdList = []
for scene_name in scene_names:
    break
    scene_dir = os.path.join(g_path,scene_name)
    print(scene_dir)
    cmd = 'python3 preprocess_merge_leaf.py --g_path %s'%(scene_dir)
    #os.system('%s' %cmd)
    cmdList.append(cmd)
pool = Pool(30)
pool.map(os.system, cmdList)
pool.close()
pool.join()

import matlab
import matlab.engine
os.chdir('./Ncut_9')
eng = matlab.engine.start_matlab()
for scene_name in scene_names:
    break
    scene_dir = os.path.join(g_path,scene_name)
    print(scene_dir)
    affinity_matrix_dir = os.path.join(scene_dir,'data','files','merge_segment_affinity_matrix.txt')
    hier_dir = os.path.join(scene_dir,'data','files','hier_ncut.txt')
    if not os.path.exists(affinity_matrix_dir):
        continue
    if len(open(affinity_matrix_dir,'r').readlines()) <= 1:
        continue
    eng.demoMatrix(affinity_matrix_dir,hier_dir)
os.chdir('..')

cmdList = []
count_in = 0
for scene_name in scene_names:
    break
    scene_dir = os.path.join(g_path,scene_name)
    print(scene_dir)
    cmd = 'python preprocess_all.py --g_path %s'%(scene_dir)
    #os.system('%s' %cmd)
    cmdList.append(cmd)
pool = Pool(30)
pool.map(os.system, cmdList)
pool.close()
pool.join()


pickle_path = './processed_data_train'
cmd = 'python3 ./gen_grass_data.py --g_path %s --pickle_path %s --scene_list_path %s'%(g_path, pickle_path, train_scene_list_path)
os.system('%s' %cmd)

pickle_path = './processed_data_test'
cmd = 'python3 ./gen_grass_data.py --g_path %s --pickle_path %s --scene_list_path %s'%(g_path, pickle_path, test_scene_list_path)
os.system('%s' %cmd)

pickle_path = './processed_data_val'
cmd = 'python3 ./gen_grass_data.py --g_path %s --pickle_path %s --scene_list_path %s'%(g_path, pickle_path, val_scene_list_path)
os.system('%s' %cmd)