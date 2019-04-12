import os
from multiprocessing import Pool

dataset = 'Matterport'
data_path = '/home/net663/Downloads/yifeis/github/data'
pretrained_pointcnn_path = '../../pretrained_model/pointcnn/Matterport/iter-191050'
num_cpu = 30
g_path = os.path.join(data_path,dataset,'data_release')
g_gt_path = os.path.join(data_path,dataset,'gt')
train_scene_list_path = os.path.join(data_path,dataset,'task','train.txt')
test_scene_list_path = os.path.join(data_path,dataset,'task','test.txt')
val_scene_list_path = os.path.join(data_path,dataset,'task','val.txt')

print('preprocess_leaf')
cmdList = []
scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    cmd = 'python3 preprocess_leaf.py --g_path %s'%(scene_dir)
    #os.system('%s' %cmd)
    cmdList.append(cmd)
pool = Pool(num_cpu)
pool.map(os.system, cmdList)
pool.close()
pool.join()

print('preprocess_merge_leaf')
cmdList = []
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    cmd = 'python3 preprocess_merge_leaf.py --g_path %s'%(scene_dir)
    #os.system('%s' %cmd)
    cmdList.append(cmd)
pool = Pool(num_cpu)
pool.map(os.system, cmdList)
pool.close()
pool.join()

print('normalized cut')
import matlab
import matlab.engine
os.chdir('./Ncut_9')
eng = matlab.engine.start_matlab()
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    affinity_matrix_dir = os.path.join(scene_dir,'data','files','merge_segment_affinity_matrix.txt')
    hier_dir = os.path.join(scene_dir,'data','files','hier_ncut.txt')
    if not os.path.exists(affinity_matrix_dir):
        continue
    if len(open(affinity_matrix_dir,'r').readlines()) <= 1:
        continue
    eng.demoMatrix(affinity_matrix_dir,hier_dir)
os.chdir('..')

print('preprocess_all')
cmdList = []
count_in = 0
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    cmd = 'python2 preprocess_all.py --g_path %s --g_gt_path %s'%(scene_dir, gt_path)
    #os.system('%s' %cmd)
    cmdList.append(cmd)
pool = Pool(num_cpu)
pool.map(os.system, cmdList)
pool.close()
pool.join()


print('gen_txt')
output_path = dataset+'_all_node_pointcnn_input.txt'

if os.path.exists(output_path):
    os.remove(output_path)
scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    cmd = 'python3 gen_txt.py --g_path %s --output_path %s'%(scene_dir, output_path)
    os.system('%s' %cmd)
    if os.path.exists(os.path.join(scene_dir,'data','files','pred_box_all_node.txt')):
        os.remove(os.path.join(scene_dir,'data','files','pred_box_all_node.txt'))

print('feature extraction')
cmd = 'python3 ./pointcnn/gen_node_prob_feature.py -t %s -s ./pointcnn -m pointcnn_cls -x example_x3_l4 -l %s --gpu 0'%(output_path, pretrained_pointcnn_path)
os.system('%s' %cmd)

print('gen_grass_data')
pickle_path = './processed_data_train'
cmd = 'python3 ./gen_grass_data.py --g_path %s --pickle_path %s --scene_list_path %s'%(g_path, pickle_path, train_scene_list_path)
os.system('%s' %cmd)

pickle_path = './processed_data_test'
cmd = 'python3 ./gen_grass_data.py --g_path %s --pickle_path %s --scene_list_path %s'%(g_path, pickle_path, test_scene_list_path)
os.system('%s' %cmd)

pickle_path = './processed_data_val'
cmd = 'python3 ./gen_grass_data.py --g_path %s --pickle_path %s --scene_list_path %s'%(g_path, pickle_path, val_scene_list_path)
os.system('%s' %cmd)
print('finished')
