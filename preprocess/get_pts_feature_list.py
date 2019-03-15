import os
from multiprocessing import Pool 

def get_FileSize(filePath):
    filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize
    return round(fsize,2)

config = util.get_args()
g_path = config.g_path
output = open('./pointcnn/pts_feature_list','w')

scene_names = os.listdir(g_path)
scene_names.sort()
for scene_name in scene_names:
    scene_dir = os.path.join(g_path,scene_name)
    leaf_dir = os.path.join(scene_dir,'data','leaf_pts')
    internal_dir = os.path.join(scene_dir,'data','internal_pts')

    object_names = os.listdir(leaf_dir)
    object_names.sort()
    for object_name in object_names:
        if len(object_name.split('.')) > 1 and object_name.split('.')[1] == 'pts':
            feature_name = object_name.split('.')[0]+'_pointcnn_feature.txt'
            feature_dir = os.path.join(leaf_dir,feature_name)
            object_dir = os.path.join(leaf_dir,object_name)
            if get_FileSize(object_dir) < 1000:
                continue
            output.write('%s %s\n' %(object_dir,feature_dir))
            
    object_names = os.listdir(internal_dir)
    object_names.sort()
    for object_name in object_names:
        if len(object_name.split('.')) > 1 and object_name.split('.')[1] == 'pts':
            feature_name = object_name.split('.')[0]+'_pointcnn_feature.txt'
            feature_dir = os.path.join(internal_dir,feature_name)
            object_dir = os.path.join(internal_dir,object_name)
            if get_FileSize(object_dir) < 1000:
                continue
            output.write('%s %s\n' %(object_dir,feature_dir))
            
output.close()
