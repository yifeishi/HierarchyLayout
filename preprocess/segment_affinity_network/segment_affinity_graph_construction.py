import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset_affinity import *
from network_affinity import *
import util
import util_affinity
from open3d import *

def getPTS(pts_file,colorPara=1):
    fpts = open(pts_file)
    count = 0
    while 1:
        line = fpts.readline()
        if not line:
            break
        count = count + 1
    if count==0:
        return np.zeros(6)
    points = np.zeros((count,6))
    count = 0
    fpts = open(pts_file)
    while 1:
        line = fpts.readline()
        if not line:
            break
        L = line.split(' ')
        points[count,0] = float(L[0])
        points[count,1] = float(L[1])
        points[count,2] = float(L[2])
        points[count,3] = float(L[3])/colorPara
        points[count,4] = float(L[4])/colorPara
        points[count,5] = float(L[5])/colorPara
        count = count + 1
    return points

def PTS2Normal(g_path,pts_name):
    #print('PTS2Normal',pts_name)
    pts_dir = os.path.join(g_path,pts_name)
    pcd_dir = os.path.join(g_path,'temp.pcd')
    fo = open(pts_dir,'r')
    lines = fo.readlines()
    pts_num = len(lines)
    fo.close()
    fw = open(pcd_dir,'w')
    fw.write('VERSION .7\n')
    fw.write('FIELDS x y z rgb\n')
    fw.write('SIZE 8 8 8 8\n')
    fw.write('TYPE F F F F\n')
    fw.write('COUNT 1 1 1 1\n')
    fw.write('WIDTH %d\n'%pts_num)
    fw.write('HEIGHT 1\n')
    fw.write('VIEWPOINT 0 0 0 1 0 0 0\n')
    fw.write('POINTS %d\n'%pts_num)
    fw.write('DATA ascii\n')
    fo = open(pts_dir,'r')
    while 1:
        line = fo.readline()
        if not line:
            break
        fw.write('%s'%line)
    fo.close()
    fw.close()

    pcd = read_point_cloud(pcd_dir)
    estimate_normals(pcd, search_param = KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
    #print(pcd.normals)
    normal = np.zeros(3)
    for i in range(pts_num):
        #print(i,pcd.normals[i])
        if pcd.normals[i][2] < 0 and 0:
            pcd.normals[i][0] = -pcd.normals[i][0]
            pcd.normals[i][1] = -pcd.normals[i][1]
            pcd.normals[i][2] = -pcd.normals[i][2]
        normal[0] += pcd.normals[i][0]
        normal[1] += pcd.normals[i][1]
        normal[2] += pcd.normals[i][2]
    normal[0] /= pts_num
    normal[1] /= pts_num
    normal[2] /= pts_num
    length = math.sqrt(math.pow(normal[0],2) + math.pow(normal[1],2) + math.pow(normal[2],2))
    normal[0] /= length
    normal[1] /= length
    normal[2] /= length

    #print(length,normal)
    #xx=yy
    return normal

def PTS2ColorDTB(pts):
    colorDTB = np.zeros(3)
    average_r = 0
    average_g = 0
    average_b = 0
    for i in range(pts.shape[0]):
       #print(pts[i,:])
       value1 = pts[i,3]*255/8
       value2 = pts[i,4]*255/8
       value3 = pts[i,5]*255/8
       average_r += value1
       average_g += value2
       average_b += value3
    average_r /= pts.shape[0]
    average_g /= pts.shape[0]
    average_b /= pts.shape[0]
    colorDTB[0],colorDTB[1],colorDTB[2] = average_r,average_g,average_b
   # print(colorDTB)
    return colorDTB

def PTS2Center(pts):
    center = np.zeros(3)
    for i in range(pts.shape[0]):
        center[0] += pts[i,0]
        center[1] += pts[i,1]
        center[2] += pts[i,2]
    center[0] /= pts.shape[0]
    center[1] /= pts.shape[0]
    center[2] /= pts.shape[0]
    #print(center)
    return center

def computeDiffDihedralAngle(normal1, normal2):
    feature = normal1[0]*normal2[0] + normal1[1]*normal2[1] + normal1[2]*normal2[2]
    feature = abs(feature)
    #print(normal1,normal2,feature)
    return feature

def computeDiffPatchSize(size1,size2):
    feature = float(abs(size1 - size2))/1000
    return feature

def computeDiffColor(color1,color2):
    feature = np.zeros(3)
    feature[0] = abs(color1[0] - color2[0])
    feature[1] = abs(color1[1] - color2[1])
    feature[2] = abs(color1[2] - color2[2])
    return feature

def computeCenDis(center1,center2):
    x = math.pow(center1[0] - center2[0],2)
    y = math.pow(center1[1] - center2[1],2)
    z = math.pow(center1[2] - center2[2],2)
    feature = math.sqrt(x+y+z)
    return feature


# load segment graph (10min)
# for value == 1: compute the feautre(20min), compute the affinity(20min)
# write affinity graph(10min)

#config = util.get_args()
config_affinity = util_affinity.get_args()
g_path = config_affinity.g_path
g_box_path = config_affinity.g_box_path

adj_matrix_dir = os.path.join(g_path,'segment_adjacent_matrix.txt')
affinity_matrix_dir = os.path.join(g_path,'segment_affinity_matrix.txt')

fam = open(adj_matrix_dir,'r')
lines = fam.readlines()
line_num = len(lines)
adjMatrix = np.zeros((line_num,line_num))
affMatrix = np.zeros((line_num,line_num))
L2Matrix = np.zeros((line_num,line_num))
fam = open(adj_matrix_dir,'r')
count = 0
while 1:
    line = fam.readline()
    if not line:
        break
    L = line.split()
    #print('L',L)
    for i in range(line_num):
        adjMatrix[count,i] =  int(L[i])
    count += 1
#print('adjMatrix',adjMatrix)

model = torch.load('./model177200.pkl')
#model.cuda(config.gpu)

for i in range(adjMatrix.shape[0]):
    pts_dir1 = os.path.join(g_path,'objects_'+str(i)+'.pts')
    pts1 = getPTS(pts_dir1)
    if pts1.shape[0] < 10:
        continue
    center1 = PTS2Center(pts1)   
    for j in range(adjMatrix.shape[1]):
        #pts_dir1 = os.path.join(g_path,'objects_'+str(i)+'.pts')
        pts_dir2 = os.path.join(g_path,'objects_'+str(j)+'.pts')
        #pts1 = getPTS(pts_dir1)
        pts2 = getPTS(pts_dir2)
        if pts2.shape[0] < 10:
            continue
        #center1 = PTS2Center(pts1)
        center2 = PTS2Center(pts2)
        feature4 = computeCenDis(center1,center2)
        L2Matrix[i,j] = feature4
        if adjMatrix[i,j] == 0:
	    continue

	# compute feature
        normal1 = PTS2Normal(g_path,'objects_'+str(i)+'.pts')
	normal2 = PTS2Normal(g_path,'objects_'+str(j)+'.pts')
	colorDTB1 = PTS2ColorDTB(pts1)
	colorDTB2 = PTS2ColorDTB(pts2)
	feature1 = computeDiffDihedralAngle(normal1,normal2)
        feature2 = computeDiffPatchSize(pts1.shape[0],pts2.shape[0])
        feature3 = computeDiffColor(colorDTB1,colorDTB2)
	feature = np.zeros((1,6))
        feature[0,0] = feature1
	feature[0,1] = feature2
	feature[0,2] = feature3[0]
	feature[0,3] = feature3[1]
	feature[0,4] = feature3[2]
	feature[0,5] = feature4
	#feature = Variable(feature.cuda(config.gpu), requires_grad=True)
	feature = torch.from_numpy(feature)
	feature = Variable(feature.float().cuda(0),requires_grad=True)
	output = model(feature)
    	outputSoftmax = nn.Softmax()(output).double()
	#print(i,j,feature.data.cpu().numpy())
	#print(output.data.cpu().numpy()[0],outputSoftmax.data.cpu().numpy()[0])
	affMatrix[i,j] = outputSoftmax.data.cpu().numpy()[0][1]
        #L2Matrix[i,j] = feature4
	print(i,j,affMatrix[i,j],feature4)

output = open(os.path.join(g_path,'segment_affinity_matrix.txt'),'w')
for i in range(0,affMatrix.shape[0]):
    for j in range(0,affMatrix.shape[1]):
        output.write('%f '%(affMatrix[i,j]))
    output.write('\n')
output.close()

output = open(os.path.join(g_path,'segment_L2_matrix.txt'),'w')
for i in range(0,L2Matrix.shape[0]):
    for j in range(0,L2Matrix.shape[1]):
        output.write('%f '%(L2Matrix[i,j]))
    output.write('\n')
output.close()


