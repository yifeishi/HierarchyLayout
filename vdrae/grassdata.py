import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import numpy as np
import pickle
import util
import numpy as np
import os

config = util.get_args()

class Tree(object):
    class NodeType(Enum):
        Leaf = 0
        Internal = 1
        
    class Node(object):
        def __init__(self, box=None, pc_feature=None, left=None, right=None, node_type=None, sym=None, index=None, category=None, category_pred=None, reg=None):
            box_np = np.zeros((1,8))
            reg_np = np.zeros((1,8))
            for i in range(8):
                box_np[:,i] = box[i]
                reg_np[:,i] = reg[i]
            box = torch.from_numpy(box_np).float()
            reg = torch.from_numpy(reg_np).float()
            pc_feature = torch.from_numpy(pc_feature).float()
            
            self.box = box
            self.reg = reg
            self.pc_feature = pc_feature
            self.encode_feature = None
            self.decode_feature = None
            self.left = left
            self.right = right
            self.node_type = node_type
            self.label = torch.LongTensor([self.node_type.value])
            self.index = index
            self.category = torch.LongTensor([category])
            self.category_pred =  torch.FloatTensor([category_pred])
            
        def is_leaf(self):
            return self.node_type == Tree.NodeType.Leaf and self.box is not None

        def is_internal(self):
            return self.node_type == Tree.NodeType.Internal

    def LoadFeature(self,file):
        feature = np.zeros((1,256))
        if file == 'empty':
            print('empty')
            return feature
        f=open(file)
        line = f.readline()
        L = line.split()
        for i in range(len(L)):
            feature[:,i] = float(L[i])
        return feature

    def __init__(self, scene_dir, feature_dir, boxes, boxes_reg, categories, categories_pred, mapFather, mapChild1, mapChild2, isLeaf, index):
        np.set_printoptions(threshold=np.inf)
        validNode = np.zeros(boxes.shape[0])
        mapNode = {}
        while 1:
            for id in range(boxes.shape[0]):
                if isLeaf[id] == 'True' and validNode[id] == 0.0:
                    pc_feature = self.LoadFeature(feature_dir[id])
                    node = Tree.Node(node_type=Tree.NodeType.Leaf, box = boxes[id,:], reg=boxes_reg[id,:], pc_feature = pc_feature, index=id, category=categories[id], category_pred=categories_pred[id])
                    mapNode[id] = node
                    validNode[id] = 1
                elif isLeaf[id] == 'False' and validNode[id] == 0.0:
                    child1 = mapChild1[id]
                    child2 = mapChild2[id]
                    if mapNode.get(child1, 'False') != 'False' and mapNode.get(child2, 'False') != 'False':
                        pc_feature = self.LoadFeature(feature_dir[id])
                        node = Tree.Node(node_type=Tree.NodeType.Internal, left=mapNode[child1], right=mapNode[child2], box = boxes[id,:], reg=boxes_reg[id,:], pc_feature = pc_feature, index=id, category=categories[id], category_pred=categories_pred[id])
                        mapNode[id] = node
                        validNode[id] = 1
            stopFlag = True
            count = 0
            for id in range(boxes.shape[0]):
                if validNode[id] == 0:
                    stopFlag = False
                    count = count + 1
            if stopFlag:
                break

        root = []
        for id in range(boxes.shape[0]):
            if mapFather.get(id, 'False') == 'False':
                root.append(id)
    
        self.root = mapNode[root[0]]
        self.scene_dir = scene_dir
        self.index = index

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

class GRASSDataset(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.trees = []
        pkl_names = os.listdir(self.dir)
        new_pkl_names = []
        for pkl_name in pkl_names:
            with open(os.path.join(self.dir,pkl_name), 'rb') as f:
                data = pickle.load(f)
            scene_dir = data['scene_dir']
            new_pkl_names.append(pkl_name)
        self.pkl_names = new_pkl_names
        
    def __getitem__(self, index):
        with open(os.path.join(self.dir,self.pkl_names[index]), 'rb') as f:
            data = pickle.load(f)

        scene_dir = data['scene_dir']
        feature_dir = data['feature_dir']
        boxes = data['boxes']
        boxes_reg = data['boxes_reg']
        categories = data['category']
        categories_pred = data['category_pred']
            
        mapFather = data['mapFather']
        mapChild1 = data['mapChild1']
        mapChild2 = data['mapChild2']
        isLeaf = data['isLeaf']
        tree = Tree(scene_dir, feature_dir, boxes, boxes_reg, categories,categories_pred, mapFather, mapChild1, mapChild2, isLeaf, index)
        return tree 

    def __len__(self):
        return len(self.pkl_names)


class GRASSDatasetTest(data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.trees = []
        pkl_names = os.listdir(self.dir)
        new_pkl_names = []
        for pkl_name in pkl_names:
            with open(os.path.join(self.dir,pkl_name), 'rb') as f:
                data = pickle.load(f)
            scene_dir = data['scene_dir']
            new_pkl_names.append(pkl_name)
        self.pkl_names = new_pkl_names
        
    def __getitem__(self, index):
        with open(os.path.join(self.dir,self.pkl_names[index]), 'rb') as f:
            data = pickle.load(f)

        scene_dir = data['scene_dir']
        feature_dir = data['feature_dir']
        boxes = data['boxes']
        boxes_reg = data['boxes_reg']
        categories = data['category']
        categories_pred = data['category_pred']
            
        mapFather = data['mapFather']
        mapChild1 = data['mapChild1']
        mapChild2 = data['mapChild2']
        isLeaf = data['isLeaf']
        tree = Tree(scene_dir, feature_dir, boxes, boxes_reg, categories, categories_pred, mapFather, mapChild1, mapChild2, isLeaf, index)
        return tree 

    def __len__(self):
        return len(self.pkl_names)


