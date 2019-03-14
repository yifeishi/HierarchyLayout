import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import torchvision
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import scipy.misc as smi

import scipy.io as sio
import scipy.misc as smi


class AffinityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform
        
        f = open(self.landmarks_frame,'r')
        lines = f.readlines()
        num = len(lines)
        self.labels = np.zeros(num)
        self.features = np.zeros((num,6))
        
        f = open(self.landmarks_frame,'r')
        count = 0
        while 1:
            line = f.readline()
            if not line:
                break
            L = line.split()
            #print('L',L)
            if len(L) != 9:
                continue
            self.labels[count] = int(L[1])
	    continueFlag = False
	    for i in range(0,6):
		if len(L[i+3].split('.')) > 2:
                    continueFlag = True
                    break
		
	        self.features[count,i] = float(L[i+3])
            if continueFlag == True:
		continue
            count = count + 1
            if count > 2000000:
                break
        self.num = count
        print('finished reading')
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        #print(idx)
        feature = self.features[idx,:]
        label = self.labels[idx]
        #print('feature',feature,'label',label)
        sample = {'feature': feature, 'label': label}
        
        return sample
