import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import torchvision
import torch.nn as nn
import math
import scipy.io as sio
import scipy.misc as smi

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(6, 256),
            nn.Sigmoid(),
            
            nn.Linear(256, 256),
            nn.Sigmoid(),
            
            nn.Linear(256, 256),
            nn.Sigmoid(),
            
            nn.Linear(256, 2),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256,momentum=0.8),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256,momentum=0.8),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256,momentum=0.8),
           
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512+256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512,momentum=0.8),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512,momentum=0.8),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512,momentum=0.8),

            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, feature):
        output = self.fc1(feature)
        return output
