import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import random

import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, (3,3), 1)
        self.conv2 = nn.Conv2d(6, 32, (3,3), 1)
        self.conv3 = nn.Conv2d(32, 16, (3,3), 1)

        random_sample = torch.randn((1,1,128,128))

        self.flattened_number = self.get_flattened_number(random_sample)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.flattened_number, 64)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(64, 16)        
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(16, 2)

    def get_flattened_number(self, X):
        X = F.max_pool2d(torch.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(torch.relu(self.conv2(X)), (2, 2))
        return X.shape[1] * X.shape[2] * X.shape[3]

    def forward(self, X):
        X = F.max_pool2d(torch.relu(self.conv1(X)), (2, 2))
        X = self.droupout2d1(X)
        X = F.max_pool2d(torch.relu(self.conv2(X)), (2, 2))
        X = self.droupout2d2(X)        
        X = F.max_pool2d(torch.relu(self.conv3(X)), (2, 2))
        X = self.droupout2d2(X)
        X = X.view(-1, self.flattened_number)
        X = self.dropout1(X)
        X = torch.relu(self.fc1(X))
        X = self.dropout2(X)
        X = torch.relu(self.fc2(X))        
        X = self.dropout3(X)
        X = torch.relu(self.fc3(X))
        return X
