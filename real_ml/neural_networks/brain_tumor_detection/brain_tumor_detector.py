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

        self.conv1 = nn.Conv2d(1, 4, (3,3), 1)
        self.dropout2d_1 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(4, 32, (3,3), 1)
        self.dropout2d_2 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(32, 64, (3,3), 1)

        random_sample = torch.randn((1,1,128,128))

        self.flattened_number = self.get_flattened_number(random_sample)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.flattened_number, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 32)        
        self.fc3 = nn.Linear(32, 16)        
        self.fc4 = nn.Linear(16, 2)

    def get_flattened_number(self, X):
        X = F.max_pool2d(torch.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(torch.relu(self.conv2(X)), (2, 2))
        X = F.max_pool2d(torch.relu(self.conv3(X)), (2, 2))
        return X.shape[1] * X.shape[2] * X.shape[3]

    def forward(self, X):
        X = F.max_pool2d(torch.relu(self.conv1(X)), (2, 2))
        X = self.dropout2d_1(X)
        X = F.max_pool2d(torch.relu(self.conv2(X)), (2, 2))
        X = self.dropout2d_1(X)
        X = F.max_pool2d(torch.relu(self.conv3(X)), (2, 2))
        X = X.view(-1, self.flattened_number)
        X = self.dropout1(X)
        X = torch.relu(self.fc1(X))
        X = self.dropout2(X)
        X = torch.relu(self.fc2(X))        
        X = torch.relu(self.fc3(X))      
        X = torch.sigmoid(self.fc4(X))
        return X
