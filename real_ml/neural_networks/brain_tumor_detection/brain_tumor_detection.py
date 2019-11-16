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

from brain_tumor_detector import Net

NUM_CLASSES = 2

training_data = np.load('training_data.npy', allow_pickle=True)
validation_data = np.load('validation_data.npy', allow_pickle=True)

print('first example--------------------------------------------')

sample = training_data[0]
img = sample[0]
label = sample[1]
print('label',label)
plt.imshow(img)
plt.show()

img = torch.from_numpy(img)
img = img.view(1,1,128,128)
print('shape',img.shape)
print('first example--------------------------------------------')

net = Net()

# Split training data into train and validation sets

VALIDATION_PERCENTAGE = 0.3
NUM_VALIDATION = int(len(training_data) * VALIDATION_PERCENTAGE)

training_set = training_data
validation_set = validation_data

print('training size', len(training_set))
print('validation size', len(validation_set))

# Define optimizer and loss function

optimizer = optim.Adam(net.parameters(), lr=0.00001)
loss_function = nn.CrossEntropyLoss()

PATH = 'net.pth'
NUM_EPOCHS = 200
BATCH_SIZE = 128

print('now everybody in the 202, wave ur hands up and down cuz fat joe is thru')
print('Training Time------------------------------------------------')
print('')

for epoch in range(NUM_EPOCHS):

    np.random.shuffle(training_set)
    np.random.shuffle(validation_set)

    num_correct = 0
    running_loss = 0

    for i in range(len(training_set)):

        sample = training_set[i]
        label = sample[1]

        img = sample[0]
        img = torch.from_numpy(img)

        print('label', torch.tensor(label).view(-1, NUM_CLASSES))
        plt.imshow(img)
        plt.show()

        img = img.view(1,1,128,128)

        target = torch.tensor(label).view(-1, NUM_CLASSES)
        output = net(img).view(-1, NUM_CLASSES)

        print('output', output)

        loss = loss_function(target, output)
        loss.backward()
        running_loss += loss.item()

        if label == 1 and output.item() >= 0.5:
            num_correct += 1
        elif label == 0 and output.item() < 0.5:
            num_correct += 1

        # Then we take a step with our optimizer
        if i % BATCH_SIZE == 0:
            optimizer.step()
            net.zero_grad()
            torch.save(net.state_dict(), PATH)

    print('EPOCH:', epoch, 'loss:', running_loss, 'accuracy:', num_correct/len(training_set))

    num_correct = 0
    running_loss = 0
    for i in range(len(validation_set)):

        with torch.no_grad():
            sample = validation_set[i]
            img = sample[0]
            label = sample[1]
            img = torch.from_numpy(img)

            # print(label)
            # plt.imshow(img)
            # plt.show()

            img = img.view(1,1,128,128)

            target = torch.tensor(label).view(-1,1).float()
            output = net(img)

            loss = loss_function(target, output)
            running_loss += loss.item()

            if label == 1 and output.item() >= 0.5:
                num_correct += 1
            elif label == 0 and output.item() < 0.5:
                num_correct += 1

    print('EPOCH:', epoch,'loss:', running_loss, 'accuracy:', num_correct/len(training_set), '<-Validation Set')
    print('----------------------------------------------------------------------------------------------------')