
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

# TODO NEED TO SHUFFLE THE DXATA FROM THE TRAIN-DATASET TORCH THING BECAUSE VALIDATION SET IS ONLY COMPOSED OF 1 LABEL
class ImageProcessingPipeline():

    VALIDATION_PERCENTAGE = 0.2
    DATA_PATH = os.path.join(os.getcwd(), 'dataset/')

    training_data = []
    validation_data = []

    class_count = {
        0: 0,
        1: 0
    }
    class_count_validation = {
        0: 0,
        1: 0
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_dataset = torchvision.datasets.ImageFolder(root=self.DATA_PATH, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=16, num_workers=0, shuffle=True)
        self.classes = self.train_dataset.classes
        self.num_classes = self.classes.__len__()
        self.VALIDATION_SIZE = int(self.VALIDATION_PERCENTAGE * len(self.train_dataset))

    def create_training_set(self):

        temp_train_dataset = []

        for i in range(int(len(self.train_dataset))):
            
            sample = self.train_dataset[i]
            label = sample[1]

            if label == 0:
                for _ in range(3):
                    temp_train_dataset.append(sample)
            else:
                for _ in range(2):
                    temp_train_dataset.append(sample)

        np.random.shuffle(temp_train_dataset)

        for _ in range(10):
            for i in range(int(len(temp_train_dataset))):
                
                if i < len(temp_train_dataset) - self.VALIDATION_SIZE:

                    sample = temp_train_dataset[i]

                    image = sample[0]
                    index = sample[1]

                    image = image.permute(1,2,0)
                    image = image.numpy()
                    image = cv2.resize(image, (128,128))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                    should_flip = random.randint(0, 1)

                    if should_flip == 1:
                        image = np.fliplr(image)

                    angle = random.randint(-35, 35)
                    scale = random.uniform(0.9, 1.0)
                    w = image.shape[1]
                    h = image.shape[0]
                    image = cv2.warpAffine(image, cv2.getRotationMatrix2D((w/2,h/2), angle, scale), dsize=(128,128))

                    self.class_count[index] += 1
                    self.training_data.append([image, index])

                else:

                    sample = temp_train_dataset[i]
                    image = sample[0]
                    image = image.permute(1,2,0)
                    image = image.numpy()
                    image = cv2.resize(image, (128,128))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                    index = sample[1]
                    self.validation_data.append([image, index])
                    self.class_count_validation[index] += 1

            np.random.shuffle(self.training_data)
            np.random.shuffle(self.validation_data)
            np.save('training_data.npy', self.training_data)
            np.save('validation_data.npy', self.validation_data)

        print('training_set', self.class_count)
        print('validation_set', self.class_count_validation)

c = ImageProcessingPipeline()
c.create_training_set()