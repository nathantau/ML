
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import random
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np

# TODO NEED TO SHUFFLE THE DXATA FROM THE TRAIN-DATASET TORCH THING BECAUSE VALIDATION SET IS ONLY COMPOSED OF 1 LABEL


class ImageProcessingPipeline():

    NUM_CLASSES = 2
    VALIDATION_PERCENTAGE = 0.25
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

        self.train_dataset = torchvision.datasets.ImageFolder(
            root=self.DATA_PATH, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=16, num_workers=0, shuffle=True)
        self.classes = self.train_dataset.classes
        self.num_classes = self.classes.__len__()

    def create_training_set(self):

        temp_train_dataset = []

        for i in range(int(len(self.train_dataset))):

            sample = self.train_dataset[i]
            label = sample[1]
            # label is either a 0 (doesn't have cancer) or a 1 (has cancer)
            # this step below is to convert to one-hot encoding

            new_sample = (sample[0], np.eye(self.NUM_CLASSES)[label])

            if label == 0:
                for _ in range(6):
                    temp_train_dataset.append(new_sample)
            else:
                for _ in range(4):
                    temp_train_dataset.append(new_sample)

        np.random.shuffle(temp_train_dataset)
        self.VALIDATION_SIZE = int(
            self.VALIDATION_PERCENTAGE * len(temp_train_dataset))

        print('total size:', len(temp_train_dataset))
        print('train-size:', len(temp_train_dataset) - self.VALIDATION_SIZE)
        print('validation size:', self.VALIDATION_SIZE)

        for _ in range(10):
            for i in range(int(len(temp_train_dataset))):

                if i < len(temp_train_dataset) - self.VALIDATION_SIZE:
                    sample = temp_train_dataset[i]

                    image = sample[0]
                    image = self.prepare_image(image)
                    image = self.augment_image(image) # Data augmentation happens over here

                    target = sample[1]
                    target = torch.from_numpy(target)
                    label = torch.argmax(target).item()

                    self.class_count[label] += 1
                    self.training_data.append([image, target])

                else:
                    sample = temp_train_dataset[i]

                    image = sample[0]
                    image = self.prepare_image(image)

                    target = sample[1]
                    target = torch.from_numpy(target)
                    label = torch.argmax(target).item()

                    self.class_count_validation[label] += 1
                    self.validation_data.append([image, target])

            np.random.shuffle(self.training_data)
            np.random.shuffle(self.validation_data)
            np.save('training_data.npy', self.training_data)
            np.save('validation_data.npy', self.validation_data)

        print('training_set', self.class_count)
        print('validation_set', self.class_count_validation)

    def prepare_image(self, image):
        image = image.permute(1, 2, 0)
        image = image.numpy()
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def augment_image(self, image):
        should_flip = random.randint(0, 1)
        if should_flip == 1:
            image = np.fliplr(image)

        angle = random.randint(-15, 15)
        scale = random.uniform(0.9, 1.0)
        w = image.shape[1]
        h = image.shape[0]
        image = cv2.warpAffine(image, cv2.getRotationMatrix2D(
            (w/2, h/2), angle, scale), dsize=(128, 128))

        return image

    def augment_image2(self, image):
        translate = random.randint(0, 1)
        if translate == 1:
            image = self.shift_right(image)
        else:
            image = self.shift_left(image)

        translate = random.randint(0, 1)
        if translate == 1:
            image = self.shift_up(image)
        else:
            image = self.shift_down(image)

        should_flip = random.randint(0, 1)
        if should_flip == 1:
            image = np.fliplr(image)

        image = self.add_noise(image)

        # angle = random.randint(-35, 35)
        angle = 0
        scale = random.uniform(0.9, 1.0)
        w = image.shape[1]
        h = image.shape[0]
        image = cv2.warpAffine(image, cv2.getRotationMatrix2D(
            (w/2, h/2), angle, scale), dsize=(128, 128))

        return image

    def shift_right(self, image):
        WIDTH = image.shape[1]
        HEIGHT = image.shape[0]

        for i in range(HEIGHT, 1, -1):
            for j in range(WIDTH):
                if (i < HEIGHT-10):
                    image[j][i] = image[j][i-10]
                elif (i < HEIGHT-1):
                    image[j][i] = 0

        return image    
        
    def shift_left(self, image):
        WIDTH = image.shape[1]
        HEIGHT = image.shape[0]

        for j in range(WIDTH):
            for i in range(HEIGHT):
                if (i < HEIGHT-10):
                    image[j][i] = image[j][i+10]

        return image

    def shift_up(self, image):
        WIDTH = image.shape[1]
        HEIGHT = image.shape[0]

        for j in range(WIDTH):
            for i in range(HEIGHT):
                if (j < WIDTH - 10 and j > 10):
                    image[j][i] = image[j+10][i]
                else:
                    image[j][i] = 0

        return image

    def shift_down(self, image):
        WIDTH = image.shape[1]
        HEIGHT = image.shape[0]

        for j in range(WIDTH, 1, -1):
            for i in range(HEIGHT):
                if (j < HEIGHT - 10 and j > 10):
                    image[j][i] = image[j-10][i]

        return image


    def add_noise(self, image):
        WIDTH = image.shape[1]
        HEIGHT = image.shape[0]

        noise = np.random.uniform(low=0,high=0.1,size = (WIDTH, HEIGHT))

        for i in range(WIDTH):
            for j in range(HEIGHT):
                if (image[i][j] >= 0.2):
                    image[i][j] += noise[i][j]

        return image

c = ImageProcessingPipeline()
c.create_training_set()
