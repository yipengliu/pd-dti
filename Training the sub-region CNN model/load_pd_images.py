import scipy.io as scio
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import torch
import random
import copy

def data_augmentation(data):
    # add noise
    size = data.shape
    noise = (torch.randn(size) / torch.randint(10, 30, (1,)).float())
    data = data + noise
    # random crop
    axis_0 = random.randint(-10, 10)
    if axis_0 > 0:
        data_copy = copy.deepcopy(data)
        data[axis_0:, :, :] = data_copy[:-axis_0, :, :]
        data[:axis_0, :, :] = 0
    elif axis_0 < 0:
        data_copy = copy.deepcopy(data)
        data[:axis_0, :, :] = data_copy[-axis_0:, :, :]
        data[axis_0:, :, :] = 0

    axis_1 = random.randint(-10, 10)
    if axis_1 > 0:
        data_copy = copy.deepcopy(data)
        data[:, axis_1:, :] = data_copy[:, :-axis_1, :]
        data[:, :axis_1, :] = 0
    elif axis_1 < 0:
        data_copy = copy.deepcopy(data)
        data[:, :axis_1, :] = data_copy[:, -axis_1:, :]
        data[:, axis_1:, :] = 0

    axis_2 = random.randint(-4, 4)
    if axis_2 > 0:
        data_copy = copy.deepcopy(data)
        data[:, :, axis_2:] = data_copy[:, :, :-axis_2]
        data[:, :, :axis_2] = 0
    elif axis_2 < 0:
        data_copy = copy.deepcopy(data)
        data[:, :, :axis_2] = data_copy[:, :, -axis_2:]
        data[:, :, axis_2:] = 0
    # flip
    random_num = random.randint(-8, 8)
    if random_num >= 0:
        data[:, :, :] = torch.flip(data[:, :, :], [1, 2])

    return data


class pd_data(torch.utils.data.Dataset):
    def __init__(self, samples, targets, is_train,
                 transform=None, target_transform=None):

        self.samples = samples
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train        

    def __getitem__(self, index):
        sample, target = self.samples[index], self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)  
        # if self.is_train:
        #     sample = data_augmentation(sample) 
        if self.target_transform is not None:
            target = self.target_transform(target)     

        return sample, target

    def __len__(self):
        return len(self.samples)
