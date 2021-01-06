import os
import random
import math

import Augmentor
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets as dset


def get_train_validation_loader(data_dir, batch_size, num_train, augment, way, trials, shuffle=False, seed=0,
                                num_workers=4, pin_memory=False):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = dset.ImageFolder(train_dir)

    print(train_dataset)
