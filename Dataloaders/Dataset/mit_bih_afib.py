import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from Utils.Augmentation import Aug
import argparse
import pickle

class mit_bih_afib_Dataset(Dataset):
    def __init__(self, args: argparse, x, x_aug, y, s):
        self.args = args
        self.x = x
        self.x_aug = x_aug
        self.y = y
        self.s = s

        if args.data_split == 'random':
            self.X, self.target, self.X_aug, self.subject = self.random_split()

        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.target[index], self.X_aug[index], self.subject[index]

    def __len__(self):
        return self.len

    def random_split(self):

        X_data = torch.FloatTensor(self.x)
        X_aug_data = torch.FloatTensor(self.x_aug)

        target_data = torch.FloatTensor(self.y)
        subject_data = torch.FloatTensor(self.s)

        return X_data, target_data, X_aug_data, subject_data
