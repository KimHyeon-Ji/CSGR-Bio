import numpy as np
import os
import torch
import pickle
from tqdm import tqdm

from torch.utils.data import Dataset
from Utils.Augmentation import Aug
from Dataloaders.Dataset.MyoArmband_preprocess_utils import load_myo_data

import argparse

class MyoArmband_Dataset(Dataset):
    def __init__(self, args: argparse, x, x_aug, y, s, mode):
        self.args = args
        self.mode = mode
        self.x = x
        self.x_aug = x_aug
        self.y = y
        self.s = s

        if args.dataset_ver == 1:
            self.users = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            self.target_users = [8, 9, 19, 20, 21]
        elif args.dataset_ver == 2:
            self.users = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21]
            self.target_users = [4, 5, 13, 14, 15]
        elif args.dataset_ver == 3:
            self.users = [0, 1, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            self.target_users = [2, 3, 10, 11, 12]
        elif args.dataset_ver == 4:
            self.users = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20]
            self.target_users = [5, 8, 12, 15, 21]
        elif args.dataset_ver == 5:
            self.users = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 20, 21]
            self.target_users = [0, 5, 13, 16, 19]

        if args.abl_ver == 1:
            self.users = self.users[:13]
        elif args.abl_ver == 2:
            self.users = self.users[:10]
        elif args.abl_ver == 3:
            self.users = self.users[:7]
        elif args.abl_ver == 4:
            self.users = self.users[:4]

        if args.data_split == 'random':
            self.X, self.target, self.X_aug, self.subject = self.random_split()

        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.target[index], self.X_aug[index], self.subject[index]

    def __len__(self):
        return self.len

    def random_split(self):

        inds = np.empty((0), dtype=int)

        if self.mode == 'random':
            for i in self.users:
                inds = np.append(inds, np.where(self.s == i)[0])
        elif self.mode == 'test':
            for i in self.target_users:
                inds = np.append(inds, np.where(self.s == i)[0])

        X_data = torch.FloatTensor(self.x[inds, :, :])
        X_aug_data = torch.FloatTensor(self.x_aug[inds, :, :])

        target_data = torch.FloatTensor(self.y[inds])
        subject_data = torch.FloatTensor(self.s[inds])

        return X_data, target_data, X_aug_data, subject_data