import numpy as np

import torch
from torch.utils.data import Dataset
import argparse


class UniMiB_Dataset(Dataset):
    def __init__(self, args: argparse, x, x_aug, y, s, mode):
        self.args = args
        self.mode = mode
        self.x = x
        self.x_aug = x_aug
        self.y = y
        self.s = s

        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        if args.dataset_ver == 1:
            self.users = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
            self.target_users = [2, 4, 9, 10, 12, 13, 18, 20, 24]
        elif args.dataset_ver == 2:
            self.users = [2, 3, 4, 6, 7, 9, 10, 13, 14, 15, 16, 17, 19, 20, 22, 23, 25, 26, 27, 28, 30]
            self.target_users = [1, 5, 8, 11, 12, 18, 21, 24, 29]
        elif args.dataset_ver == 3:
            self.users = [1, 2, 4, 5, 8, 9, 11, 12, 13, 16, 17, 18, 21, 22, 23, 25, 26, 27, 28, 29, 30]
            self.target_users = [3, 6, 7, 10, 14, 15, 19, 20, 24]
        elif args.dataset_ver == 4:
            self.users = [1, 2, 4, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 21, 23, 24, 27, 28, 29, 30]
            self.target_users = [3, 5, 10, 13, 16, 20, 22, 25, 26]
        elif args.dataset_ver == 5:
            self.users = [1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 15, 16, 17, 19, 20, 21, 24, 25, 27, 28, 30]
            self.target_users = [4, 7, 12, 14, 18, 22, 23, 26, 29]

        if args.abl_ver == 1:
            self.users = self.users[:18]
        elif args.abl_ver == 2:
            self.users = self.users[:15]
        elif args.abl_ver == 3:
            self.users = self.users[:12]
        elif args.abl_ver == 4:
            self.users = self.users[:9]

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