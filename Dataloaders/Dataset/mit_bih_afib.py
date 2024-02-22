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

    # def subject_wise_split(self):
    #     with open(os.path.join(self.data_path, f'x_train_{self.dataset_ver}.pkl'), 'rb') as fr:
    #         x_train_valid = pickle.load(fr)
    #     with open(os.path.join(self.data_path, f'state_train_{self.dataset_ver}.pkl'), 'rb') as fr:
    #         y_train_valid = pickle.load(fr)
    #
    #     len_train = int(len(x_train_valid) * 0.7)
    #
    #     if self.mode == 'train':
    #         x_np = x_train_valid[:len_train]
    #         y_np = y_train_valid[:len_train]
    #         s_np = np.arange(len_train)
    #
    #     elif self.mode == 'valid':
    #         x_np = x_train_valid[len_train:]
    #         y_np = y_train_valid[len_train:]
    #         s_np = np.arange(len_train, len(x_train_valid), 1)
    #
    #     elif self.mode == 'test':
    #         with open(os.path.join(self.data_path, f'x_test_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             x_np = pickle.load(fr)
    #         with open(os.path.join(self.data_path, f'state_test_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             y_np = pickle.load(fr)
    #         s_np = np.arange(len(x_train_valid), len(x_train_valid)+len(x_np), 1)
    #
    #     x_np = x_np.transpose((0,2,1))
    #
    #     X_list = []
    #     target_list = []
    #     subject_list = []
    #
    #     for i in range(x_np.shape[0]):
    #         x_df = pd.DataFrame(x_np[i])
    #         s = s_np[i]
    #         y = y_np[i]
    #         for start_idx in range(0, x_df.shape[0] - self.window_size + 1, self.window_size):
    #             X_list.append(x_df[start_idx:start_idx + self.window_size])
    #             target_list.append(np.bincount(y[start_idx:start_idx + self.window_size].astype(int)).argmax())
    #             subject_list.append(s)
    #
    #     X_data = torch.FloatTensor(np.array(X_list))
    #     X_aug_data = torch.FloatTensor(Aug(np.array(X_list), args=self.args))
    #
    #     target_data = torch.FloatTensor(np.array(target_list))
    #     subject_data = torch.FloatTensor(np.array(subject_list))
    #
    #     return X_data, target_data, X_aug_data, subject_data
    #
    # def random_split(self):
    #     if self.mode == 'random':
    #         with open(os.path.join(self.data_path, f'x_train_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             X_np = pickle.load(fr)
    #         with open(os.path.join(self.data_path, f'state_train_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             y_np = pickle.load(fr)
    #         s_np = np.arange(X_np.shape[0])
    #
    #     elif self.mode == 'test':
    #         with open(os.path.join(self.data_path, f'x_test_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             X_np = pickle.load(fr)
    #         with open(os.path.join(self.data_path, f'state_test_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             y_np = pickle.load(fr)
    #         s_np = np.arange(X_np.shape[0])
    #
    #     X_np = X_np.transpose((0, 2, 1))
    #
    #     X_list = []
    #     target_list = []
    #     subject_list = []
    #
    #     for i in range(X_np.shape[0]):
    #         x_df = pd.DataFrame(X_np[i])
    #         s = s_np[i]
    #         y = y_np[i]
    #         for start_idx in range(0, x_df.shape[0] - self.window_size + 1, self.window_size):
    #             X_list.append(x_df[start_idx:start_idx + self.window_size])
    #             target_list.append(np.bincount(y[start_idx:start_idx + self.window_size].astype(int)).argmax())
    #             subject_list.append(s)
    #
    #     X_data = torch.FloatTensor(np.array(X_list))
    #     X_aug_data = torch.FloatTensor(Aug(np.array(X_list), args=self.args))
    #
    #     target_data = torch.FloatTensor(np.array(target_list))
    #     subject_data = torch.FloatTensor(np.array(subject_list))
    #
    #     return X_data, target_data, X_aug_data, subject_data


    # def random_split(self):
    #     if self.mode == 'random':
    #         with open(os.path.join(self.data_path, f'x_train_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             X_np = pickle.load(fr)
    #         with open(os.path.join(self.data_path, f'state_train_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             y_np = pickle.load(fr)
    #
    #
    #     elif self.mode == 'test':
    #         with open(os.path.join(self.data_path, f'x_test_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             X_np = pickle.load(fr)
    #         with open(os.path.join(self.data_path, f'state_test_{self.dataset_ver}.pkl'), 'rb') as fr:
    #             y_np = pickle.load(fr)
    #
    #     T = X_np.shape[-1]
    #     x_window = np.split(X_np[:, :, :self.window_size * (T // self.window_size)], (T // self.window_size), -1)
    #     y_window = np.concatenate(
    #         np.split(y_np[:, :self.window_size * (T // self.window_size)], (T // self.window_size), -1), 0).astype(int)
    #
    #     s_np = np.full((T // self.window_size), 0)
    #     for i in range(X_np.shape[0] - 1):
    #         s_np = np.append(s_np, np.full((T // self.window_size), i + 1))
    #
    #     X_np = np.concatenate(x_window, 0)
    #     X_np = X_np.transpose((0, 2, 1))
    #     y_np = np.array([np.bincount(yy).argmax() for yy in y_window])
    #
    #     X_data = torch.FloatTensor(X_np)
    #     X_aug_data = torch.FloatTensor(Aug(X_np, args=self.args))
    #
    #     target_data = torch.FloatTensor(y_np)
    #     subject_data = torch.FloatTensor(s_np)
    #
    #     return X_data, target_data, X_aug_data, subject_data


