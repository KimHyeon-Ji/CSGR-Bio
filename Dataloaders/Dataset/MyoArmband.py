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

        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        if args.dataset_ver == 1:
            self.users = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            self.target_users = [8, 9, 19, 20, 21]
        elif args.dataset_ver == 2:
            self.users = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21]
            self.target_users = [6, 7, 16, 17, 18]
        elif args.dataset_ver == 3:
            self.users = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21]
            self.target_users = [4, 5, 13, 14, 15]
        elif args.dataset_ver == 4:
            self.users = [0, 1, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            self.target_users = [2, 3, 10, 11, 12]
        elif args.dataset_ver == 5:
            self.users = [2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            self.target_users = [0, 1, 10, 15, 17]
        elif args.dataset_ver == 6:
            self.users = [0, 1, 3, 4, 5, 6, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]
            self.target_users = [2, 7, 9, 13, 14]
        elif args.dataset_ver == 7:
            self.users = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20]
            self.target_users = [5, 8, 12, 15, 21]
        elif args.dataset_ver == 8:
            self.users = [0, 1, 2, 3, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 21]
            self.target_users = [4, 9, 11, 19, 20]
        elif args.dataset_ver == 9:
            self.users = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20]
            self.target_users = [0, 7, 13, 17, 21]
        elif args.dataset_ver == 10:
            self.users = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 18, 19, 21]
            self.target_users = [2, 8, 15, 17, 20]
        elif args.dataset_ver == 11:
            self.users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            self.target_users = [17, 18, 19, 20, 21]
        elif args.dataset_ver == 12:
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

    # def random_split(self):
    #
    #     folder_map = {}
    #     for i in range(0, 9 + 1):
    #         folder_map[i] = (os.path.join(self.pre_folder, "Female{}".format(i)), False)
    #     for i in range(10, 21 + 1):
    #         folder_map[i] = (os.path.join(self.pre_folder, "Male{}".format(i - 10)), False)
    #     for i in range(22, 23 + 1):
    #         folder_map[i] = (os.path.join(self.eval_folder, "Female{}".format(i - 22)), True)
    #     for i in range(24, 39 + 1):
    #         folder_map[i] = (os.path.join(self.eval_folder, "Male{}".format(i - 24)), True)
    #
    #     data_train = []
    #     labels_train = []
    #     subjects_train = []
    #     data_test = []
    #     labels_test = []
    #     subjects_test = []
    #
    #     for user in tqdm(self.users):
    #         folder, is_evaluation = folder_map[user]
    #         user_data_train, user_labels_train = self.get_data(folder, "training0")
    #         user_subjects_train = np.full(user_labels_train.shape, user)
    #
    #         user_data_test = None
    #         user_labels_test = None
    #
    #         if is_evaluation:
    #             user_data_test0, user_labels_test0 = self.get_data(folder, "Test0")
    #             user_data_test1, user_labels_test1 = self.get_data(folder, "Test1")
    #             user_data_test = np.vstack([user_data_test0, user_data_test1]).astype(np.float32)
    #             user_labels_test = np.hstack([user_labels_test0, user_labels_test1]).astype(np.float32)
    #             user_subjects_test = np.full(user_labels_test.shape, user)
    #
    #         data_train.append(user_data_train)
    #         labels_train.append(user_labels_train)
    #         subjects_train.append(user_subjects_train)
    #
    #         if user_data_test is not None:
    #             data_test.append(user_data_test)
    #             labels_test.append(user_labels_test)
    #             subjects_test.append(user_subjects_test)
    #
    #     x_train = np.vstack(data_train).astype(np.float32)
    #     y_train = np.hstack(labels_train).astype(np.float32)
    #     s_train = np.hstack(subjects_train).astype(np.float32)
    #
    #     if len(data_test) > 0 and len(labels_test) > 0:
    #         x_test = np.vstack(data_test).astype(np.float32)
    #         y_test = np.hstack(labels_test).astype(np.float32)
    #         s_test = np.hstack(subjects_test).astype(np.float32)
    #     else:
    #         x_test = np.array([], dtype=np.float32)
    #         y_test = np.array([], dtype=np.float32)
    #         s_test = np.array([], dtype=np.float32)
    #
    #     if self.mode == 'random':
    #         X_data = torch.FloatTensor(x_train)
    #         X_aug_data = torch.FloatTensor(Aug(x_train, args=self.args))
    #         target_data = torch.FloatTensor(y_train)
    #         subject_data = torch.FloatTensor(s_train)
    #     elif self.mode == 'test':
    #         X_data = torch.FloatTensor(x_test)
    #         X_aug_data = torch.FloatTensor(Aug(x_test, args=self.args))
    #         target_data = torch.FloatTensor(y_test)
    #         subject_data = torch.FloatTensor(s_test)
    #
    #     return X_data, target_data, X_aug_data, subject_data
    #
    # def get_data(self, folder, data_type):
    #     """ Load one user's data, based on load_myo_data.read_data() """
    #     user_examples = []
    #     user_labels = []
    #
    #     for i in range(load_myo_data.number_of_classes * 4):
    #         data_file = os.path.join(folder, data_type, "classe_{}.dat".format(i))
    #         data_read_from_file = np.fromfile(data_file, dtype=np.int16)
    #         data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
    #         dataset_example = load_myo_data.format_data_to_train(data_read_from_file)
    #         labels = (i % load_myo_data.number_of_classes) + np.zeros(dataset_example.shape[0])
    #
    #         # print("{}:".format(i),
    #         #     "class", i%load_myo_data.number_of_classes,
    #         #     "iter", i//load_myo_data.number_of_classes,
    #         #     "-", dataset_example.shape, labels.shape)
    #
    #         user_examples.append(dataset_example)
    #         user_labels.append(labels)
    #
    #     user_examples, user_labels = load_myo_data.shift_electrodes(user_examples, user_labels)
    #
    #     # Convert from list of examples to one matrix
    #     user_examples = np.vstack(user_examples).astype(np.float32)
    #     user_labels = np.hstack(user_labels).astype(np.float32)
    #
    #     # Remove extra 1 dimension, e.g. [examples, 1, features, time_steps]
    #     user_examples = np.squeeze(user_examples, axis=1)
    #
    #     # Transpose from [examples, features, time_steps] to
    #     # [examples, time_steps, features]
    #     user_examples = np.transpose(user_examples, axes=[0, 2, 1])
    #
    #     return user_examples, user_labels