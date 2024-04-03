import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import StratifiedShuffleSplit

from Dataloaders.Dataset import mit_bih_afib, UniMiB, MyoArmband
from Utils.Path import mit_bih_afib_path, UniMiB_path, MyoArmband_path

def make_dataloader(args: argparse, **kwargs: dict):

    if args.dataset == 'MyoArmband':

        data_path = MyoArmband_path
        x = np.load(os.path.join(data_path, 'x_data.npy'))
        x_aug = np.load(os.path.join(data_path, 'x_aug_data.npy'))
        y = np.load(os.path.join(data_path, 'y_data.npy'))
        s = np.load(os.path.join(data_path, 's_data.npy'))

        dataset = MyoArmband.MyoArmband_Dataset(args, x, x_aug, y, s, mode='random')
        testset = MyoArmband.MyoArmband_Dataset(args, x, x_aug, y, s, mode='test')

        n_val = int(0.2 * len(dataset))
        trainset, valset = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    elif args.dataset == 'mit_bih_afib':

        data_path = mit_bih_afib_path
        x_train = np.load(os.path.join(data_path, f'x_train_{args.dataset_ver}.npy'))
        x_aug_train = np.load(os.path.join(data_path, f'x_aug_train_{args.dataset_ver}.npy'))
        y_train = np.load(os.path.join(data_path, f'y_train_{args.dataset_ver}.npy'))
        s_train = np.load(os.path.join(data_path, f's_train_{args.dataset_ver}.npy'))

        x_test = np.load(os.path.join(data_path, f'x_test_{args.dataset_ver}.npy'))
        x_aug_test = np.load(os.path.join(data_path, f'x_aug_test_{args.dataset_ver}.npy'))
        y_test = np.load(os.path.join(data_path, f'y_test_{args.dataset_ver}.npy'))
        s_test = np.load(os.path.join(data_path, f's_test_{args.dataset_ver}.npy'))

        dataset = mit_bih_afib.mit_bih_afib_Dataset(args, x_train, x_aug_train, y_train, s_train)
        testset = mit_bih_afib.mit_bih_afib_Dataset(args, x_test, x_aug_test, y_test, s_test)

        n_val = int(0.2 * len(dataset))
        trainset, valset = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    elif args.dataset == 'UniMiB_adl':

        data_path = UniMiB_path
        x = np.load(os.path.join(data_path, 'x_data_adl.npy'))
        x_aug = np.load(os.path.join(data_path, 'x_aug_data_adl.npy'))
        y = np.load(os.path.join(data_path, 'y_data_adl.npy'))
        s = np.load(os.path.join(data_path, 's_data_adl.npy'))

        dataset = UniMiB.UniMiB_Dataset(args, x, x_aug, y, s, mode='random')
        testset = UniMiB.UniMiB_Dataset(args, x, x_aug, y, s, mode='test')

        n_val = int(0.2 * len(dataset))
        trainset, valset = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs)

    return trainloader, valloader, testloader