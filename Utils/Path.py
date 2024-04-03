import os
import argparse

mit_bih_afib_path = './Data/mit-bih-atrial-fibrillation-database-1.0.0/data/'
UniMiB_path = './Data/UniMiB-SHAR/'
MyoArmband_path = './Data/MyoArmbandDataset'

class Path(object): 
    @staticmethod
    def pathology_root_dir():
        return './PathologyDataSet_256'


def mypath(args: argparse):
    subloss = args.sub_loss
    dataset = args.dataset
    datase_ver = args.dataset_ver
    task = args.task
    rep = args.rep

    top1_save_path = f'./Results/{subloss}'

    if not os.path.exists(top1_save_path):
        os.makedirs(top1_save_path)

    top2_save_path = f'./Results/{subloss}/{dataset}'

    if not os.path.exists(top2_save_path):
        os.makedirs(top2_save_path)

    top3_save_path = f'./Results/{subloss}/{dataset}/{datase_ver}'

    if not os.path.exists(top3_save_path):
        os.makedirs(top3_save_path)

    top4_save_path = f'./Results/{subloss}/{dataset}/{datase_ver}/{task}'

    if not os.path.exists(top4_save_path):
        os.makedirs(top4_save_path)

    if task == 'single_model':
        if len(os.listdir(top4_save_path)) > 0:
            x = len(os.listdir(top4_save_path))
            x = x + 1
        else:
            x = 1

        save_path = f'./Results/{subloss}/{dataset}/{datase_ver}/{task}/Experiment_{x}'

    else:

        top5_save_path = f'./Results/{subloss}/{dataset}/{datase_ver}/{task}/{rep}'

        if not os.path.exists(top5_save_path):
            os.makedirs(top5_save_path)

        if len(os.listdir(top5_save_path)) > 0:
            x = len(os.listdir(top5_save_path))
            x = x + 1
        else:
            x = 1

        save_path = f'./Results/{subloss}/{dataset}/{datase_ver}/{task}/{rep}/Experiment_{x}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path