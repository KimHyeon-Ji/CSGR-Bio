import os
import argparse

mit_bih_afib_path = './Data/mit-bih-atrial-fibrillation-database-1.0.0/data/'
UCI_HAR_path = './Data/UCI_HAR_Dataset/'
UniMiB_path = './Data/UniMiB-SHAR/'
sleep_edf_path = './Data/sleepEDF20_fpzcz_subjects'
WISDM_ar_path = './Data/WISDM/WISDM_ar_latest.tar.gz'
WISDM_at_path = './Data/WISDM/WISDM_at_latest.tar.gz'
MyoArmband_path = './Data/MyoArmbandDataset'
BCI_motor_imagery_path = './Data/BCI_4_c_motor_imagery/'
MNE_alphawaves_path = './Data/MNE_alphawaves_data'
Auditory_EEG_path = './Data/Auditory_EEG'
EEG_mental_path = './Data/eeg-during-mental-arithmetic-tasks-1.0.0'

class Path(object): # 이건 뭐지?
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