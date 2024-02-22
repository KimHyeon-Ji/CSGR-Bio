from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import wfdb
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.model_selection import train_test_split

# Local imports
# from deepecg.config.config import DATA_DIR
# DATA_DIR = "./Data/mit-bih-atrial-fibrillation-database-1.0.0"
DATA_DIR = "C:/Users/DMQA/Desktop/MLRL/Data/mit-bih-atrial-fibrillation-database-1.0.0"
afib_dict = {"AFIB":0, "AFL":1, "J":2, "N":3}

dataset_ver = 5
ablation = 15


class AFDB(object):
    """
        The MIT-BIH Atrial Fibrillation Database
        https://physionet.org/physiobank/database/afdb/
        """

    def __init__(self):
        # Set attributes
        self.db_name = 'afdb'
        self.raw_path = os.path.join(DATA_DIR, 'raw')
        self.processed_path = os.path.join(DATA_DIR, 'processed')
        self.label_dict = {'AFIB': 'atrial fibrillation', 'AFL': 'atrial flutter', 'J': 'AV junctional rhythm'}
        self.fs = 300
        self.length = 60
        self.length_sp = self.length * self.fs
        self.record_ids = None
        self.sections = None
        self.samples = None
        self.labels = None

    def generate_db(self):
        """Generate raw and processed databases."""
        # Generate raw database
        self.generate_raw_db()

        # Generate processed database
        self.generate_processed_db()

    def generate_raw_db(self):
        """Generate the raw version of the MIT-BIH Atrial Fibrillation database in the 'raw' folder."""
        # Download database
        if len(os.listdir(self.raw_path))==0:
            print('Generating Raw MIT-BIH Atrial Fibrillation Database ...')
            wfdb.dl_database(self.db_name, self.raw_path)
            print('Complete!\n')

        # Get list of recordings
        self.record_ids = [file.split('.')[0] for file in os.listdir(self.raw_path) if '.dat' in file]

    def generate_processed_db(self):
        """Generate the processed version of the MIT-BIH Atrial Fibrillation database in the 'processed' folder."""
        print('Generating Processed MIT-BIH Atrial Fibrillation Database ...')
        all_signals, all_labels = self._get_sections()

        signal_lens = [len(sig) for sig in all_labels]
        all_signals = np.array([sig[:,:min(signal_lens)] for sig in all_signals])
        all_labels = np.array([sig[:min(signal_lens)] for sig in all_labels])

        # # 1
        # n_train = int(0.8*len(all_signals)) # n_train: 18
        # train_data = all_signals[:n_train]
        # test_data = all_signals[n_train:]
        # train_state = all_labels[:n_train]
        # test_state = all_labels[n_train:]

        # # 1 ablation
        # n_train = 3  # n_train: 18
        # train_data = all_signals[:n_train]
        # test_data = all_signals[18:]
        # train_state = all_labels[:n_train]
        # test_state = all_labels[18:]

        # # 2
        # train_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,21,22]
        # test_list = [15,16,17,18,20]
        # train_data = all_signals[train_list]
        # test_data = all_signals[test_list]
        # train_state = all_labels[train_list]
        # test_state = all_labels[test_list]

        # # 2 ablation
        # train_list = [8,10,0,1,2,3,4,5,6,7,9,11,12,13,14,19,21,22]
        # test_list = [15,16,17,18,20]
        # train_data = all_signals[train_list[:15]]
        # test_data = all_signals[test_list]
        # train_state = all_labels[train_list[:15]]
        # test_state = all_labels[test_list]

        # # 3
        # train_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22]
        # test_list = [10,11,12,13,14]
        # train_data = all_signals[train_list]
        # test_data = all_signals[test_list]
        # train_state = all_labels[train_list]
        # test_state = all_labels[test_list]

        # # 3 ablation
        # train_list = [8, 15, 0, 1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 22]
        # test_list = [10,11,12,13,14]
        # train_data = all_signals[train_list[:3]]
        # test_data = all_signals[test_list]
        # train_state = all_labels[train_list[:3]]
        # test_state = all_labels[test_list]

        # 4
        # train_list = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        # test_list = [5, 6, 7, 8, 9]
        # train_data = all_signals[train_list]
        # test_data = all_signals[test_list]
        # train_state = all_labels[train_list]
        # test_state = all_labels[test_list]

        # 4 ablation
        # train_list = [10, 15, 19, 16, 0, 1, 2, 3, 4,  11, 12, 13, 14, 17, 18, 19, 20, 21, 22]
        # test_list = [5, 6, 7, 8, 9]
        # train_data = all_signals[train_list[:ablation]]
        # test_data = all_signals[test_list]
        # train_state = all_labels[train_list[:ablation]]
        # test_state = all_labels[test_list]

        # 5
        # train_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        # test_list = [0, 1, 2, 3, 4]
        # train_data = all_signals[train_list]
        # test_data = all_signals[test_list]
        # train_state = all_labels[train_list]
        # test_state = all_labels[test_list]

        # 5 ablation
        train_list = [5, 19, 10, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22]
        test_list = [0, 1, 2, 3, 4]
        train_data = all_signals[train_list[:ablation]]
        test_data = all_signals[test_list]
        train_state = all_labels[train_list[:ablation]]
        test_state = all_labels[test_list]

        # Normalize signals
        train_data_n, test_data_n = self._normalize(train_data, test_data)

        # Save signals to file
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
        with open(os.path.join(self.processed_path, f'x_train_{dataset_ver}_ab{ablation}.pkl'), 'wb') as f:
            pickle.dump(train_data_n, f)
        # with open(os.path.join(self.processed_path, 'x_test_1.pkl'), 'wb') as f:
        #     pickle.dump(test_data_n, f)
        with open(os.path.join(self.processed_path, f'state_train_{dataset_ver}_ab{ablation}.pkl'), 'wb') as f:
            pickle.dump(train_state, f)
        # with open(os.path.join(self.processed_path, 'state_test_4.pkl'), 'wb') as f:
        #     pickle.dump(test_state, f)

    def _normalize(self, train_data, test_data):
        """ Calculate the mean and std of each feature from the training set
        """
        feature_means = np.mean(train_data, axis=(0, 2))
        feature_std = np.std(train_data, axis=(0, 2))
        train_data_n = (train_data - feature_means[np.newaxis, :, np.newaxis]) / \
                       np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        test_data_n = (test_data - feature_means[np.newaxis, :, np.newaxis]) /\
                      np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
        return train_data_n, test_data_n

    def _get_sections(self):
        """Collect continuous arrhythmia sections."""
        # Empty dictionary for arrhythmia sections
        all_signals = []
        all_labels = []

        # Loop through records
        for record_id in self.record_ids:
            # Import recording
            record = wfdb.rdrecord(os.path.join(self.raw_path, record_id))

            # Import annotations
            annotation = wfdb.rdann(os.path.join(self.raw_path, record_id), 'atr')

            # Get sample frequency
            fs = record.__dict__['fs']

            # Get waveform
            waveform = record.__dict__['p_signal']  #shape: (length, n_channels=2)

            # labels
            labels = [label[1:] for label in annotation.__dict__['aux_note']]

            # Samples
            sample = annotation.__dict__['sample']

            padded_labels = np.zeros(len(waveform))
            for i,l in enumerate(labels):
                if i==len(labels)-1:
                    padded_labels[sample[i]:] = afib_dict[l]
                else:
                    padded_labels[sample[i]:sample[i+1]] = afib_dict[l]
            padded_labels = padded_labels[sample[0]:]
            all_labels.append(padded_labels)
            all_signals.append(waveform[sample[0]:,:].T)

        return all_signals, all_labels


if __name__=="__main__":
    a = AFDB()
    a.generate_raw_db()
    a.generate_processed_db()