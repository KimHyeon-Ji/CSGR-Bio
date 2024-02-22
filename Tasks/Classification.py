import torch.optim

import os
import pandas as pd
import numpy as np
import tqdm
import argparse
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as po
import seaborn as sns
import matplotlib.pyplot as plt

from Utils.Saver import Saver
from Utils.Loggers import get_tqdm_config
from Utils.Summary import TensorboardSummary
from Dataloaders.Dataloader import make_dataloader
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import f1_score
from Models.Backbone.LSTM_FCN import *
from Models.Backbone.WFEncoder import *
from Models.Classification_Model import *

class Classification_Trainer(object):
    def __init__(self, args: argparse, check_path: str):

        self.args = args
        self.check_path = check_path

        # Define DataLoader
        kwargs = {'pin_memory': True, 'num_workers': 4}
        self.train_loader, self.valid_loader, self.test_loader = make_dataloader(args, **kwargs)

        # Define Saver
        self.saver = Saver(path=check_path)

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(directory=self.saver.path)
        self.writer = self.summary.create_summary()

        # Dataset Setting
        if args.dataset == 'sleep_edf':
            self.num_features = args.sleep_edf_num_features
            self.num_classes = args.sleep_edf_num_classes
            self.window_size = args.sleep_edf_window_size
            self.FCN_kernel_size = args.sleep_edf_FCN_kernel_size
            self.FCN_stride = args.sleep_edf_FCN_stride
        elif args.dataset == 'mit_bih_afib':
            self.num_features = args.mit_bih_afib_num_features
            self.num_classes = args.mit_bih_afib_num_classes
            self.window_size = args.mit_bih_afib_window_size
            self.FCN_kernel_size = args.mit_bih_afib_FCN_kernel_size
            self.FCN_stride = args.mit_bih_afib_FCN_stride
        elif args.dataset == 'UCI_HAR':
            self.num_features = args.UCI_HAR_num_features
            self.num_classes = args.UCI_HAR_num_classes
            self.window_size = args.UCI_HAR_window_size
            self.FCN_kernel_size = args.UCI_HAR_FCN_kernel_size
            self.FCN_stride = args.UCI_HAR_FCN_stride
        elif args.dataset == 'WISDM_ar':
            self.num_features = args.WISDM_ar_num_features
            self.num_classes = args.WISDM_ar_num_classes
            self.window_size = args.WISDM_ar_window_size
            self.FCN_kernel_size = args.WISDM_ar_FCN_kernel_size
            self.FCN_stride = args.WISDM_ar_FCN_stride
        elif args.dataset == 'WISDM_at':
            self.num_features = args.WISDM_at_num_features
            self.num_classes = args.WISDM_at_num_classes
            self.window_size = args.WISDM_at_window_size
            self.FCN_kernel_size = args.WISDM_at_FCN_kernel_size
            self.FCN_stride = args.WISDM_at_FCN_stride
        elif args.dataset == 'MyoArmband':
            self.num_features = args.MyoArmband_num_features
            self.num_classes = args.MyoArmband_num_classes
            self.window_size = args.MyoArmband_window_size
            self.FCN_kernel_size = args.MyoArmband_FCN_kernel_size
            self.FCN_stride = args.MyoArmband_FCN_stride
        elif args.dataset == 'BCI_motor_imagery':
            self.num_features = args.BCI_motor_imagery_num_features
            self.num_classes = args.BCI_motor_imagery_num_classes
            self.window_size = args.BCI_motor_imagery_window_size
            self.FCN_kernel_size = args.BCI_motor_imagery_FCN_kernel_size
            self.FCN_stride = args.BCI_motor_imagery_FCN_stride
        elif args.dataset == 'MNE_alphawaves':
            self.num_features = args.MNE_alphawaves_num_features
            self.num_classes = args.MNE_alphawaves_num_classes
            self.window_size = args.MNE_alphawaves_window_size
            self.FCN_kernel_size = args.MNE_alphawaves_FCN_kernel_size
            self.FCN_stride = args.MNE_alphawaves_FCN_stride
        elif args.dataset == 'Auditory_EEG':
            self.num_features = args.Auditory_EEG_num_features
            self.num_classes = args.Auditory_EEG_num_classes
            self.window_size = args.Auditory_EEG_window_size
            self.FCN_kernel_size = args.Auditory_EEG_FCN_kernel_size
            self.FCN_stride = args.Auditory_EEG_FCN_stride
        elif args.dataset == 'EEG_mental':
            self.num_features = args.EEG_mental_num_features
            self.num_classes = args.EEG_mental_num_classes
            self.window_size = args.EEG_mental_window_size
            self.FCN_kernel_size = args.EEG_mental_FCN_kernel_size
            self.FCN_stride = args.EEG_mental_FCN_stride
        elif args.dataset == 'UniMiB_adl':
            self.num_features = args.UniMiB_adl_num_features
            self.num_classes = args.UniMiB_adl_num_classes
            self.window_size = args.UniMiB_adl_window_size
            self.FCN_kernel_size = args.UniMiB_adl_FCN_kernel_size
            self.FCN_stride = args.UniMiB_adl_FCN_stride

        # Define Model
        self.LSTM_FCNs = LSTM_FCNs(output_dim=args.LSTM_FCN_output_dim, num_features=self.num_features,
                                   num_layers=args.LSTM_FCN_num_layer, lstm_drop_p=args.LSTM_FCN_lstm_drop_out,
                                   fc_drop_p=args.LSTM_FCN_fc_drop_out)

        backbone = [self.LSTM_FCNs, args.LSTM_FCN_output_dim]

        model = Classifier(backbone=backbone, rep_dim=args.LSTM_FCN_output_dim, num_classes=self.num_classes)

        # self.WFEncoder = WFEncoder(output_dim=args.LSTM_FCN_output_dim)
        # backbone = [self.WFEncoder, args.LSTM_FCN_output_dim]
        # model = Classifier(backbone=backbone, rep_dim=args.LSTM_FCN_output_dim, num_classes=self.num_classes)

        # Using Cuda
        self.model = model.to(device=args.cuda)

        # Define Optimizer lr_scheduler & Criterion
        if args.optim == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError('In-valid optimizer choice')

        if args.lr_scheduler == 'LambdaLR':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                               lr_lambda=lambda epoch: 0.95 ** epoch)
        elif args.lr_scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=10, gamma=0.5)
        elif args.lr_scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=30, eta_min=0)
        elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10,
                                                                                  T_mult=1, eta_min=0.00001)
        else:
            raise ValueError('In-valid lr_scheduler choice')

        self.criterion = nn.CrossEntropyLoss().to(device=args.cuda)


    def run(self, epochs):

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='red')) as pbar:

            valid_best_loss = float('inf')

            for epoch in range(1, epochs + 1):

                # train & valid
                train_loss, train_acc = self.train()
                valid_loss, valid_acc = self.valid()

                self.scheduler.step()

                epoch_history = {
                    'Loss' : {
                        'train' : train_loss,
                        'valid' : valid_loss
                    },
                    'Acc': {
                        'train': train_acc,
                        'valid': valid_acc
                    }
                }

                # Tensorboard summary
                for metric_name, metric_dict in epoch_history.items():
                    self.writer.add_scalars(
                        main_tag=metric_name,
                        tag_scalar_dict=metric_dict,
                        global_step=epoch
                    )

                # Save model if it is the current best
                current_valid_loss = epoch_history['Loss']['valid']
                if current_valid_loss < valid_best_loss:
                    valid_best_loss = current_valid_loss
                    self.saver.checkpoint('best_model', self.model, is_best=True)

                # Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04} |"
                for metric_name, metric_dict in epoch_history.items():

                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"

                pbar.set_description_str(desc)
                pbar.update(1)

            # Testing
            self.saver.load('best_model', self.model, is_best=True)
            preds_result, test_acc, test_avg_precision, test_f1_score_micro, test_f1_score_macro, test_f1_score_weighted = self.test()
            preds_result.to_csv(os.path.join(self.check_path, f'preds_result.csv'), index=False)

            # Result
            result = pd.DataFrame(
                {'valid_best_loss': valid_best_loss, 'test_acc': test_acc, 'test_avg_precision': test_avg_precision,
                 'test_f1_score_micro': test_f1_score_micro, 'test_f1_score_macro': test_f1_score_macro,
                 'test_f1_score_weighted': test_f1_score_weighted}, index=[0])
            result.to_csv(os.path.join(self.check_path, f'result.csv'), index=False)

            # Test data representation
            Fvecs, targets, subjects = self.feature(data_loader=self.test_loader)
            np.save(os.path.join(self.check_path, f'Fvecs.npy'), Fvecs)
            np.save(os.path.join(self.check_path, f'targets.npy'), targets)
            np.save(os.path.join(self.check_path, f'subjects.npy'), subjects)

            labels = pd.DataFrame({'targets': targets, 'subjects': subjects})
            labels['targets'] = labels['targets'].astype(str)
            labels['subjects'] = labels['subjects'].astype(str)

            if self.args.dataset == 'MNE_alphawaves':
                tsne = TSNE(n_components=2, perplexity= 20).fit_transform(Fvecs)
            else:
                tsne = TSNE(n_components=2).fit_transform(Fvecs)

            df = pd.DataFrame()
            df['y'] = labels.targets
            df['s'] = labels.subjects
            df["comp-1"] = tsne[:, 0]
            df["comp-2"] = tsne[:, 1]

            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), s=30, palette=sns.color_palette("Set3", 10),
                            data=df).set(title="T-SNE visualization of representation")
            plt.legend(sorted(list(map(float, list(np.unique(df.y))))), loc=2, bbox_to_anchor=(1, 1), title="Target")
            plt.savefig(os.path.join(self.check_path,'T-SNE_target.png'), bbox_inches = 'tight')

            plt.cla()

            sns.scatterplot(x="comp-1", y="comp-2", hue=df.s.tolist(), s=30, palette=sns.color_palette("Set3"),
                            data=df).set(title="T-SNE visualization of representation")
            plt.legend(sorted(list(map(float, list(np.unique(df.s))))), loc = 2, bbox_to_anchor = (1,1), title = "Subject")
            plt.savefig(os.path.join(self.check_path,'T-SNE_subject.png'), bbox_inches = 'tight')

            plt.cla()

            # # Train data representation
            # Fvecs, targets, subjects = self.feature(data_loader=self.train_loader)
            # # np.save(os.path.join(self.check_path, f'train_Fvecs.npy'), Fvecs)
            # # np.save(os.path.join(self.check_path, f'train_targets.npy'), targets)
            # # np.save(os.path.join(self.check_path, f'train_subjects.npy'), subjects)
            #
            # labels = pd.DataFrame({'targets': targets, 'subjects': subjects})
            # labels['targets'] = labels['targets'].astype(str)
            # labels['subjects'] = labels['subjects'].astype(str)
            #
            # if self.args.dataset == 'MNE_alphawaves':
            #     tsne = TSNE(n_components=2, perplexity=20).fit_transform(Fvecs)
            # else:
            #     tsne = TSNE(n_components=2).fit_transform(Fvecs)
            #
            # df = pd.DataFrame()
            # df['y'] = labels.targets
            # df['s'] = labels.subjects
            # df["comp-1"] = tsne[:, 0]
            # df["comp-2"] = tsne[:, 1]
            #
            # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), s=30, palette=sns.color_palette("Set3", 10),
            #                 data=df).set(title="T-SNE visualization of representation")
            # plt.legend(sorted(list(map(float, list(np.unique(df.y))))), loc=2, bbox_to_anchor=(1, 1), title="Target")
            # plt.savefig(os.path.join(self.check_path, 'T-SNE_target_train.png'), bbox_inches='tight')
            #
            # plt.cla()
            #
            # sns.scatterplot(x="comp-1", y="comp-2", hue=df.s.tolist(), s=30, palette=sns.color_palette("Set3"),
            #                 data=df).set(title="T-SNE visualization of representation")
            # plt.legend(sorted(list(map(float, list(np.unique(df.s))))), loc=2, bbox_to_anchor=(1, 1), title="Subject")
            # plt.savefig(os.path.join(self.check_path, 'T-SNE_subject_train.png'), bbox_inches='tight')
            #
            # plt.cla()
            #
            # # Valid data representation
            # Fvecs, targets, subjects = self.feature(data_loader=self.valid_loader)
            # # np.save(os.path.join(self.check_path, f'valid_train_Fvecs.npy'), Fvecs)
            # # np.save(os.path.join(self.check_path, f'valid_train_targets.npy'), targets)
            # # np.save(os.path.join(self.check_path, f'valid_train_subjects.npy'), subjects)
            #
            # labels = pd.DataFrame({'targets': targets, 'subjects': subjects})
            # labels['targets'] = labels['targets'].astype(str)
            # labels['subjects'] = labels['subjects'].astype(str)
            #
            # if self.args.dataset == 'MNE_alphawaves':
            #     tsne = TSNE(n_components=2, perplexity=20).fit_transform(Fvecs)
            # else:
            #     tsne = TSNE(n_components=2).fit_transform(Fvecs)
            #
            # df = pd.DataFrame()
            # df['y'] = labels.targets
            # df['s'] = labels.subjects
            # df["comp-1"] = tsne[:, 0]
            # df["comp-2"] = tsne[:, 1]
            #
            # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), s=30, palette=sns.color_palette("Set3", 10),
            #                 data=df).set(title="T-SNE visualization of representation")
            # plt.legend(sorted(list(map(float, list(np.unique(df.y))))), loc=2, bbox_to_anchor=(1, 1), title="Target")
            # plt.savefig(os.path.join(self.check_path, 'T-SNE_target_valid.png'), bbox_inches='tight')
            #
            # plt.cla()
            #
            # sns.scatterplot(x="comp-1", y="comp-2", hue=df.s.tolist(), s=30, palette=sns.color_palette("Set3"),
            #                 data=df).set(title="T-SNE visualization of representation")
            # plt.legend(sorted(list(map(float, list(np.unique(df.s))))), loc=2, bbox_to_anchor=(1, 1), title="Subject")
            # plt.savefig(os.path.join(self.check_path, 'T-SNE_subject_valid.png'), bbox_inches='tight')
            #
            # plt.cla()

        return epoch_history

    def train(self):
        steps_per_epoch = len(self.train_loader)

        self.model.train()

        train_loss = 0.0
        correct = 0
        total = 0
        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='cyan')) as pbar:
            for i, data in enumerate(self.train_loader, 0):
                self.optimizer.zero_grad()

                inputs, target, _, _ = data

                inputs = inputs.to(device=self.args.cuda)
                target = target.to(device=self.args.cuda, dtype=torch.long)

                output = self.model(inputs)

                loss = self.criterion(output, target)
                loss.backward()

                self.optimizer.step()

                # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                _, predicted = torch.max(output.data, 1)

                train_loss += loss.item()
                total += target.size(0)
                correct += (predicted == target).sum().item()

                desc = f" Batch [{i + 1:>04}/{len(self.train_loader):>04}"
                pbar.set_description_str(desc)
                pbar.update(1)

            train_loss = train_loss / len(self.train_loader)
            train_acc = 100 * correct / total

        return train_loss, train_acc

    def valid(self):
        steps_per_epoch = len(self.valid_loader)
        self.model.eval()

        valid_loss = 0.0
        correct = 0
        total = 0
        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='yellow')) as pbar:
            with torch.no_grad():
                for i, data in enumerate(self.valid_loader, 0):
                    inputs, target, _, _ = data

                    inputs = inputs.to(device=self.args.cuda)
                    target = target.to(device=self.args.cuda, dtype=torch.long)

                    output = self.model(inputs)

                    loss = self.criterion(output, target)

                    # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                    _, predicted = torch.max(output.data, 1)

                    valid_loss += loss.item()
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                    desc = f" Batch [{i + 1:>04}/{len(self.valid_loader):>04}"
                    pbar.set_description_str(desc)
                    pbar.update(1)

                valid_loss = valid_loss / len(self.valid_loader)
                valid_acc = 100 * correct / total

        return valid_loss, valid_acc

    def test(self):
        steps_per_epoch = len(self.test_loader)
        self.model.eval()

        correct = 0
        total = 0
        targets = []
        preds = []
        probs = []
        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='blue')) as pbar:
            with torch.no_grad():
                for i, data in enumerate(self.test_loader, 0):
                    inputs, target, _, _ = data

                    inputs = inputs.to(device=self.args.cuda)
                    target = target.to(device=self.args.cuda, dtype=torch.long)

                    output = self.model(inputs)
                    prob = output
                    prob = nn.Softmax(dim=1)(prob)

                    _, predicted = torch.max(output.data, 1)

                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                    targets.extend(target.detach().cpu().numpy())
                    preds.extend(predicted.detach().cpu().numpy())
                    probs.extend(prob.detach().cpu().numpy())

                test_acc = 100 * correct / total

                probs_result = probs

                preds = np.array(preds)
                probs = np.array(probs)
                targets = np.array(targets)

                test_f1_score_micro = f1_score(targets, preds, average='micro')
                test_f1_score_macro = f1_score(targets, preds, average='macro')
                test_f1_score_weighted = f1_score(targets, preds, average='weighted')

                if self.args.dataset == 'mit_bih_afib':
                    num = np.unique(targets, axis=0)
                    num = int(np.max(targets) + 1)
                elif self.args.dataset == 'WISDM_at' and self.args.dataset_ver == '2023':
                    num = int(np.max(targets) + 1)
                else:
                    num = np.unique(targets, axis=0)
                    num = num.shape[0]
                targets_onehot = np.eye(num)[targets]

                test_avg_precision = average_precision_score(targets_onehot, probs)

                # precision, recall, _ = precision_recall_curve(targets_onehot, probs, pos_label='your_label')
                #
                # test_auprc = auc(recall, precision)

                preds_result = pd.DataFrame({'target': targets, 'pred': preds, 'prob': probs_result})

            desc = f" Batch [{i + 1:>04}/{len(self.test_loader):>04}"
            pbar.set_description_str(desc)
            pbar.update(1)

        return preds_result, test_acc, test_avg_precision, test_f1_score_micro, test_f1_score_macro, test_f1_score_weighted

    def feature(self, data_loader):
        steps_per_epoch = len(data_loader)
        self.model.eval()

        Fvecs = []
        targets = []
        subjects = []
        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='blue')) as pbar:
            with torch.no_grad():
                for i, data in enumerate(data_loader, 0):
                    inputs, target, _, subject = data

                    inputs = inputs.to(device=self.args.cuda)
                    target = target.to(device=self.args.cuda, dtype=torch.long)

                    fvec = self.model(inputs, True)

                    Fvecs.extend(fvec.detach().cpu().numpy())
                    targets.extend(target.detach().cpu().numpy())
                    subjects.extend(subject.detach().cpu().numpy())

                Fvecs = np.array(Fvecs)
                targets = np.array(targets)
                subjects = np.array(subjects)


            desc = f" Batch [{i + 1:>04}/{len(data_loader):>04}"
            pbar.set_description_str(desc)
            pbar.update(1)

        return Fvecs, targets, subjects




