import torch.optim

import os
import pandas as pd
import numpy as np
import tqdm
import argparse
from Utils.Saver import Saver
from Utils.Loggers import get_tqdm_config
from Utils.Summary import TensorboardSummary
from Dataloaders.Dataloader import make_dataloader
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from Models.Backbone.LSTM_FCN import *
from Models.Classification_Model import *

class DS_Classification_Trainer(object):
    def __init__(self, args: argparse, check_path: str):

        self.args = args
        self.check_path = check_path

        # Define DataLoader
        kwargs = {'pin_memory': True, 'num_workers': 0}
        self.train_loader, self.valid_loader, self.test_loader = make_dataloader(args, **kwargs)

        # Define Saver
        self.saver = Saver(path=check_path)

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(directory=self.saver.path)
        self.writer = self.summary.create_summary()

        # Dataset Setting
        if args.dataset == 'mit_bih_afib':
            self.num_features = args.mit_bih_afib_num_features
            self.num_classes = args.mit_bih_afib_num_classes
            self.window_size = args.mit_bih_afib_window_size
            self.FCN_kernel_size = args.mit_bih_afib_FCN_kernel_size
            self.FCN_stride = args.mit_bih_afib_FCN_stride
        elif args.dataset == 'MyoArmband':
            self.num_features = args.MyoArmband_num_features
            self.num_classes = args.MyoArmband_num_classes
            self.window_size = args.MyoArmband_window_size
            self.FCN_kernel_size = args.MyoArmband_FCN_kernel_size
            self.FCN_stride = args.MyoArmband_FCN_stride
        elif args.dataset == 'UniMiB_adl':
            self.num_features = args.UniMiB_adl_num_features
            self.num_classes = args.UniMiB_adl_num_classes
            self.window_size = args.UniMiB_adl_window_size
            self.FCN_kernel_size = args.UniMiB_adl_FCN_kernel_size
            self.FCN_stride = args.UniMiB_adl_FCN_stride

        # Define Model
        self.trained_rep = torch.load(f'{self.check_path}/representation_model.pt')
        if args.task == 'rep_frozen':
            for param in self.trained_rep.parameters():
                param.requires_grad = False
        elif args.task == 'rep_fine_tuning':
            pass

        backbone = [self.trained_rep, args.Rep_output_dim]

        model = Classifier(backbone=backbone, rep_dim=args.LSTM_FCN_output_dim, num_classes=self.num_classes)

        # Using Cuda
        self.model = model.to(device=args.cuda)

        # Define Optimizer lr_scheduler & Criterion
        if args.task == 'rep_frozen':
            if args.optim == 'Adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.ds_lr, weight_decay=args.weight_decay)
            elif args.optim == 'SGD':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.ds_lr, weight_decay=args.weight_decay)
            elif args.optim == 'RMSprop':
                self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.ds_lr, weight_decay=args.weight_decay)
            else:
                raise ValueError('In-valid optimizer choice')
        elif args.task == 'rep_fine_tuning':
            if args.optim == 'Adam':
                self.optimizer = torch.optim.Adam(model.parameters(), lr=args.ds_lr, weight_decay=args.weight_decay)
            elif args.optim == 'SGD':
                self.optimizer = torch.optim.SGD(model.parameters(), lr=args.ds_lr, weight_decay=args.weight_decay)
            elif args.optim == 'RMSprop':
                self.optimizer = torch.optim.RMSprop(model.parameters(), lr=args.ds_lr, weight_decay=args.weight_decay)
            else:
                raise ValueError('In-valid optimizer choice')

        if args.lr_scheduler == 'LambdaLR':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        elif args.lr_scheduler == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=10, gamma=0.5)
        elif args.lr_scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=30, eta_min=0)
        elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10, T_mult=1, eta_min=0.00001)
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
                    'Cls_Loss' : {
                        'train' : train_loss,
                        'valid' : valid_loss
                    },
                    'Cls_Acc': {
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
                current_valid_loss = epoch_history['Cls_Loss']['valid']
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
            result = pd.DataFrame({'valid_best_loss': valid_best_loss, 'test_acc': test_acc, 'test_avg_precision': test_avg_precision,
                                   'test_f1_score_micro': test_f1_score_micro, 'test_f1_score_macro' : test_f1_score_macro,
                                   'test_f1_score_weighted': test_f1_score_weighted}, index = [0])
            result.to_csv(os.path.join(self.check_path, f'result.csv'), index=False)

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
                else:
                    num = np.unique(targets, axis=0)
                    num = num.shape[0]
                targets_onehot = np.eye(num)[targets]

                test_avg_precision = average_precision_score(targets_onehot, probs)

                preds_result = pd.DataFrame({'target': targets, 'pred': preds, 'prob': probs_result})

            desc = f" Batch [{i + 1:>04}/{len(self.test_loader):>04}"
            pbar.set_description_str(desc)
            pbar.update(1)

        return preds_result, test_acc, test_avg_precision, test_f1_score_micro, test_f1_score_macro, test_f1_score_weighted