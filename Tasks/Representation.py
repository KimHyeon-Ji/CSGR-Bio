import torch.optim
import torch

import os
import torch.nn as nn
from pytorch_metric_learning import losses
import pandas as pd
import numpy as np
import tqdm
import argparse
import time

from Utils.Saver import Saver
from Utils.Loggers import get_tqdm_config
from Utils.Summary import TensorboardSummary
from Dataloaders.Dataloader import make_dataloader
from Models.Backbone.LSTM_FCN import *
from Models.Representation_Model import *
from Losses.Inter_sub_sim_v1 import *
from Losses.Inter_sub_sim import *
from Losses.Inter_sub_sim_v2 import *


class Representation_Trainer(object):
    def __init__(self, args: argparse, check_path: str):

        self.args = args
        self.check_path = check_path

        # Define DataLoader
        kwargs = {'pin_memory': True, 'num_workers': 8}

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
        self.LSTM_FCNs = LSTM_FCNs(output_dim=args.LSTM_FCN_output_dim, num_features=self.num_features,
                                   num_layers=args.LSTM_FCN_num_layer, lstm_drop_p=args.LSTM_FCN_lstm_drop_out,
                                   fc_drop_p=args.LSTM_FCN_fc_drop_out)

        backbone = [self.LSTM_FCNs, args.LSTM_FCN_output_dim]

        model = Rep_Model(backbone=backbone, head=args.Rep_head, output_dim=args.Rep_output_dim)

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

        self.supconloss = losses.SupConLoss(temperature=self.args.SupCon_temp).to(device=args.cuda)

        if args.sub_loss == 'Inter_sub_sim':
            self.protosimloss = Inter_sub_sim(args=self.args).to(device=args.cuda)
        elif args.sub_loss == 'Inter_sub_sim_v1':
            self.protosimloss = Inter_sub_sim_v1(args=self.args).to(device=args.cuda)
        elif args.sub_loss == 'Inter_sub_sim_v2':
            self.protosimloss = Inter_sub_sim_v2(args=self.args).to(device=args.cuda)
        

        self.start_loss_ratio = args.start_loss_ratio
        self.finish_loss_ratio = args.finish_loss_ratio


    def run(self, epochs):

        with tqdm.tqdm(**get_tqdm_config(total=epochs, leave=True, color='red')) as pbar:

            loss_ratio = self.start_loss_ratio
            rate_of_increase = (1 - (self.start_loss_ratio + (1 - self.finish_loss_ratio))) / epochs

            best_loss = float('inf')

            for epoch in range(1, epochs + 1):

                start_time = time.monotonic()

                # train
                train_loss, train_supcon_loss, train_protosim_loss = self.train(loss_ratio=loss_ratio)

                loss_ratio += rate_of_increase

                self.scheduler.step()

                epoch_history = {'Rep_Loss': {'train': train_loss},
                                 'Supcon_Loss': {'supcon': train_supcon_loss},
                                 'ProtoSim_Loss': {'proto': train_protosim_loss},
                                 'Loss_Ratio': {'loss_ratio': loss_ratio}}

                # Tensorboard summary
                for metric_name, metric_dict in epoch_history.items():
                    self.writer.add_scalars(
                        main_tag=metric_name,
                        tag_scalar_dict=metric_dict,
                        global_step=epoch
                    )

                # Save model if it is the current best
                current_loss = epoch_history['Rep_Loss']['train']
                if current_loss < best_loss:
                    best_loss = current_loss
                    self.saver.checkpoint('best_representation', self.model, is_rep_model=True, is_best=True)

                # Logging
                desc = f" Epoch [{epoch:>04}/{epochs:>04} |"
                for metric_name, metric_dict in epoch_history.items():

                    for k, v in metric_dict.items():
                        desc += f" {k}_{metric_name}: {v:.4f} |"


                pbar.set_description_str(desc)
                pbar.update(1)

            # Save the last model
            self.saver.checkpoint('representation_model', self.model, is_rep_model=True)

        return epoch_history

    def train(self, loss_ratio):
        steps_per_epoch = len(self.train_loader)

        self.model.train()

        train_loss = 0.0
        train_supcon_loss = 0.0
        train_protosim_loss = 0.0
        with tqdm.tqdm(**get_tqdm_config(total=steps_per_epoch, leave=False, color='cyan')) as pbar:
            for i, data in enumerate(self.train_loader, 0):
                self.optimizer.zero_grad()

                inputs, target, inputs_aug, subject = data

                inputs = inputs.to(device=self.args.cuda)
                inputs_aug = inputs_aug.to(device=self.args.cuda)
                target = target.to(device=self.args.cuda, dtype=torch.long)
                subject = subject.to(device=self.args.cuda, dtype=torch.long)

                inputs = inputs.to(device=self.args.cuda)
                inputs_aug = inputs_aug.to(device=self.args.cuda)

                new_inputs = torch.cat([inputs, inputs_aug], dim=0)

                bsz = target.shape[0]

                features = self.model(new_inputs)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

                if self.args.rep == 'supcon':
                    loss = self.supconloss(contrast_feature, torch.cat((target,target)))
                elif self.args.rep == 'intersubsim':
                    loss = self.protosimloss(features, subject, target)
                elif self.args.rep == 'supcon+intersubsim':
                    supcon_loss = self.supconloss(contrast_feature, torch.cat((target,target)))
                    protosim_loss = self.protosimloss(features, subject, target)

                    loss = (loss_ratio * supcon_loss) + ((1 - loss_ratio) * protosim_loss)

                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()

                if self.args.rep == 'supcon+intersubsim':
                    train_supcon_loss += supcon_loss
                    train_protosim_loss += protosim_loss

                desc = f" Batch [{i + 1:>04}/{len(self.train_loader):>04}"
                pbar.set_description_str(desc)
                pbar.update(1)

            train_loss = train_loss / len(self.train_loader)
            train_supcon_loss = train_supcon_loss / len(self.train_loader)
            train_protosim_loss = train_protosim_loss / len(self.train_loader)

        return train_loss, train_supcon_loss, train_protosim_loss