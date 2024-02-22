from __future__ import print_function

import torch
import torch.nn as nn

import argparse
import numpy as np

class SubLoss(nn.Module):
    def __init__(self, args: argparse):
        super(SubLoss, self).__init__()
        self.temperature = args.ProtoSim_temp
        self.base_temperature = args.ProtoSim_base_temp
        self.dens_temperature = args.ProtoSim_dens_temp

        self.device = args.cuda

    def forward(self, X_data, subject=None, mask=None): # shape: (256, 32)
        batch_size = X_data.shape[0] # 128
        subject_np = subject.numpy()

        if subject is not None and mask is not None:
            raise ValueError('Cannot define both `subject` and `mask`')
        elif subject is None and mask is None:
            instance_mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif subject is not None:  # 현재 여기에 해당
            subject = subject.contiguous().view(-1, 1)
            if subject.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            # 같은 피험자면 1, 다른 피험자면 0
            instance_mask = torch.eq(subject, subject.T).float().to(self.device)

            # mask: centroid 계산을 위한, 같은 피험자면 0, 다른 피험자면 1
            mask = torch.eq(subject, torch.unique(subject)).float().to(self.device)
            mask = torch.where(mask + 1 < 2, mask + 1, torch.zeros_like(mask))  # shape: (128, 21)

        else:
            instance_mask = mask.float().to(self.device)

        contrast_count = X_data.shape[1] # 2
        contrast_feature = torch.cat(torch.unbind(X_data, dim=1), dim=0)  # shape: (256, 32)

        # prototype centroid density
        centroid_list = []
        for i in np.unique(subject_np):
            X = contrast_feature[np.where(subject_np==i), :].squeeze(dim=0) # (개수->10, 32)
            cen = torch.mean(X, dim=0) # cen.shape: (32,)
            centroid_list.append(cen)
        ## centroid
        centroid = torch.stack(centroid_list)  # (21, 32)

        # tile mask
        mask = mask.repeat(contrast_count, 1)  # shape: (256, 21)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, centroid.T),
            self.temperature)
            # contrast_feature.shape: (256, 32) / centroid.shape: (32, 21)
            # anchor_dot_contrast.shape: (256, 21)

        masked_anchor_dot_contrast = anchor_dot_contrast * mask  # torch.Size([256, 21])
        max_masked_anchor_dot_contrast, _ = torch.max(masked_anchor_dot_contrast, dim=1, keepdim=True)
        pos = torch.exp(max_masked_anchor_dot_contrast)  # torch.Size([256, 1])

        ##==============================================================================================

        contrast_count_ins = X_data.shape[1]  # 2
        contrast_feature_ins = torch.cat(torch.unbind(X_data, dim=1), dim=0)  # shape: torch.Size([256, 32])

        anchor_feature_ins = contrast_feature_ins
        anchor_count_ins = contrast_count_ins

        # tile mask
        instance_mask = instance_mask.repeat(anchor_count_ins, contrast_count_ins)

        # compute logits
        anchor_dot_contrast_ins = torch.div(torch.matmul(anchor_feature_ins, contrast_feature_ins.T), self.temperature)

        # 다른 피험자는 0이 곱해져서 없어짐
        masked_anchor_dot_contrast_ins = anchor_dot_contrast_ins * instance_mask  # torch.Size([256, 256])
        # 가장 가까운 centroid 보다 거리가 가까우면 남기기 # torch.Size([256, 256])
        masked_anchor_dot_contrast_ins = torch.where(masked_anchor_dot_contrast_ins > max_masked_anchor_dot_contrast,
                                                     masked_anchor_dot_contrast_ins,
                                                     torch.zeros_like(masked_anchor_dot_contrast_ins))

        # logit에 곱해줄 mask: 가장 가까운 centroid 보다 거리가 먼 경우 0
        logits_ins_mask = torch.where(masked_anchor_dot_contrast_ins == 0,
                                      torch.zeros_like(masked_anchor_dot_contrast_ins),
                                      torch.ones_like(masked_anchor_dot_contrast_ins))  # torch.Size([256, 256])

        exp_ins_logits = torch.exp(masked_anchor_dot_contrast_ins) * logits_ins_mask
        neg = exp_ins_logits.sum(1, keepdim=True)

        loss = (- torch.log(pos / (pos + neg))).mean()

        return loss