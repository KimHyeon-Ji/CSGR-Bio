from __future__ import print_function

import torch
import torch.nn as nn

import argparse

class SupConLoss(nn.Module):
    def __init__(self, args: argparse):
        super(SupConLoss, self).__init__()
        self.temperature = args.SupCon_temp
        self.base_temperature = args.SupCon_base_temp
        self.contrast_mode = args.SupCon_contrast_mode

        self.device = args.cuda

    def forward(self, X_data, labels=None, mask=None):
        # X_data.shape: (128, 2, 32)
        # labels.shape: (128,)
        batch_size = X_data.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None: # 현재 코드 여기에 해당
            labels = labels.contiguous().view(-1, 1)  # shape: (128, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device) # 요소 별 동등성 계산 -> 레이블이 같으면 1, 다르면 0
        else:
             mask = mask.float().to(self.device)

        # mask.shape: (128, 128)

        contrast_count = X_data.shape[1] # contrast_count: 2
        contrast_feature = torch.cat(torch.unbind(X_data, dim=1), dim=0)  # contrast_featrue.shape: (256,32)
        if self.contrast_mode == 'one':
            anchor_feature = X_data[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # anchor_dot_contrast.shape: (256, 256)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # logits_max.shape: (256, 1)
        logits = anchor_dot_contrast - logits_max.detach()  # logits.shape: (256, 256)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # mask.shape: (256, 256)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),  # torch.ones_like(mask).shape: (256, 256)
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device), # shape: (256, 1) -> 그냥 0부터 255
            0
        )  # logits_mask.shape: (256, 256)
        mask = mask * logits_mask  # mask.shape: (256, 256)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # exp_logis.shape: (256, 256) # logits mask self-contrast만 0이라서 logit이 0이 됨
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # log_prob.shape: (256, 256)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # mean_log_prob_pos.shape: (256,)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # loss.shape: (256,)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss