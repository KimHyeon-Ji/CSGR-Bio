from __future__ import print_function

import torch
import torch.nn as nn

import argparse
import numpy as np

class ProtoSimLoss(nn.Module):
    def __init__(self, args: argparse):
        super(ProtoSimLoss, self).__init__()
        self.temperature = args.ProtoSim_temp
        self.base_temperature = args.ProtoSim_base_temp
        self.dens_temperature = args.ProtoSim_dens_temp
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.device = args.cuda

    def forward(self, X_data, subject=None, mask=None): # shape: (256, 32)
        subject_np = subject.numpy()
        batch_size = X_data.shape[0]

        contrast_count = X_data.shape[1]
        contrast_feature = torch.cat(torch.unbind(X_data, dim=1), dim=0)  # shape: (256, 32)

        # prototype centroid density
        centroid_list = []
        dist_list = []
        for i in np.unique(subject_np):
            X = contrast_feature[np.where(subject_np==i), :].squeeze(dim=0) # (개수->10, 32)
            cen = torch.mean(X, dim=0) # cen.shape: (32,)
            dist = (torch.cdist(X, cen.contiguous().view(-1, 1).T).squeeze(dim=1)).tolist()
            centroid_list.append(cen)
            dist_list.append(dist)
        ## centroid
        centroid = torch.stack(centroid_list)  # (21, 32)
        ## density
        k = len(np.unique(subject_np))  # ex) k: 21
        density = np.zeros(k)
        for i, dist in enumerate(dist_list):
            if len(dist) > 1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                density[i] = d
        ## 만약 cluster의 포함되는 관측치가 1개인 경우 최대값으로
        dmax = density.max()  # ex) 0.0777548
        for i, dist in enumerate(dist_list):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10), np.percentile(density, 90)) # clamp extreme values for stability
        density = (self.dens_temperature * density) / density.mean() # scale the mean to temperature
        density = torch.FloatTensor(density).to(self.device)  # shape: (21,)

        # mask
        subject = subject.contiguous().view(-1, 1)  # shape: (128, 1)
        mask = torch.eq(subject, torch.unique(subject)).float().to(self.device)
        mask = torch.where(mask+1 < 2, mask+1, torch.zeros_like(mask))  # shape: (128, 21)
        # tile mask
        mask = mask.repeat(contrast_count, 1)  # shape: (256, 21)

        # compute logits
        sim = (torch.matmul(contrast_feature, centroid.T) / (
                    torch.norm(contrast_feature) * torch.norm(centroid.T))) / density  # torch.Size([256, 21])

        # for numerical stability
        positive_sim = sim[torch.gt(mask, 0)].reshape(batch_size * 2, -1)  # torch.Size([256, 20])

        labels = torch.ones_like(positive_sim)  # torch.Size([256, 20])

        loss = self.criterion(positive_sim, labels) / mask.sum(1)  # torch.Size([256])
        loss = loss.view(contrast_count, batch_size).mean()

        return loss