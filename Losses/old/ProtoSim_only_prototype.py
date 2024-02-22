import torch
import torch.nn as nn

import argparse
import numpy as np

class ProtoSimLoss_only_prototype(nn.Module):
    def __init__(self, args: argparse):
        super(ProtoSimLoss_only_prototype, self).__init__()
        self.temperature = args.ProtoSim_temp
        self.base_temperature = args.ProtoSim_base_temp
        self.dens_temperature = args.ProtoSim_dens_temp
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.device = args.cuda

    def forward(self, X_data, subject=None, labels=None, mask=None): # shape: (256, 32)
        subject_np = subject.numpy()
        labels_np = labels.numpy()

        subject_np = np.concatenate((subject_np, subject_np), axis=0)  # (256,)
        labels_np = np.concatenate((labels_np, labels_np), axis=0)  # (256,)

        contrast_feature = torch.cat(torch.unbind(X_data, dim=1), dim=0)  # shape: (256, 32)

        # prototype centroid density
        centroid_list = []
        dist_list = []
        subject_list = []
        labels_list = []
        for i in np.unique(subject_np):
            for j in np.unique(labels_np[np.where(subject_np == i)]).tolist():
                X = contrast_feature[np.where((subject_np == i) & (labels_np == j)), :].squeeze(dim=0)
                cen = torch.mean(X, dim=0)  # cen.shape: (32,)
                dist = (torch.cdist(X, cen.contiguous().view(-1, 1).T).squeeze(dim=1)).tolist()
                centroid_list.append(cen)
                dist_list.append(dist)
                subject_list.append(i)
                labels_list.append(j)

        ## centroid
        centroid = torch.stack(centroid_list)  # (21, 32)
        ## density
        k = len(dist_list)  # ex) k: 21
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

        cen_of_cen = torch.mean(centroid, dim=0)  # torch.Size([32])

        # compute logits
        sim = torch.exp(torch.matmul(centroid, cen_of_cen) / density)  # torch.Size([32])
        sim = sim - sim.max()

        # loss = criterion(positive_sim, loss_labels) / mask.sum(1) # torch.Size([256])
        loss = -sim.mean()

        return loss