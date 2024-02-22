import torch
import torch.nn as nn

import argparse
import numpy as np

class ProtoSimLoss_intra_calss(nn.Module):
    def __init__(self, args: argparse):
        super(ProtoSimLoss_intra_calss, self).__init__()
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

        batch_size = X_data.shape[0]

        contrast_count = X_data.shape[1]
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

        # mask
        subject = subject.contiguous().view(-1, 1)  # shape: (128, 1)
        labels = labels.contiguous().view(-1, 1)  # shape: (128, 1)

        subject_mask = torch.eq(subject, torch.FloatTensor(np.array(subject_list))).float().to(self.device)  # torch.Size([128, 36])
        subject_mask = torch.where(subject_mask == 1, 0, 1)  # torch.Size([128, 36])
        subject_mask = subject_mask.repeat(contrast_count, 1)  # 다른 피험자 1 # torch.Size([256, 36])

        labels_mask = torch.eq(labels, torch.FloatTensor(np.array(labels_list))).float().to(self.device)  # torch.Size([128, 36])
        labels_mask = labels_mask.repeat(contrast_count, 1)  # 같은 클래스 1

        mask = subject_mask * labels_mask  # torch.Size([256, 36]) # 같은 클래스이면서 다른 피험자
        mask = mask.detach().cpu().numpy()
        zero_pos_idx = np.where(mask.sum(1) == 0)
        mask = torch.FloatTensor(np.delete(mask, zero_pos_idx, axis=0)).float().to(self.device)

        # compute logits
        sim = torch.matmul(contrast_feature, centroid.T) / density  # torch.Size([256, 36]) -> -1과 1사이로 변환해야 함
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        sim = sim.detach().cpu().numpy()
        sim = torch.FloatTensor(np.delete(sim, zero_pos_idx, axis=0)).float().to(self.device)

        positive_sim = sim * mask
        loss_labels = mask

        # loss = self.criterion(positive_sim, loss_labels) / torch.where(mask.sum(1) == 0, 1, mask.sum(1))  # torch.Size([256])
        loss = self.criterion(positive_sim, loss_labels) / mask.sum(1)
        # loss = torch.nan_to_num(loss)

        loss = loss.view(contrast_count, int(mask.shape[0]/2)).mean()

        return loss