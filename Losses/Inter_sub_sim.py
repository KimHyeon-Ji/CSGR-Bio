import torch
import torch.nn as nn

import argparse
# ProtoSimLoss_intra_calss

class Inter_sub_sim(nn.Module):
    def __init__(self, args: argparse):
        super(Inter_sub_sim, self).__init__()
        self.temperature = args.Intersubsim_temp
        self.base_temperature = args.Intersubsim_base_temp
        self.dens_temperature = args.Intersubsim_dens_temp
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.device = args.cuda

    def forward(self, X_data, subject=None, labels=None, mask=None):
        """
        Args:
            X_data: hidden vector of shape [bsz, hidden vector dim]
            subject: subject of shape [bsz]
            labels: ground truth of shape [bsz]
            mask: mask of shape [bsz, bsz]
        Returns:
            A loss scalar
        """
        subject_ten = torch.cat((subject, subject), dim=0)
        labels_ten = torch.cat((labels, labels), dim=0)

        contrast_count = X_data.shape[1]
        contrast_feature = torch.cat(torch.unbind(X_data, dim=1), dim=0) 

        ## prototype centroid density
        centroid_list = []
        dist_list = []
        subject_list = []
        labels_list = []
        for i in range(torch.unique(subject_ten).shape[0]):
            sub = int(torch.unique(subject_ten)[i])
            for j in range(torch.unique(labels_ten[torch.where(subject_ten == sub)]).shape[0]):
                lab = int(torch.unique(labels_ten[torch.where(subject_ten == sub)])[j])
                X = contrast_feature[torch.where((subject_ten == sub) & (labels_ten == lab))[0], :].squeeze(dim=0)
                cen = torch.mean(X, dim=0) 
                dist = (torch.cdist(X, cen.contiguous().view(-1, 1).T).squeeze(dim=1)).tolist()
                centroid_list.append(cen)
                dist_list.append(dist)
                subject_list.append(i)
                labels_list.append(j)

        ## centroid
        centroid = torch.stack(centroid_list)  
        
        ## density
        k = len(dist_list)  
        density = torch.zeros(k).to(self.device)
        for i, dist in enumerate(dist_list):
            if len(dist) > 1:
                d = (torch.as_tensor(dist).to(self.device)**0.5).mean()/torch.log(torch.tensor(len(dist)+10).to(self.device))
                density[i] = d
        
        dmax = density.max()  # If the cluster contains 1 observation, assign it as the maximum value
        for i, dist in enumerate(dist_list):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(torch.quantile(density, 0.1),
                               torch.quantile(density, 0.9))  # clamp extreme values for stability
        density = (self.dens_temperature * density) / density.mean()  # scale the mean to temperature

        ## mask
        subject = subject.contiguous().view(-1, 1) 
        labels = labels.contiguous().view(-1, 1)  

        subject_mask = torch.eq(subject, torch.tensor(subject_list).to(self.device)).float() 
        subject_mask = torch.where(subject_mask == 1, 0, 1)  
        subject_mask = subject_mask.repeat(contrast_count, 1) 

        labels_mask = torch.eq(labels, torch.tensor(labels_list).to(self.device)).float() 
        labels_mask = labels_mask.repeat(contrast_count, 1)  

        mask = subject_mask * labels_mask  
        not_zero_idx = torch.where(mask.sum(1) != 0)
        mask = torch.index_select(mask, dim=0, index=not_zero_idx[0])

        ## compute logits
        sim = torch.matmul(contrast_feature, centroid.T) / density 
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max
        sim = torch.index_select(sim, dim=0, index=not_zero_idx[0])

        positive_sim = sim * mask
        loss_labels = mask

        loss = self.criterion(positive_sim, loss_labels) / mask.sum(1)

        loss = loss.view(contrast_count, int(mask.shape[0]/2)).mean()

        return loss