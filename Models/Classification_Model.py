import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, backbone, rep_dim, num_classes):
        super(Classifier, self).__init__()
        model_fun, dim_in = backbone
        self.encoder = model_fun

        self.fc1 = nn.Linear(dim_in, rep_dim)
        self.fc2 = nn.Linear(rep_dim, num_classes)


    def forward(self, x, return_rep=False):
        x = self.encoder(x)

        if not return_rep:
            x = self.fc1(x)
            x = self.fc2(x)
            x = F.log_softmax(x) # F.softmax(x)
            return x
        else:
            x = self.fc1(x)
            return x
