import torch.nn as nn
import torch.nn.functional as F

class Rep_Model(nn.Module):
    def __init__(self, backbone, head, output_dim):
        super(Rep_Model, self).__init__()
        model_fun, dim_in = backbone
        self.encoder = model_fun

        if head == 'linear':
            self.head = nn.Linear(dim_in, output_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, output_dim)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x