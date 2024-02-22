import os
import json
import argparse
import random
import numpy as np
import torch

from Tasks.Classification import Classification_Trainer
from Tasks.Representation import Representation_Trainer
from Tasks.Downstream_Classification import DS_Classification_Trainer
from Utils.Args import parser
from Utils.Path import mypath

from itertools import product

import warnings
warnings.filterwarnings(action='ignore')


def my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

def main(args: argparse):
    args.dataset = 'mit_bih_afib'
    args.sub_loss = 'protosim_intra_class'
    args.task = 'rep_frozen'
    args.rep = 'supcon+protosim'

    args.start_loss_ratio = 0.3
    args.finish_loss_ratio = 0.7

    save_path = mypath(args=args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(os.path.join(save_path, 'arg_parser.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    Re_Trainer = Representation_Trainer(args=args, check_path=save_path)
    Re_Trainer.run(epochs=args.epochs)

    Trainer = DS_Classification_Trainer(args=args, check_path=save_path)
    Trainer.run(epochs=args.epochs)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    main(args)