import os
import shutil

import torch

from warnings import warn


class Saver:
    def __init__(self, path):
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

            warn(f'{path} does not exist. Creating.')

    def checkpoint(self, tag, payload, is_best=False, is_rep_model=False):
        checkpoint_path = self.get_path(tag)

        if is_rep_model:
            with open(checkpoint_path, "wb+") as fp:
                torch.save(payload, fp)
        else:
            with open(checkpoint_path, "wb+") as fp:
                _payload = payload.state_dict()
                torch.save(_payload, fp)

        if is_best:
            # best_path = f'{checkpoint_path}_best.pt'
            check_path = checkpoint_path[:-3]
            best_path = f'{check_path}_best.pt'
            shutil.copyfile(checkpoint_path, best_path)

    def get_path(self, tag):
        fname = f'{tag}.pt'
        checkpoint_path = os.path.join(self.path, fname)

        return checkpoint_path

    def load(self, tag, dest, is_best=False):
        checkpoint_path = self.get_path(tag)

        if is_best:
            # checkpoint_path = f'{checkpoint_path}_best.pt'
            check_path = checkpoint_path[:-3]
            checkpoint_path = f'{check_path}_best.pt'

        if os.path.exists(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            _payload = dest.state_dict()
            _payload.update(payload)

            dest.load_state_dict(_payload)

        else:
            warn(f'Error: {checkpoint_path} No Weight loaded')