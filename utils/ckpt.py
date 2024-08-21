import re
import os
import torch
import glob2

def get_epoch_from_name(ckpt_url):
            s = re.findall("ckpt_e(\d+).pth", ckpt_url)
            epoch = int(s[0]) if s else -1
            return epoch, ckpt_url

def load_checkpoint(ckpt_path):
    if os.path.isdir(ckpt_path):
        ckpts = glob2.glob(os.path.join(ckpt_path, '/**/*.pth'))
        assert ckpts, "No checkpoints to resume from!"

        # load checkpoint with highest epoch
        start_epoch, ckpt = max(get_epoch_from_name(c) for c in ckpts)
        ckpt_path = ckpt

    ckpt = torch.load(ckpt_path)

    if 'params' in ckpt.keys():
        return (ckpt['params'], 0, 10000000000.0)

    if not hasattr(ckpt, 'model'):
        return (ckpt, 0, 10000000000.0)
        
    state_dict = ckpt['model']
    if hasattr(state_dict, 'model'): state_dict = state_dict['model']
    if hasattr(state_dict, 'model'): state_dict = state_dict['model']
    start_epoch = ckpt['epoch'] if hasattr(ckpt, 'epoch') else 0
    best_mse = ckpt['mse_val'] if hasattr(ckpt, 'mse_val') else 10000000000.0

    return (state_dict, start_epoch, best_mse)