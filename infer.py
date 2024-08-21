from arguments import get_arguments
from utils.ckpt import load_checkpoint
from torch.utils.data import DataLoader
from utils.util import get_luminance
from model.original import DRLN
from model.swin import swinir_builder as SwinIR
from utils.trainer import testModel
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from ignite.metrics import PSNR
from ignite.metrics import SSIM
from dataset import Banchmark
import pandas as pd
import gc
import numpy as np
import glob
import torch
import torch_pruning as tp
import os

from tqdm import tqdm 
from utils.trainer import ForwardManager

ORIGINAL_DRLN = "/trinity/home/mangelini/data/mangelini/Pruning/Train/Train_originalx|scale|_e200_FT200/checkpoints/unpruned_finetuned_model.pth" 
ORIGINAL_SWIN = "/trinity/home/mangelini/data/mangelini/Pruning/Train/Train_swinir_cx|scale|_e200_FT200/checkpoints/unpruned_finetuned_model.pth" 

def get_upscaling_rate(path):
    if 'x2,' in path: return 2
    elif 'x3,' in path: return 3
    elif 'x4,' in path: return 4
    elif 'x8,' in path: return 8
    return 1


if __name__ == '__main__':

    args = get_arguments()
    experiment_path = args.experiment_path
    inference_out_path = f"{experiment_path}/Infer"
    profiler_out_path = f"{experiment_path}/Stats2"

    os.makedirs(inference_out_path, exist_ok=True)
    os.makedirs(profiler_out_path, exist_ok=True)

    # create the model
    useBGR=False
    orignial_path = ""
    model_name = "original" if  "original" in experiment_path else "swinir"
    if (model_name == "original"):
        model = DRLN(int(args.scale))
        orignial_path = ORIGINAL_DRLN.replace("|scale|", f"{args.scale}")
    elif (model_name == "swinir"):
        model = SwinIR(args)
        orignial_path = ORIGINAL_SWIN.replace("|scale|", f"{args.scale}")
        useBGR=True
    else:
        raise Exception("Invalid model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Dataset
    dataset_test = Banchmark(scaling=args.scale, useBGR=useBGR, take=args.infer_count, shuffle=True)
    test_loader = DataLoader(dataset_test, shuffle=False, batch_size=1, num_workers=1, pin_memory=True, worker_init_fn = lambda id: np.random.seed(id))


    original_model_path = [orignial_path]
    rest = glob.glob(f"{experiment_path}/checkpoints/pruned_iteraion_*.pth")
    rest.sort()
    ckpts = original_model_path + rest
    for idx, prune_step_ckpt in enumerate(ckpts):
        
        name = ""
        # Load pretrained weights
        model_dict, epoch, mse = load_checkpoint(prune_step_ckpt)
        if model_dict is None: raise Exception("The ckpt dose not have the model state_dict!")
        # Need to use the torch_pruning model loader to handled the pruned model parts
        if "unpruned" in prune_step_ckpt: 
            model.load_state_dict(model_dict['model']) #original Model Case
            name = "unpruned"
        else: 
            tp.load_state_dict(model, state_dict=model_dict['model']) # If not original we need to use the torch_pruning loader
            name = int(os.path.basename(prune_step_ckpt).replace(".pth", "").replace("pruned_iteraion_", ""))

        if name != 8: continue # Do only fully pruned and unpruned
        
        # To handle new torch versions
        if "swinir" in experiment_path :
            # replace all torch-10 GELU's by torch-12 GELU
            def torchmodify(l_name) :
                a=l_name.split('.')
                for i,s in enumerate(a) :
                    if s.isnumeric() :
                        a[i]="_modules['"+s+"']"
                return '.'.join(a)
            import torch.nn as nn
            for m_name, module in model.named_modules() :
                if isinstance(module,nn.GELU) :
                    exec('model.'+torchmodify(m_name)+'=nn.GELU(approximate=\'tanh\')')

        model_upscale_rate = get_upscaling_rate(experiment_path)
        model = model.to(device)

        # Folder SetUp
        profiler_folder = f"{profiler_out_path}/{name}"
        out_folder = f"{inference_out_path}/{name}"
        os.makedirs(profiler_folder, exist_ok=True)
        os.makedirs(out_folder, exist_ok=True)
        
    
        # Eval Original Model on The TestSet
        psnr_pretrain = PSNR(data_range=args.data_range, output_transform=get_luminance, device=device)
        ssim_pretrain = SSIM(data_range=args.data_range, output_transform=get_luminance, device=device)
        
        if args.infer_count > 0 :
            test_loss, test_table = testModel(
                loader=test_loader, 
                model=model,
                args=args,
                psnr=psnr_pretrain, 
                ssim=ssim_pretrain,
                data_range=args.data_range,
                device=device,
                out=out_folder
            )
        
        metrics = {}
        if args.infer_count > 0 :
            metrics["mse"] = test_loss
            metrics['ssim'] = float(ssim_pretrain.compute())
            metrics['psnr'] = float(psnr_pretrain.compute())


        print(metrics)



