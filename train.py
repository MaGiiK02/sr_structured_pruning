import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch_pruning as tp
import numpy as np
import random
import time
from tqdm import tqdm
from ignite.metrics import PSNR
from ignite.metrics import SSIM

# Model Variants
from model.original import DRLN
from model.deepcamera_v1 import DRLN as DRLNDeepCameraV1
from model.deepcamera_v2 import DRLN as DRLNDeepCameraV2
from model.swin import swinir_builder as SwinIR

from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from utils.ckpt import load_checkpoint
from utils.util import get_luminance
from dataset import Div2KTrain, Banchmark, CropDataset
from utils.trainer import trainFor, testModel, sparsityLearning
from arguments import get_arguments
from pruners import pruner_constructor

# transforms that works with pure matrix
from utils.transforms import DescreteRandomRotation

def generateRunName(args):
    tags = []
    model = args.model
    epoch = args.epochs
    prune_upscale = args.prune_upscale
    swinir_config = args.swinir_config
    large_model = args.large_model
    train_mode = f"FT{epoch}" if args.weigths else f"RE{epoch}"
    
    if (model.lower() == 'swinir'):
        large = "L" if large_model else ""
        model = f"{model}_{swinir_config[0]}{large}"

    # Constract name
    name = f"Train_{model}x{args.scale}_e{epoch}_{train_mode}"

    return name, tags

if __name__ == '__main__':
    args = get_arguments()
    
    print(args)

    # SetUp Random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    ### Prepare output Dirs
    params = vars(args)
    params['dataset'] = os.path.basename(os.path.normpath(args.loader))

    run_name, tags = generateRunName(args)
    run_dir = os.path.join(args.runs, run_name)
    print("Outupt Folder root:", run_dir)
    log_metrics_path = os.path.join(run_dir, 'pruning_results.csv')
    original_test_path = os.path.join(run_dir, 'original')
    checkpoints_path = os.path.join(run_dir, 'checkpoints')
    param_file = os.path.join(run_dir, 'params.csv')
    os.makedirs(original_test_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    
    pd.DataFrame(params, index=[0]).to_csv(param_file, index=False)

    ##############################
    ## Experiment Tracker ########
    ############################## 
    wandb.init(
        mode="disabled", 
        project="SuperResolution Pruning (Thesis)", 
        entity="deepcamera",
        name=run_name,
        tags=tags
    )
    wandb.config = args
    
    
    # create the model
    useBGR=False
    if (args.model.lower() == 'original'):
        model = DRLN(int(args.scale))
    elif (args.model.lower() == 'deepcamera_v1'):
        model = DRLNDeepCameraV1(int(args.scale))
    elif (args.model.lower() == 'deepcamera_v2'):
        model = DRLNDeepCameraV2(int(args.scale))
    elif (args.model.lower() == 'swinir'):
        model = SwinIR(args)
        useBGR = True
    else:
        raise Exception("Invalid model")

    # Dataset Set Up ###########################################

    ############ Image data augmentation for retrain ###########
    transform = transforms.Compose([
        DescreteRandomRotation(angles=[0, 90, 180, -90]),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
    ])

    # Select the Data loader for the datasets (biggest difference is the folder structure and the loading process)
    if (args.loader.lower() == 'div2k_rgb'):
        dataset = Div2KTrain(scaling=args.scale, useBGR=useBGR)
    else:
        raise Exception("Invalid DataLoader")
    

    # Split the dataset in train and validation
    train_percentual = 0.8
    train_num = round(len(dataset) * train_percentual)
    val_num = round(len(dataset) * (1-train_percentual))
    train_data, val_data = random_split(dataset, [train_num, val_num], generator=torch.Generator().manual_seed(args.seed))
    train_data = CropDataset(train_data, scaling=args.scale, transform=transform, seed=args.seed, crop=args.crop, cropMode="random", noiseInensity=args.noise)
    val_data = CropDataset(val_data, scaling=args.scale, transform=None, seed=None, crop=args.crop, cropMode="center")
    
    # create the loader for the training set
    train_loader = DataLoader(train_data, shuffle=True,  batch_size=args.batch, num_workers=1, pin_memory=True, worker_init_fn = lambda id: np.random.seed(id))
    # create the loader for the validation set (to select the model after prune)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch, num_workers=1, pin_memory=True, worker_init_fn = lambda id: np.random.seed(id))
        
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained weights
    if args.weigths:
        model_dict, epoch, mse = load_checkpoint(args.weigths)
        if model_dict is None: raise Exception("The ckpt dose not have the model state_dict!")
        model.load_state_dict(model_dict, strict=True)
    
    model = model.to(device)

    (best_model, best_mse, last_epoch_model, last_epoch_mse, optimizer, run_logs) = trainFor(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        args=args,
        epochs=args.epochs,
        run_folder=original_test_path
    )
    model.load_state_dict(best_model.state_dict()) # Load weights of the best model!!
    wandb.log({"train_logs":  wandb.Table(dataframe=run_logs)})
    run_logs.to_csv(os.path.join(original_test_path, f'traning_log.csv'),index=False)
    
    ckpt_path = os.path.join(checkpoints_path, 'unpruned_finetuned_model.pth')
    torch.save({
        'model': model.state_dict(),
    }, ckpt_path) 

