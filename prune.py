import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch_pruning as tp
import numpy as np
import random
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
    prunng_method = args.pruning_method
    pruning_different_layer_rateo = args.pruning_different_layer_rateo
    pruning_different_layer_rateo_tag = 'groupwise-rateo' if args.pruning_different_layer_rateo else 'uniform-rateo'
    pruning_steps = args.pruning_steps
    pruning_target_ratio = args.pruning_target_ratio
    global_pruning = args.use_global_pruning
    prune_upscale = args.prune_upscale
    noise = args.noise
    swinir_config = args.swinir_config
    large_model = args.large_model
    
    if (model.lower() == 'swinir'):
        large = "L" if large_model else ""
        model = f"{model}_{swinir_config[0]}{large}"
        
    # Construct tags
    tags+= [
        model,
        prunng_method,
        pruning_different_layer_rateo_tag,
    ]
    
    # optional
    if args.use_global_pruning:
        tags+= ['global-pruning']
        
    if prune_upscale:
        tags+= ['prune-upscale']
    
    # Constract name
    name = f"Pruning({model}x{args.scale}, {prunng_method}({pruning_steps},{pruning_target_ratio},{pruning_different_layer_rateo},{global_pruning},{prune_upscale}))-RetrainedOn(DIV2K-{args.epochs}-{args.crop}x{args.crop}-N{noise})"

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
        dataset_test = Banchmark(scaling=args.scale, useBGR=useBGR)
    else:
        raise Exception("Invalid DataLoader")
    
    if args.test_mode:
        use_only = 0.2
        
        kept_num = round(len(dataset) * use_only)
        igonored_num = round(len(dataset) * (1-use_only))
        dataset, _ = random_split(dataset, [kept_num, igonored_num], generator=torch.Generator().manual_seed(args.seed))

        use_only = 0.01
        kept_num = round(len(dataset_test) * use_only)
        igonored_num = round(len(dataset_test) * (1-use_only))
        dataset_test, _ = random_split(dataset_test, [kept_num, igonored_num], generator=torch.Generator().manual_seed(args.seed))

    #Split the dataset in train and validation
    train_percentual = 0.8
    train_num = round(len(dataset) * train_percentual)
    val_num = round(len(dataset) * (1-train_percentual))
    train_data, val_data = random_split(dataset, [train_num, val_num], generator=torch.Generator().manual_seed(args.seed))
    train_data = CropDataset(train_data, scaling=args.scale, transform=transform, seed=args.seed, crop=args.crop, cropMode="random", noiseInensity=args.noise)
    val_data = CropDataset(val_data, scaling=args.scale, transform=None, seed=None, crop=args.crop, cropMode="center")
    
    # create the loader for the training set
    train_loader = DataLoader(train_data, shuffle=True,  batch_size=args.batch, num_workers=0, pin_memory=True, worker_init_fn = lambda id: np.random.seed(id))
    # create the loader for the validation set (to select the model after prune)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch, num_workers=0, pin_memory=True, worker_init_fn = lambda id: np.random.seed(id))
    # create the loader for the test set (to evaluate the model performaces after selection)
    test_loader = DataLoader(dataset_test, shuffle=False, batch_size=1, num_workers=0, pin_memory=True, worker_init_fn = lambda id: np.random.seed(id))
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained weights
    if args.weigths:
        model_dict, epoch, mse = load_checkpoint(args.weigths)
        
        if model_dict is None: raise Exception("The ckpt dose not have the model state_dict!")
        model.load_state_dict(model_dict['model'])
    
        # Saving original Model
        ckpt_path = os.path.join(checkpoints_path, 'unpruned_model.pth')
        if not os.path.exists(ckpt_path):
            torch.save({
                'model': model_dict['model'],
            }, ckpt_path)
    
    model = model.to(device)
    
    pruner = pruner_constructor(args, model, train_data, device)  

    # Defining test inputs to evaluate the model performances
    example_input_sd = torch.randn(1, 3, 964, 540).to(device) # Needed for some pruners

    log_metrics = pd.DataFrame()  
    base_macs_sd, base_nparams = tp.utils.count_ops_and_params(model, example_input_sd)
    
    # Eval Original Model on The TestSet
    psnr_pretrain = PSNR(data_range=args.data_range, output_transform=get_luminance, device=device)
    ssim_pretrain = SSIM(data_range=args.data_range, output_transform=get_luminance, device=device)
    
    test_loss, test_table = testModel(
        loader=test_loader, 
        model=model,
        args=args,
        psnr=psnr_pretrain, 
        ssim=ssim_pretrain,
        data_range=args.data_range,
        device=device
    )
    
    prune_iter_metrics = {}
    prune_iter_metrics["pruning_step"] = 0
    prune_iter_metrics["pruning_rateo"] = 0
    prune_iter_metrics["parameters_(M)"] = base_nparams / 1e6
    prune_iter_metrics["inference_SD_HD_flops(G)"] = base_macs_sd / 1e9
    prune_iter_metrics["mse"] = test_loss
    prune_iter_metrics['ssim'] = float(ssim_pretrain.compute())
    prune_iter_metrics['psnr'] = float(psnr_pretrain.compute())
    wandb.log(prune_iter_metrics)
    wandb.log({"test_results":  wandb.Table(dataframe=test_table)})
    wandb.log({"test_results_by_sequence":  wandb.Table(dataframe=test_table.groupby('sequence').mean().reset_index())})
    wandb.log({"model_description":  f"{model}"})
    test_table.to_csv(os.path.join(original_test_path, 'original_test.csv'), index=False)
    test_table.groupby('sequence').mean().reset_index().to_csv(os.path.join(original_test_path, 'original_test_by_sequence.csv'), index=False)
    log_metrics = log_metrics.append(prune_iter_metrics, ignore_index=True)
    log_metrics.to_csv(log_metrics_path, index=False)
    
    # Save depenency graph visualization
    tp.utils.draw_dependency_graph(pruner.DG, save_as=os.path.join(original_test_path, 'draw_dep_graph.png'), title=None)
    tp.utils.draw_groups(pruner.DG, save_as=os.path.join(original_test_path, 'draw_groups.png'), title=None)
    tp.utils.draw_computational_graph(pruner.DG, save_as=os.path.join(original_test_path, 'draw_comp_graph.png'), title=None)
    
    # Save original model structure
    macs_sd, nparams = tp.utils.count_ops_and_params(model, example_input_sd)
    with open(os.path.join(original_test_path, 'model_details.txt'), 'w') as model_details_file:
        model_details_file.write(f"{model}\n")
        wandb.log({"model_description":  f"{model}"})
        model_details_file.write(
            "  Iter %d/%d, Params: %.2f M => %.2f M\n"
            % (0, args.pruning_steps, base_nparams / 1e6, nparams / 1e6)
        )
        model_details_file.write(
            "  Iter %d/%d, MACs SD_Input: %.2f G => %.2f G\n"
            % (0, args.pruning_steps, base_macs_sd / 1e9, macs_sd / 1e9)
        )

    ####################################
    # Pruning Cycles ###################
    ####################################
    training_iter = 0
    for i in tqdm(range(1, args.pruning_steps + 1)):
        step_path = os.path.join(run_dir, 'pruning_iter_{}'.format(i))
        os.makedirs(step_path, exist_ok=True)
        
        # Learning Sparsity (Some pruning techniques require a treaning step to learn the sparsity)
        if args.sparsity_learning:
            sparsityLearning(
                model=model,
                pruner=pruner,
                loader=train_loader,
                args=args,
            )
            
        # Pruning Step
        pruner.step()
        
        macs_sd, nparams = tp.utils.count_ops_and_params(model, example_input_sd)
        with open(os.path.join(step_path, 'model_details.txt'), 'w') as model_details_file:
            model_details_file.write(f"{model}\n")
            wandb.log({"model_description":  f"{model}"})
            model_details_file.write(
                "  Iter %d/%d, Params: %.2f M => %.2f M\n"
                % (i, args.pruning_steps, base_nparams / 1e6, nparams / 1e6)
            )
            model_details_file.write(
                "  Iter %d/%d, MACs SD_Input: %.2f G => %.2f G\n"
                % (i, args.pruning_steps, base_macs_sd / 1e9, macs_sd / 1e9)
            )

        
        # Model finetuing to recover the loast performacies
        best_model_current_pruning = model # If noting better is found the initial model is the betst
        best_mse_current_pruning = None
        best_optimizer_current_pruning = None
        if args.epochs and args.epochs > 0:
            print("Retraining for recovery!")
            (best_model, best_mse, last_epoch_model, last_epoch_mse, optimizer, run_logs) = trainFor(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                device=device,
                args=args,
                epochs=args.epochs,
                run_folder=step_path,
                pruner=pruner
            )
            model.load_state_dict(best_model.state_dict()) # Load weights of the best model!!
            wandb.log({"train_logs":  wandb.Table(dataframe=run_logs)})
            run_logs.to_csv(os.path.join(step_path, f'traning_log.csv'),index=False) 
            
            model.load_state_dict(best_model.state_dict()) # Load weights of the best model!!
            wandb.log({"train_logs":  wandb.Table(dataframe=run_logs)})
            run_logs.to_csv(os.path.join(step_path, f'traning_log.csv'),index=False) 
            best_model_current_pruning = best_model
            best_mse_current_pruning = best_mse
        
        # Test Best Retrained Model And log results
        prune_iter_metrics = {}
        prune_iter_metrics["pruning_step"] = i
        prune_iter_metrics["pruning_rateo"] = 1 - (nparams / base_nparams)
        prune_iter_metrics["parameters_(M)"] = nparams / 1e6
        prune_iter_metrics["inference_SD_HD_flops(G)"] = macs_sd / 1e9 
        
        psnr_test = PSNR(data_range=args.data_range, device=device)
        ssim_test = SSIM(data_range=args.data_range, device=device)
        test_loss, test_table = testModel(
            loader=test_loader, 
            model=model,
            args=args,
            psnr=psnr_test, 
            ssim=ssim_test,
            data_range=args.data_range,
            device=device
        )
        
        test_table.to_csv(os.path.join(step_path, f'test.csv'),index=False) 
        test_table.groupby('sequence').mean().reset_index().to_csv(os.path.join(step_path, f'test_by_sequence.csv'), index=False)
     
          
        prune_iter_metrics["mse"]= test_loss
        prune_iter_metrics['ssim'] = float(ssim_test.compute())
        prune_iter_metrics['psnr'] = float(psnr_test.compute())
        
        wandb.log(prune_iter_metrics)
        wandb.log({"test_results_by_sequence":  wandb.Table(dataframe=test_table.groupby('sequence').mean().reset_index())})
        wandb.log({"test_results":  wandb.Table(dataframe=test_table)})
        log_metrics = log_metrics.append(prune_iter_metrics, ignore_index=True)
        log_metrics.to_csv(log_metrics_path, index=False)
        
        # Saving the fine tuned (BEST) model
        ckpt_path = os.path.join(checkpoints_path, 'pruned_iteraion_{}.pth'.format(i))
        if not os.path.exists(ckpt_path):
            torch.save({
                'metrics': test_loss,
                'model': tp.state_dict(best_model_current_pruning),
            }, ckpt_path)
