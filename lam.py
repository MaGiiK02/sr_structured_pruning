from lam.lam import multiple_patch_lam_visualization as LAM
from arguments import get_arguments
from utils.ckpt import load_checkpoint
from model.original import DRLN
from model.swin import swinir_builder as SwinIR
import glob
import torch
import pandas as pd
import torch_pruning as tp
import os
from tqdm import tqdm 

ORIGINAL_DRLN = "/trinity/home/mangelini/data/mangelini/Pruning/Train/Train_originalx|scale|_e200_FT200/checkpoints/unpruned_finetuned_model.pth" 
ORIGINAL_SWIN = "/trinity/home/mangelini/data/mangelini/Pruning/Train/Train_swinir_cx|scale|_e200_FT200/checkpoints/unpruned_finetuned_model.pth" 


# Define Tests
TESTS = []
# TESTS.append({
#     "image_name": "Manga109-WarewareHaOniDearu",
#     "hd_path": "/trinity/home/mangelini/data/mangelini/Pruning/Dataset/eval/benchmark/Manga109/HR/WarewareHaOniDearu.png",
#     "crops":[(356, 243, 16), (356, 243, 32), (428, 1083, 16), (428, 1083, 32)]
# })
# TESTS.append({
#     "image_name": "B100-8023",
#     "hd_path":"/trinity/home/mangelini/data/mangelini/Pruning/Dataset/eval/benchmark/B100/HR/8023.png",
#     "crops":[(200, 160, 16), (200, 160, 32)]
# })
# TESTS.append({
#     "image_name": "B100-159008",
#     "hd_path":"/trinity/home/mangelini/data/mangelini/Pruning/Dataset/eval/benchmark/B100/HR/159008.png",
#     "crops":[(188, 100, 16), (188, 100, 32), (300, 125, 16), (300, 125, 32)]
# })
# TESTS.append({
#     "image_name": "Urban100-img019",
#     "hd_path": "/trinity/home/mangelini/data/mangelini/Pruning/Dataset/eval/benchmark/Urban100/HR/img019.png",
#     "crops":[(497, 267, 16), (497, 267, 32), (740, 25, 16), (740, 25, 32), (80, 235, 16), (80, 235, 32) ]
# })

TESTS.append({
    "image_name": "Urban100-img022",
    "hd_path": "/trinity/home/mangelini/data/mangelini/Pruning/Dataset/eval/benchmark/Urban100/HR/img022.png",
    "crops":[(497, 267, 16), (497, 267, 32)]
})


def get_upscaling_rate(path):
    if 'x2,' in path: return 2
    elif 'x3,' in path: return 3
    elif 'x4,' in path: return 4
    elif 'x8,' in path: return 8
    return 1


if __name__ == '__main__':
    args = get_arguments()

    experiment_path = args.experiment_path
    lam_out = f"{experiment_path}/Lam"
    os.makedirs(lam_out, exist_ok=True)


    # create the model
    model_name = "original" if  "original" in experiment_path else "swinir"
    if (model_name == "original"):
        model = DRLN(int(args.scale))
        orignial_path = ORIGINAL_DRLN.replace("|scale|", f"{args.scale}")
    elif (model_name == "swinir"):
        model = SwinIR(args)
        orignial_path = ORIGINAL_SWIN.replace("|scale|", f"{args.scale}")
    else:
        raise Exception("Invalid model")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_model_path = [orignial_path]
    rest = glob.glob(f"{experiment_path}/checkpoints/pruned_*.pth")
    rest.sort()

    # dataframe to store diffusion ndes=xes
    di_df = pd.DataFrame(columns=['test_name', 'pruning_step', 'diffusion_index'])   
    ckpts = original_model_path + rest
    for idx, prune_step_ckpt in enumerate(ckpts):
        # Load pretrained weights
        print(f"CKPT: {prune_step_ckpt}")
        model_dict, epoch, mse = load_checkpoint(prune_step_ckpt)
        if model_dict is None: raise Exception("The ckpt dose not have the model state_dict!")
       
        if prune_step_ckpt == orignial_path:
            model.load_state_dict(state_dict=model_dict['model']) 
        else:
             # Need to use the torch_pruning model loader to handled the pruned model parts
            tp.load_state_dict(model, state_dict=model_dict['model']) 
        
        model_upscale_rate = get_upscaling_rate(experiment_path)
        model = model.to(device)
    
        for test in tqdm(TESTS):
            name = test["image_name"]
            hr_image_path = test["hd_path"]

            composed, visualizations, diffusion_index = LAM(model, hr_image_path, test['crops'], scale=model_upscale_rate, half_input_size=True)
            
            di_df = di_df.append({
                'test_name': name, 
                'pruning_step': idx, 
                'diffusion_index': diffusion_index
            }, ignore_index=True)
           
            composed.save(f'{lam_out}/{name}_{idx}.png') 
            di_df.to_csv(f'{lam_out}/diffusion_index.csv')



