import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import os.path
import shutil
import glob

EXPERIMENTS_PATH="/trinity/home/mangelini/data/mangelini/Pruning/Test/finetuned(200)"
OUT_PATH="/trinity/home/mangelini/develop/drln_pruning/plot/Complexity"
ASPECT="square"

METHODS = {
    "growing_reg": "GReg",
    "group_norm": "GLasso",
    "norm": "Lasso"
}

def returnDictMatch(key_check, d: dict):
    for k in d.keys():
        if k in key_check: return d[k]
    
    raise Exception ("Key Not Found!!!")

def extractExperimentInfo(exp):
    model = "DRLN" if "original" in exp else "SWIN"
    method = returnDictMatch(exp, METHODS)
    scaling = 2 if 'x2,' in exp else 4
    return (model, method, scaling)

def plot(data: pd.DataFrame, savepath:str):
    print(f'plotting on {savepath}')
    figsize=(10,5)
    if ASPECT == "square": figsize=(5,5)
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    
    # PltGeneral
    data[['uncrop_infernce_HD_time_s','uncrop_infernce_HD_CPUtime_s','uncrop_infernce_HD_GPUtime_s']].plot(legend=True)

    axes.grid(True)
    plt.savefig(f"{savepath}/test.png")
    plt.close()    
    

if __name__ == '__main__':
    if os.path.exists(OUT_PATH):shutil.rmtree(OUT_PATH)
    experiments = glob.glob(f'{EXPERIMENTS_PATH}/*' )

    # Filter No Retrain
    experiments = [r for r in experiments if 'DIV2K-50' in r]
    # Only Upsaple Pruning
    experiments = [r for r in experiments if 'False,True))' in r]
    # Filter Only 8 steps
    experiments = [r for r in experiments if '(8,' in r]
    #Remove Random
    experiments = [r for r in experiments if 'growing_reg' in r]

    for exp in experiments:
        model, method, scaling = extractExperimentInfo(exp)
        print(model, method, scaling)
        
        experiment_out = f'{OUT_PATH}/{model}_x{scaling}'
        os.makedirs(experiment_out, exist_ok=True)
        
        data = pd.read_csv(f'{exp}/pruning_results_time_2.csv')
        
        plot(data, experiment_out)