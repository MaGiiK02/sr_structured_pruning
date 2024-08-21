import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import glob


IN_PATH = '/trinity/home/mangelini/data/mangelini/Pruning/Test/finetuned(200)'
OUT_pics = '/trinity/home/mangelini/develop/drln_pruning/plot'
#Y_AXIS = ('pruning_rateo', 'Pruning Rate')
#Y_AXIS = ('psnr', 'PSNR')
Y_AXIS = ('psnr', 'PSNR')
USE_STYELS = False

SR=2
Y_RANGE = (25, 34.5) # Range x2
TEXT_POS_SWIN = (0.975, 0.85)
TEXT_POS_DRLN = (0.975, 0.88)
SR=4
Y_RANGE = (17.5, 28.5) # Range x4
TEXT_POS_SWIN = (0.975, 0.87)
TEXT_POS_DRLN = (0.975, 0.88)


ASPECT="auto" # Fit to data
# ASPECT="square" # Print a square

X_AXIS =('parameters_(M)', "Parameters (M)")
X_AXIS =('pruning_rateo', "Pruning Rate")
X_AXIS = ('crop_infernce_SD_time_s', "Runtime (S)")
# X_AXIS = ('inference_SD_HD_flops(G)', "Flops (G)")
# X_AXIS = ('infernce_128x128_CPUtime_s', "CPU Inference Time (s)")
# X_AXIS = ('infernce_128x128_GPUtime_s' "GPU Inference Time (s)")

COLORS = {
    "swin": "b",
    "original": "r"
}

LINE = {
    "False))": "-",
    "True))": "--",
    "group_norm": "-.",
    "norm": "--",
    "random": ":",
    "growing_reg": "-",
}

MARKER = {
    "(1,": "+",
    "(4,": "x",
    "(8,": "*"
}

def rewrite_name(name):
    replaces = [
        # Pruning(originalx2, norm(1,0.9,False,False,False))-RetrainedOn(DIV2K-150-64x64-N0)
        ("Pruning(", ""),
        # originalx2, norm(1,0.9,False,False,False))-RetrainedOn(DIV2K-150-64x64-N0)
        ("-RetrainedOn(DIV2K-50-64x64-N0)", ""),
        ("-RetrainedOn(DIV2K-150-64x64-N0)", ""),
        # originalx2, norm(1,0.9,False,False,False))
        ("original", "DRLN"),
        ("swinir_c", "SwinIR"),
        # MODELx2, norm(1,0.9,False,False,False))
        ("x2, ", "(x2,"),
        ("x4, ", "(x4,"),
        # MODEL(xS norm(1,0.9,False,False,False))
        ("growing_reg_1(", " Greg,"),
        # ("random(", " Random,"),
        # ("group_norm(", " GNorm,"),
        # ("norm(", " Norm,"),
        ("group_norm(", " GLasso,"),
        ("norm(", " Lasso,"),
        # MODEL(xS, MODE,1,0.9,False,False,False))
        (",1,", ", Steps-1)"),
        (",4,", ", Steps-4)"),
        (",8,", ", Steps-8)"),
        # MODEL(xS, MODE, Steps-8),0.9,False,False,False))
        ("0.9,False,False,True))", ""),
        # MODEL(xS, MODE, Steps-8)
    ]

    for x,y in replaces: name = name.replace(x,y)
    return name

def extract_specific_model_performance(metrics):
    return metrics['yax'], metrics['xax']

def get_match(run_name, dic):
    for key in dic.keys():
        if key in run_name:
            return dic[key]
    return ""

def get_style(run_name):
    line = get_match(run_name, LINE)
    color = get_match(run_name, COLORS)
    marker = get_match(run_name, MARKER)
    return f'{color}{marker}{line}'

def plot_curve(metrics, savepath, y_range, text_pos):

    print(f'plotting on {savepath}')
    figsize=(10,5)
    if ASPECT == "square": figsize=(5,5)
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    (x, y) = text_pos
    (x, y) = (0.143, 0.88)
    plt.figtext(x, y, 'Unpruned', fontsize=8, ha='right', backgroundcolor='w')
    for run_name in metrics.keys():
        yax, xax = extract_specific_model_performance(metrics[run_name]) 
        rewritten_name = rewrite_name(run_name)
        if USE_STYELS:
            style = get_style(run_name)
            axes.plot(xax, yax, style, label = rewritten_name, markersize=20)
        else:
            style = "-x"
            axes.plot(xax, yax, style, label = rewritten_name, markersize=10)

    _, ay_name = Y_AXIS
    _, ax_name = X_AXIS
    axes.set_ylabel(ay_name)
    axes.set_xlabel(ax_name)
    axes.title.set_text(f'{ay_name} over {ax_name}')
    axes.grid()
    axes.legend(loc='lower right')
    plt.tight_layout()

    ly_lim, uy_lim = y_range
    plt.ylim(ly_lim, uy_lim)

    # for ax in axes:
    axes.grid(True)
    if os.path.exists(savepath): os.remove(savepath)
    plt.savefig(savepath)
    plt.close()    


def loadRun(path):
    file_data = pd.read_csv(path, sep=',', header=0)
    name = os.path.basename(os.path.dirname(path))
    
    file_data = file_data.reset_index()  # make sure indexes pair with number of rows

    yax = []
    xax = []
    y_col,_ = Y_AXIS
    x_col,_ = X_AXIS
    for index, row in file_data.iterrows():
        yax.append(row[y_col])
        xax.append(row[x_col])
            
    data = {
        f'{name}': {
            'yax': yax,
            'xax': xax
        }
    }

    return data


if __name__ == '__main__':
    
    runs = glob.glob(f'{IN_PATH}/**/pruning_results_time_2.csv' ,recursive=True)

    # Filter No Retrain
    runs = [r for r in runs if 'DIV2K-0' not in r]
    #runs = [r for r in runs if 'DIV2K-50' not in r]
    # runs = [r for r in runs if 'DIV2K-150' in r]
    runs = [r for r in runs if '4,0.9,False,False,True))-RetrainedOn(DIV2K-150-64x64-N0)' not in r]
    runs = [r for r in runs if '8,0.9,False,False,True))-RetrainedOn(DIV2K-150-64x64-N0)' not in r]
    
    runs = [r for r in runs if f'x{SR},' in r]


    # Only Swin
    # runs = [r for r in runs if 'swin' in r]
    text_pos = TEXT_POS_SWIN

    # Only DRLN
    # runs = [r for r in runs if 'original' in r]
    text_pos = TEXT_POS_DRLN

    # Only No Upsaple Pruning
    # runs = [r for r in runs if 'False))' in r]

    # # Only Upsaple Pruning
    runs = [r for r in runs if 'True))' in r]
    
    # Only Upsaple Pruning
    runs = [r for r in runs if '8' in r]


    methods = [
        # ("random","RND"),
        ("growing","Greg"),
        # ("_norm(","GNorm"),
        # (" norm(","Norm"),
    ]

    for f, n in methods:
        runs_method = [r for r in runs if f in r]

        datas = {}
        runs_method.sort()
        for r in runs_method:
            data = loadRun(r)
            datas.update(data)

        os.makedirs(OUT_pics, exist_ok=True)
        assert len(data.keys()) > 0
        _, ay_name = Y_AXIS
        _, ax_name = X_AXIS

        plot_curve(
            metrics=datas,
            y_range=Y_RANGE,
            text_pos=text_pos,
            savepath=f'{OUT_pics}/{n}_x{SR}_{ay_name}_over_{ax_name}.png',)
        