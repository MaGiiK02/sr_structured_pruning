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

def get_cpu_gpu_time(recap):
    last_rows = recap.split("\n")[-3:]
    is_s_CPU = not "ms" in last_rows[0]
    is_s_GPU = not "ms" in last_rows[1]
    cpu_time_ms = float(last_rows[0].replace("Self CPU time total: ", "").replace("ms", "").replace("s", ""))
    gpu_time_ms = float(last_rows[1].replace("Self CUDA time total:", "").replace("ms", "").replace("s", ""))

    if is_s_CPU : cpu_time_ms * 1000
    if is_s_GPU : gpu_time_ms * 1000

    return cpu_time_ms, gpu_time_ms


def eval_complexity(model, example_inputs, folder, device, args, crop=True, repetitions=50):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    forward_manager = ForwardManager(model, False, args)
    warmup=2
    active=5

    model.to(device)
    example_inputs.to(device)

    
    with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],

    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step
    
    schedule=torch.profiler.schedule(
        wait=0,
        warmup=warmup,
        active=active,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(folder)
    # used when outputting for tensorboard
    ) as p:
        # Some iterations to fix the model in the GPU
        with torch.no_grad():
            for _ in range(0+warmup+active):
                if crop:
                    _ = forward_manager.forward(example_inputs)
                else:
                    _ = model(example_inputs)
                p.step()

                for rep in range(repetitions+warmup):
                    if crop:
                        _ = forward_manager.forward(example_inputs)
                    else: 
                        _ = model(example_inputs)
                    p.step()
    
    
    # Runned aside to avoid overhead with the profiler
    timings=np.zeros((15,1))
    with torch.no_grad():
        for rep in range(15):
                starter.record()
                if crop:
                    _ = forward_manager.forward(example_inputs)
                else: 
                    _ = model(example_inputs)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
                p.step()
    
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)

    # Write profiler recap
    f = open(f"{folder}/recap.txt", "w")
    recap = p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10)
    f.write(recap)
    f.close()

    recap = pd.DataFrame(map(vars, p.key_averages(group_by_stack_n=active)))
    cpu_max_row = recap.iloc[recap['cpu_time_total'].idxmax()]
    gpu_max_row = recap.iloc[recap['cuda_time_total'].idxmax()] 
    cpu_time = (cpu_max_row["cpu_time_total"] / cpu_max_row["count"]) / 1000000 # times in us
    gpu_time = (gpu_max_row["cuda_time_total"] / gpu_max_row["count"]) / 1000000 # times in us

    return mean_syn/1000, std_syn/1000, cpu_time, gpu_time


if __name__ == '__main__':

    args = get_arguments()
    experiment_path = args.experiment_path
    inference_out_path = f"{experiment_path}/Infer"
    profiler_out_path = f"{experiment_path}/Stats2"

    os.makedirs(inference_out_path, exist_ok=True)
    os.makedirs(profiler_out_path, exist_ok=True)

    # Load CSV To add time-related results
    experiment_data = pd.read_csv(f"{experiment_path}/pruning_results.csv")
    infer_times = []
    infer_times_std = []
    infer_times_cpu = []
    infer_times_gpu = []
    infer_times_SD = []
    infer_times_std_SD = []
    infer_times_cpu_SD = []
    infer_times_gpu_SD = []
    infer_times_HD = []
    infer_times_std_HD = []
    infer_times_cpu_HD = []
    infer_times_gpu_HD = []

    infer_times_SD_uncrop = []
    infer_times_std_SD_uncrop = []
    infer_times_cpu_SD_uncrop = []
    infer_times_gpu_SD_uncrop = []
    infer_times_HD_uncrop = []
    infer_times_std_HD_uncrop = []
    infer_times_cpu_HD_uncrop = []
    infer_times_gpu_HD_uncrop = []

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

    # Example Inference
    example_input_patch = torch.randn(1, 3, 128, 128).to(device) # Used to eval themodel infer time
    example_input_SD = torch.randn(1, 3, 512, 512).to(device)
    example_input_HD = torch.randn(1, 3, 1024, 1024).to(device)

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
        
        base_macs_sd, base_nparams = tp.utils.count_ops_and_params(model, example_input_patch)
        
        metrics = {}
        metrics["parameters_(M)"] = base_nparams / 1e6
        metrics["inference_SD_HD_flops(G)"] = base_macs_sd / 1e9

        
        # Free Up some Memory from the GPU
        del model_dict
        gc.collect()
        torch.cuda.empty_cache()

        print("Evaluating model complexity------->")
        print("Testing 128x128 inputs")

        # Runtime Evaluation
        p_folder = f"{profiler_folder}/128x128"
        time_infer, time_infer_std, cpu_time, gpu_time = eval_complexity(model, example_input_patch, p_folder, device, args, crop=False)
        metrics['infernce_128x128_time(s)'] = time_infer
        metrics['infernce_128x128_time(s)_STD'] = time_infer_std
        metrics['infernce_128x128_CPUtime(s)'] = cpu_time
        metrics['infernce_128x128_GPUtime(s)'] = gpu_time

        infer_times.append(time_infer)
        infer_times_std.append(time_infer_std)
        infer_times_cpu.append(cpu_time)
        infer_times_gpu.append(gpu_time)


        print("Testing SD inputs")
        p_folder = f"{profiler_folder}/cropped/SD"
        time_infer = time_infer_std = cpu_time = gpu_time  = -1
        try:
            time_infer, time_infer_std, cpu_time, gpu_time = eval_complexity(model, example_input_SD, p_folder, device, args, crop=True)
            metrics['infernce_SD_time_cropped(s)'] = time_infer
            metrics['infernce_SD_time_cropped(s)_STD'] = time_infer_std
            metrics['infernce_SD_CPUtime_cropped(s)'] = cpu_time
            metrics['infernce_SD_GPUtime_cropped(s)'] = gpu_time
        except: 
            metrics['infernce_SD_time_cropped(s)'] = -1
            metrics['infernce_SD_time_cropped(s)_STD'] = -1
            metrics['infernce_SD_CPUtime_cropped(s)'] = -1
            metrics['infernce_SD_GPUtime_cropped(s)'] = -1

        infer_times_SD.append(time_infer)
        infer_times_std_SD.append(time_infer_std)
        infer_times_cpu_SD.append(cpu_time)
        infer_times_gpu_SD.append(gpu_time)

        p_folder = f"{profiler_folder}/uncropped/SD"
        time_infer_uncrop = time_infer_std_uncrop = cpu_time_uncrop = gpu_time_uncrop  = -1
        try:
            time_infer_uncrop, time_infer_std_uncrop, cpu_time_uncrop, gpu_time_uncrop = eval_complexity(model, example_input_SD, p_folder, device, args, crop=False)
            metrics['infernce_SD_time_uncropped(s)'] = time_infer_uncrop
            metrics['infernce_SD_time_uncropped(s)_STD'] = time_infer_std_uncrop
            metrics['infernce_SD_CPUtime_uncropped(s)'] = cpu_time_uncrop
            metrics['infernce_SD_GPUtime_uncropped(s)'] = gpu_time_uncrop
        except: 
            metrics['infernce_SD_time_uncropped(s)'] = -1
            metrics['infernce_SD_time_uncropped(s)_STD'] = -1
            metrics['infernce_SD_CPUtime_uncropped(s)'] = -1
            metrics['infernce_SD_GPUtime_uncropped(s)'] = -1

        infer_times_SD_uncrop.append(time_infer_uncrop)
        infer_times_std_SD_uncrop.append(time_infer_std_uncrop)
        infer_times_cpu_SD_uncrop.append(cpu_time_uncrop)
        infer_times_gpu_SD_uncrop.append(gpu_time_uncrop)



        print("Testing HD inputs")
        p_folder = f"{profiler_folder}/cropped/HD"
        time_infer = time_infer_std = cpu_time = gpu_time  = -1
        try:
            time_infer, time_infer_std, cpu_time, gpu_time = eval_complexity(model, example_input_HD, p_folder, device, args, crop=True)
            metrics['infernce_HD_time_cropped(s)'] = time_infer
            metrics['infernce_HD_time_cropped(s)_STD'] = time_infer_std
            metrics['infernce_HD_CPUtime_cropped(s)'] = cpu_time
            metrics['infernce_HD_GPUtime_cropped(s)'] = gpu_time
        except: 
            metrics['infernce_HD_time_cropped(s)'] = -1
            metrics['infernce_HD_time_cropped(s)_STD'] = -1
            metrics['infernce_HD_CPUtime_cropped(s)'] = -1
            metrics['infernce_HD_GPUtime_cropped(s)'] = -1

        infer_times_HD.append(time_infer)
        infer_times_std_HD.append(time_infer_std)
        infer_times_cpu_HD.append(cpu_time)
        infer_times_gpu_HD.append(gpu_time)

        p_folder = f"{profiler_folder}/uncropped/HD"
        time_infer_uncropped = time_infer_std_uncropped = cpu_time_uncropped = gpu_time_uncropped  = -1
        try:
            time_infer_uncropped, time_infer_std_uncropped, cpu_time_uncropped, gpu_time_uncropped = eval_complexity(model, example_input_HD, p_folder, device, args, crop=False)
            metrics['infernce_HD_time_uncropped(s)'] = time_infer
            metrics['infernce_HD_time_uncropped(s)_STD'] = time_infer_std
            metrics['infernce_HD_CPUtime_uncropped(s)'] = cpu_time
            metrics['infernce_HD_GPUtime_uncropped(s)'] = gpu_time
        except: 
            metrics['infernce_HD_time_uncropped(s)'] = -1
            metrics['infernce_HD_time_uncropped(s)_STD'] = -1
            metrics['infernce_HD_CPUtime_uncropped(s)'] = -1
            metrics['infernce_HD_GPUtime_uncropped(s)'] = -1

        infer_times_HD_uncrop.append(time_infer_uncropped)
        infer_times_std_HD_uncrop.append(time_infer_std_uncropped)
        infer_times_cpu_HD_uncrop.append(cpu_time_uncropped)
        infer_times_gpu_HD_uncrop.append(gpu_time_uncropped)


        print("------------> Complexity evaluation done!")
        print(metrics)


    # Add Fixed metrics to experiment csv
    experiment_data = experiment_data.assign(infernce_128x128_time_s=infer_times)
    experiment_data = experiment_data.assign(infernce_128x128_time_s_STD=infer_times_std)
    experiment_data = experiment_data.assign(infernce_128x128_CPUtime_s=infer_times_cpu)
    experiment_data = experiment_data.assign(infernce_128x128_GPUtime_s=infer_times_gpu)

    experiment_data = experiment_data.assign(crop_infernce_SD_time_s=infer_times_SD)
    experiment_data = experiment_data.assign(crop_infernce_SD_time_s_STD=infer_times_std_SD)
    experiment_data = experiment_data.assign(crop_infernce_SD_CPUtime_s=infer_times_cpu_SD)
    experiment_data = experiment_data.assign(crop_infernce_SD_GPUtime_s=infer_times_gpu_SD)

    experiment_data = experiment_data.assign(crop_infernce_HD_time_s=infer_times_HD)
    experiment_data = experiment_data.assign(crop_infernce_HD_time_s_STD=infer_times_std_HD)
    experiment_data = experiment_data.assign(crop_infernce_HD_CPUtime_s=infer_times_cpu_HD)
    experiment_data = experiment_data.assign(crop_infernce_HD_GPUtime_s=infer_times_gpu_HD)

    experiment_data = experiment_data.assign(uncrop_infernce_SD_time_s=infer_times_SD_uncrop)
    experiment_data = experiment_data.assign(uncrop_infernce_SD_time_s_STD=infer_times_std_SD_uncrop)
    experiment_data = experiment_data.assign(uncrop_infernce_SD_CPUtime_s=infer_times_cpu_SD_uncrop)
    experiment_data = experiment_data.assign(uncrop_infernce_SD_GPUtime_s=infer_times_gpu_SD_uncrop)

    experiment_data = experiment_data.assign(uncrop_infernce_HD_time_s=infer_times_HD_uncrop)
    experiment_data = experiment_data.assign(uncrop_infernce_HD_time_s_STD=infer_times_std_HD_uncrop)
    experiment_data = experiment_data.assign(uncrop_infernce_HD_CPUtime_s=infer_times_cpu_HD_uncrop)
    experiment_data = experiment_data.assign(uncrop_infernce_HD_GPUtime_s=infer_times_gpu_HD_uncrop)

    experiment_data.to_csv(f"{experiment_path}/pruning_results_time_2.csv")




