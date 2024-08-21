import argparse
 
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 
def get_arguments():
    parser = argparse.ArgumentParser(description='Train Dansley Residual Super Resolution Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pruning_method', type=str, help='The pruning Algoritm To use')
    parser.add_argument('--weigths', type=str, help='Path to the pretrained weigths')
    parser.add_argument('--low_res_data', type=str, help='Path to low-res data dir')
    parser.add_argument('--high_res_data', type=str, help='Path to high-res data dir')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs between pruning steps')
    parser.add_argument('-ei', '--epochs_iterative', type=int, default=1, help='Number of training epochs for learning sparsity')
    parser.add_argument('-b', '--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=10e-5, help='Learning rate')
    parser.add_argument('-p','--patience', type=float, default=15, help='The amount of epoch that the plateaou scheduler have to wait(whitout changes) before changing the lr')
    parser.add_argument('-m', '--model', type=str, default='DRLN', help='The model to use: (Original, DeepCamera, SwinIR)')
    parser.add_argument('-f', '--filter', type=str, default=None, help='A filter to use to get only wanted imagase from the one present in the path')
    parser.add_argument('-r', '--runs', type=str, default='runs/', help='Base dir for runs outputs and artifacts')
    parser.add_argument('-s', '--scale', type=int, default=2, help='The super resolution scaling factor')
    parser.add_argument('-c', '--crop', type=int, default=None, help='The size of the crop to extract from the image')
    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), help='FP precision for test (single(32bit) | half(16bit))')
    parser.add_argument('--chop', type=str2bool, default=False,help='enable memory-efficient forward, needed for train, or forward to big resolution')
    parser.add_argument('--self_ensemble', type=str2bool, default=False,help='Needed if memory issue arise')
    parser.add_argument('--loader', type=str, default='sequence_yuv', help='The loader to use (folder_yuv, sequence_yuv, div2k_rgb)')
    parser.add_argument('--seed',  type=int,default=0, help='The random seed to use')
    parser.add_argument('--data_range', type=int, default=255, help='Size of the input Data')
    parser.add_argument('--test_mode', type=str2bool, default=False, help='If True the prune will be done in test mode, without oploading the data on WandB, and with reduced dataset')
    parser.add_argument('--loss_epsylon', type=float, default=10e-5, help='The value of epsylon for the Charbonnier Loss (default 10e-5)')
    parser.add_argument('--noise', type=float, default=0, help='The noise STD to generete a gaussian noise for the image')
  
    # Pruning
    parser.add_argument('--use_global_pruning', type=str2bool, default=False, help='If to use Global or Layer Pruning (lowest in all the model or lowest in each layer)')
    parser.add_argument('--pruning_different_layer_rateo',type=str2bool, default=False, help='Allow the pruner to allocate different prunig rateo to the model layers on global importance of the layers (can lead to overpruning)')
    parser.add_argument('--pruning_steps',type=int, default=25, help='Number of pruning stpes to reach the wanted rateo')
    parser.add_argument('--pruning_target_ratio', type=float, default=0.5, help='The wanted sparsity for each step')
    parser.add_argument('--iterative', type=str2bool, default=False,help='Instead of waiting to train the model for x epochs it execute a single batch train before pruning (only taylor)')
    parser.add_argument('--max-pruning-ratio', type=float, default=1.0, help='Maximum pruning percentual (to limit the max pruning possible in some corner scenarios)')
    parser.add_argument('--reg', type=float, default=5e-4,  help='Starting Multiplier for the regularization factor in the Grow Regularization Pruner')
    parser.add_argument('--delta_reg', type=float, default=1e-4, help='The update delta to perform on the growing regularization factor after each pruning step')
    parser.add_argument('--prune_upscale', type=str2bool, default=True, help='If to Prune or Not the Upsamling modules based on PixelShiffle')
    parser.add_argument('--eval_each', type=int, default=1, help='How many pruning step to wait before tesing the model')
    
    # Pruning-Growing Reg Specific
    parser.add_argument('--update_reg_interval', type=int, default=5, help='The amount of mini-batch to wait before increase the regularization')
    parser.add_argument('--target_regularization', type=float, default=1.0, help='The regularization to meet to stop the learning sparsity step in Growing Reg')

    # DRLN
    
    # SWIN
    parser.add_argument('--swinir_config', type=str, default='classical_sr', help='A parameter to define wich configuration of swin to use(classical_sr, lightweight_sr, real_sr)')
    parser.add_argument('--large_model', type=str2bool, default=False, help='If to use the bigger embedding idmension in the real_sr config with 3x3 conv instead of 1x1')
    
    # Infer
    parser.add_argument('--infer_count', type=int, default=-1, help='The amount of images to process in the inference.')
    parser.add_argument('--experiment_path', type=str, default=None, help='The path to the experiment folder from where to get wights and results')

    args = parser.parse_args()
    return args