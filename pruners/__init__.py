import torch.nn as nn
import torch_pruning as tp
from functools import partial

# Special Layers
from model.ops import _UpsampleBlock
from model.swin import PatchMerging, WindowAttention, Upsample, UpsampleOneStep

# Special Pruners
from pruners.SwinIRPruner import SwinIRPatchMergingPruner, SwinIRAttentionPruner
from pruners.UpsamplePruner import UpsamplePruner

def pruner_constructor(args, model, train_data, device):
    args.sparsity_learning = False
    # First only evaluate the layer from wich the group start, this is equal to standard pruning methods
    group_reduction = "mean" if "group_" in args.pruning_method else "first"
    method = args.pruning_method.replace("group_", "")

    # Random 
    if args.pruning_method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner)
    
    elif method == "slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance(group_reduction=group_reduction)
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, group_lasso=True)
    
    elif method == "norm":
        imp = tp.importance.GroupNormImportance(p=2, group_reduction=group_reduction)
        pruner_entry = partial(tp.pruner.GroupNormPruner)
    
    elif method == "sl":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max', group_reduction=group_reduction) # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg)
    
    elif method == "growing_reg":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, group_reduction=group_reduction)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg)

    else:
        raise Exception("Invalid Pruning Method:", args.pruning_method)
    

    num_heads = {}
    ignored_layers = []
    # Cycle to extract key modules from the model
    for m in model.modules():
        if (args.model.lower() != 'swinir'):
            if (
                m == model.tail or # Final layer that reconstruct the image
                m == model.add_mean  # Layer to normalize the input values n the DRLN (same size as output)
            ):
                ignored_layers.append(m)

        if (args.model.lower() == 'swinir'):
            if (m == model.conv_last): # Final Layer
                ignored_layers.append(m)

        # Extract the head numbers
        if isinstance(m, WindowAttention):
            num_heads[m] = m.num_heads
    
        # Ignore-By-Parameters
        if not args.prune_upscale and ( 
            isinstance(m, _UpsampleBlock) or  isinstance(m, Upsample) or isinstance(m, UpsampleOneStep)
        ):
            ignored_layers.append(m)
            
    
    # Special Custom Pruners maps
    custom_pruners ={
        _UpsampleBlock: UpsamplePruner(), 
        PatchMerging: SwinIRPatchMergingPruner(), 
        Upsample: UpsamplePruner(), 
        UpsampleOneStep: UpsamplePruner(),
        WindowAttention: SwinIRAttentionPruner()
    }
           

    pruning_ratio_dict = {}
    pruner = pruner_entry(
        model,
        importance=imp,
        example_inputs=train_data[0][0].unsqueeze(0).to(device),
        iterative_steps=args.pruning_steps,
        pruning_ratio=args.pruning_target_ratio,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=args.max_pruning_ratio,
        global_pruning=args.use_global_pruning,
        num_heads=num_heads,
        # head_pruning_ratio=args.pruning_target_ratio,
        # prune_head_dims=args.use_global_pruning, # Given the custom prung for the window attention this is the case for local pruning
        ignored_layers=ignored_layers,
        customized_pruners=custom_pruners,
        root_module_types=[ WindowAttention, nn.modules.conv._ConvNd, nn.Linear]
    )
    return pruner