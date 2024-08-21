import torch
import logging
from typing import Sequence
import torch.nn as nn
import torch_pruning as tp
from model.swin import Upsample, UpsampleOneStep

class UpsamplePruner(tp.BasePruningFunc):
    '''
    Used to prune the upsample block each intenal convolution requires that the output and are grouped by values of (self.scale^2)
    '''
    def prune_out_channels(self, layer: nn.Module, idxs: list):
        channel_group_size = 9 if layer.scale == 3 else 4
        if isinstance(layer, UpsampleOneStep):
            # Swin Transformer (single SR step)
            channel_group_size = (layer.scale)**2
            for key in range(len(layer)):
                module = layer[key]
                if isinstance(module, nn.Conv2d):                
                    module = tp.prune_conv_in_channels(module, idxs)
                    
                    # Group Output Channels y the size of the self.scale value
                    # This will keep aligned the output channels with the input channel enabling pixleshuffle to work
                    to_prune_out = []
                    for idx in idxs:
                        to_prune_out += [(idx*channel_group_size)+i for i in range(channel_group_size)]
                    
                    module = tp.prune_conv_out_channels(module, to_prune_out)
                    layer[key] = module
                    layer.in_channels = layer.in_channels-len(idxs)

            return layer   
        if isinstance(layer, Upsample):
            # Swin Transformer
            for key in range(len(layer)):
                module = layer[key]
                if isinstance(module, nn.Conv2d):
                    print(module.in_channels, module.out_channels)      
                    module = tp.prune_conv_in_channels(module, idxs)
                    
                    # Group Output Channels y the size of the self.scale value
                    # This will keep aligned the output channels with the input channel enabling pixleshuffle to work
                    to_prune_out = []
                    for idx in idxs:
                        to_prune_out += [(idx*channel_group_size)+i for i in range(channel_group_size)]
                    
                    module = tp.prune_conv_out_channels(module, to_prune_out)
                    layer[key] = module
                    layer.in_channels = layer.in_channels-len(idxs)
                    print(module.in_channels-len(idxs), module.out_channels-len(to_prune_out))
            return layer   
        else:
            # DRLN 
            modules = layer.body._modules
            for key in modules:
                module = modules[key]
                if isinstance(module, nn.Conv2d):
                    print(module.in_channels, module.out_channels)      
                    module = tp.prune_conv_in_channels(module, idxs)
                    
                    # Group Output Channels y the size of the self.scale value
                    # This will keep aligned the output channels with the input channel enabling pixleshuffle to work
                    to_prune_out = []
                    for idx in idxs:
                        to_prune_out += [(idx*channel_group_size)+i for i in range(channel_group_size)]
                    
                    module = tp.prune_conv_out_channels(module, to_prune_out)
                    
                    modules[key] = module
                    layer.in_channels = layer.in_channels-len(idxs)
                    print(module.in_channels-len(idxs), module.out_channels-len(to_prune_out))
            return layer   

    # Means we have an inter_dipendency, if we prune the out we need to remove the input and viceversa
    prune_in_channels = prune_out_channels


    def get_out_channels(self, layer):
        # After the each shuffle operation we get the initial number of channels
        return layer.in_channels 

    get_in_channels = get_out_channels