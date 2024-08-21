import torch
from typing import Sequence
import torch.nn as nn
import torch_pruning as tp

class SwinIRPatchMergingPruner(tp.BasePruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: list):
        tp.prune_linear_out_channels(layer.reduction, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        dim = layer.dim
        idxs_repeated = idxs + \
            [i+dim for i in idxs] + \
            [i+2*dim for i in idxs] + \
            [i+3*dim for i in idxs]
        tp.prune_linear_in_channels(layer.reduction, idxs_repeated)
        tp.prune_layernorm_out_channels(layer.norm, idxs_repeated)
        return layer

    def get_out_channels(self, layer):
        return layer.reduction.out_features

    def get_in_channels(self, layer):
        return layer.dim


class SwinIRAttentionPruner(tp.BasePruningFunc):

    def check(self, layer, idxs, to_output):
        print("cheking :", layer.dim - len(idxs), layer.num_heads)
        super().check(layer, idxs, to_output)
        assert (layer.dim - len(idxs)) % layer.num_heads == 0, "dim (%d) of MultiheadAttention after pruning must divide evenly by `num_heads` (%d)" % (layer.dim, layer.num_heads)

    def prune_out_channels(self, layer, idxs: list) -> nn.Module:
        keep_idxs = list(set(range(layer.dim)) - set(idxs))
        keep_idxs.sort()

        pruning_idxs_repeated = idxs + \
            [i+layer.dim for i in idxs] + \
            [i+2*layer.dim for i in idxs]
        keep_idxs_3x_repeated = list(
            set(range(3*layer.dim)) - set(pruning_idxs_repeated))
        keep_idxs_3x_repeated.sort()
        
        layer.qkv.weight = self._prune_parameter_and_grad(layer.qkv.weight, keep_idxs_3x_repeated, 0) # prune out
        layer.qkv.weight = self._prune_parameter_and_grad(layer.qkv.weight, keep_idxs, 1) # prune in
        if hasattr(layer.qkv, 'bias'): layer.qkv.bias = self._prune_parameter_and_grad(layer.qkv.bias, keep_idxs_3x_repeated, 0)

        layer.proj.weight = self._prune_parameter_and_grad(layer.proj.weight, keep_idxs, 0) # prune out
        layer.proj.weight = self._prune_parameter_and_grad(layer.proj.weight, keep_idxs, 1) # prune in
        if hasattr(layer.proj, 'bias') : layer.proj.bias = self._prune_parameter_and_grad(layer.proj.bias, keep_idxs, 0)

        layer.qkv.in_features = layer.qkv.in_features - len(idxs)
        layer.qkv.out_features = layer.qkv.out_features - len(keep_idxs_3x_repeated)
        layer.proj.in_features = layer.qkv.in_features - len(idxs)
        layer.proj.out_features = layer.qkv.out_features - len(idxs)

        layer.dim = layer.dim - len(idxs)
        layer.attention_head_size = layer.dim // layer.num_heads
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.dim

    def get_in_channels(self, layer):
        return self.get_out_channels(layer)