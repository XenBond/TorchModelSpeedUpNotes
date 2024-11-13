'''
most pruning needs many manual work and need to involve in the begining of the model development. Though tensorrt, pytorch, onnx support model optimization,
and some of them have limited support to pruning, There's no silver button that automatically help the torch model pruning.

By November 2024, the best lib for automatic structural model pruning is "DepGraph" (DeependencyGraph) in CVPR 2023: https://github.com/VainF/Torch-Pruning
However, this library's support to self defined models are still limited since it is still an academic project. The graph builidng seems still cannot 
automatically detect the "Node" defined by the user, and might still search through the internal arch of customized "Node" and causes error.

Despite the downside, this work is still very useful for structural model pruning, which is meaningful for hardware that doesn't support unstructural pruning
Here's an example of pruning PVT-v2's MHA and gcn_lib's GCN using this lib. Note that for simplicity, we only touch the input/output layer. head pruning / dim
 pruning are detailed in the metapruner of the library, and not support customized implementation yet.
'''
import torch
import torch.nn as nn
import torch_pruning as tp

class PVT_MHA_Pruner(tp.pruner.BasePruningFunc):
    '''
    PVT_MHA pruning strategy:

    The code only prune the following architecture:
    - MultiheadAttention
        a. input to q, k, v;
        b. if spatial reduction layer, then the convolution in and out channel are pruned
        c. output of the projection layer (the mixing of multi-head attention before the feedforward layer)

    source code of the PVT model's forward method is also edited to support this pruning strategy
    '''
    def check(self, layer, idxs, to_output):
        super().check(layer, idxs, to_output)
        assert (layer.dim - len(idxs)) % layer.num_heads == 0, "embed_dim (%d) of MultiheadAttention after pruning must divide evenly by `num_heads` (%d)" % (layer.embed_dim, layer.num_heads)

    # prune_in_channels = prune_out_channels only apply when input and output has dependency
    def prune_in_channels(self, layer, idxs: list) -> nn.Module:
        tp.prune_linear_in_channels(layer.q, idxs)
        
        if layer.sr_ratio > 1:
            tp.prune_conv_in_channels(layer.sr, idxs)
            tp.prune_conv_out_channels(layer.sr, idxs)
            tp.prune_layernorm_out_channels(layer.norm, idxs)
            tp.prune_linear_in_channels(layer.kv, idxs)
        else:
            tp.prune_linear_in_channels(layer.kv, idxs)
        return layer

    def prune_out_channels(self, layer, idxs: list) -> nn.Module:
        tp.prune_linear_out_channels(layer.proj, idxs)
        return layer

    def get_out_channels(self, layer):
        return layer.proj.out_features

    def get_in_channels(self, layer):
        return layer.q.in_features
    
# Addtional gcn_lib graph convolution pruner, have dependency due to the readout method
class GCONV_Pruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer, idxs: list) -> nn.Module:
        # prune the conv2d layer
        tp.prune_conv_in_channels(layer.fc1[0], idxs)
        tp.prune_conv_out_channels(layer.fc2[0], idxs)
        tp.prune_batchnorm_out_channels(layer.fc2[1], idxs)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.fc2[0].out_channels

    def get_in_channels(self, layer):
        return layer.fc1[0].in_channels
    
if __name__ == '__main__':
    model = pvt_v2_b5() # pseudo code to creat pvt
    ratio = 0.5
    iterative_steps = 10

    imp = tp.importance.MagnitudeImportance(p=1, group_reduction='mean')
    example_input = torch.randn(1, 3, 224, 224)
    pruner = tp.pruner.MetaPruner(
        model, 
        example_input, 
        global_pruning=True, # If False, a uniform ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        iterative_steps=iterative_steps,
        pruning_ratio=ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        root_module_types=[torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm],# model_def.GCB, Grapher],
        customized_pruners={
            pvt.Attention: PVT_MHA_Pruner(),
            gcn_lib.GCB: GCONV_Pruner(),
        }
    )

    for i in range(iterative_steps):
        pruner.step()