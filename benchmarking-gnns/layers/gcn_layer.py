import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

# from pyJoules.energy_meter import measure_energy
# from pyJoules.device.nvidia_device import NvidiaGPUDomain

import numpy as np

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
    
# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
msg = fn.copy_u('h', 'm')
reduce = fn.mean('m', 'h')

class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}

class GCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """
    # def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, residual=False, dgl_builtin=False):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation,
        dropout,
        batch_norm,
        residual=False,
        dgl_builtin=True,
        qat=False,
        qat_power_of_2_scale=True,
    ):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        self.qat = qat
        self.qat_power_of_2_scale = qat_power_of_2_scale
        
        if in_dim != out_dim:
            self.residual = False
        
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if self.dgl_builtin == False:
            self.apply_mod = NodeApplyModule(in_dim, out_dim)
        elif dgl.__version__ < "0.5":
            print("Using this function - dgl version", dgl.__version__)
            # self.conv = GraphConv(in_dim, out_dim)
            self.conv = GraphConv(in_dim, out_dim, norm="none", bias=False)
        else:
            self.conv = GraphConv(in_dim, out_dim, allow_zero_in_degree=True)

        if self.qat:
            from layers.fake_quantize import FakeQuantizeInt8
            self.fake_quant_weight = FakeQuantizeInt8(
                is_weight=True,
                power_of_2_scale=self.qat_power_of_2_scale,
            )
            self.fake_quant_output = FakeQuantizeInt8(
                is_weight=False,
                power_of_2_scale=self.qat_power_of_2_scale,
            )

        
    def _conv_with_fake_quant(self, g, feature):
        """Run GraphConv with fake-quantized weights (STE via data swap)."""
        with torch.no_grad():
            orig_weight = self.conv.weight.data.clone()
            weight_fq = self.fake_quant_weight(self.conv.weight)
            self.conv.weight.data.copy_(weight_fq.data)

        h = self.conv(g, feature)

        with torch.no_grad():
            self.conv.weight.data.copy_(orig_weight)
        return h

    def forward(self, g, feature):
        h_in = feature   # to be used for residual connection

        if self.dgl_builtin == False:
            g.ndata['h'] = feature
            g.update_all(msg, reduce)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h'] # result of graph convolution
        elif self.qat:
            h = self._conv_with_fake_quant(g, feature)
        else:
            h = self.conv(g, feature)
        
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
       
        if self.activation:
            h = self.activation(h)
        
        if self.residual:
            h = h_in + h # residual connection

        if self.qat:
            h = self.fake_quant_output(h)
            
        h = self.dropout(h)
        return h
    
    # @measure_energy(domains=[NvidiaGPUDomain(0)])
    def inference(self, g, feature):
        h_in = feature   # to be used for residual connection

        if self.dgl_builtin == False:
            g.ndata['h'] = feature
            g.update_all(msg, reduce)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h'] # result of graph convolution
        elif self.qat:
            h = self._conv_with_fake_quant(g, feature)
        else:
            h = self.conv(g, feature)
        
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
       
        if self.activation:
            h = self.activation(h)
        
        if self.residual:
            h = h_in + h # residual connection

        if self.qat:
            h = self.fake_quant_output(h)
            
        h = self.dropout(h)
        return h


    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.residual)