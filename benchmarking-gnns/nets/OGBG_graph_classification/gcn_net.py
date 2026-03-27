from multiprocessing import synchronize
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

import numpy as np
import timeit


"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout

class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.pos_enc = net_params['pos_enc']
        self.qat = net_params.get('qat', False)
        self.qat_power_of_2_scale = net_params.get('qat_power_of_2_scale', True)
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = net_params['in_dim']
            print("net_params['in_dim']", net_params['in_dim'], type(net_params['in_dim']))
            self.embedding_h = nn.Linear(in_dim, hidden_dim)
        

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        #self.embedding_h = nn.Embedding(num_node_type, hidden_dim)
        
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, self.batch_norm, self.residual,
                                              qat=self.qat,
                                              qat_power_of_2_scale=self.qat_power_of_2_scale) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu,
                                    dropout, self.batch_norm, self.residual,
                                    qat=self.qat,
                                    qat_power_of_2_scale=self.qat_power_of_2_scale))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        # dir for saved data(/param)
        self.out_dir = net_params['out_dir']

        if self.qat:
            from layers.fake_quantize import FakeQuantizeInt8
            self.fake_quant_input = FakeQuantizeInt8(
                is_weight=False,
                power_of_2_scale=self.qat_power_of_2_scale,
            )
        

    def forward(self, g, h, e, pos_enc=None):
        # input embedding
        
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)

        if self.qat:
            h = self.fake_quant_input(h)
        
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)

    def inference(self, g, h, e, pos_enc=None):
        # input embedding
        
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)

        if self.qat:
            h = self.fake_quant_input(h)
        
        # save input features of the network
        print("features info:", h.dtype, h.size())
        if(h.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()
            np.savetxt(self.out_dir + "data/features.txt", h.cpu().numpy(), delimiter=',', fmt="%.7f")
            print("the tensor is still on GPU after transferring:", h.is_cuda)
            torch.cuda.synchronize()
        else:
            np.savetxt(self.out_dir + "data/features.txt", h.numpy(), delimiter=',', fmt="%.7f")

        if self.qat:
            self._save_qat_data(h)
        
        if(h.is_cuda):
            print("synchronizing CPU and GPU...")
            torch.cuda.synchronize()
        
        print("Running GNN model...")
        start = timeit.default_timer()

        for conv in self.layers:
            h = conv.inference(g, h)
        
        if(h.is_cuda):
            torch.cuda.synchronize()
        end = timeit.default_timer()
        print("Inference time: %s Seconds" % (end-start))

        # save the inference time
        with open(self.out_dir + 'data/infer_time.log', 'w') as f:
            if(h.is_cuda):
                f.write("GPU Inference time: %s Seconds" % (end-start))
            else:
                f.write("CPU Inference time: %s Seconds" % (end-start))


        g.ndata['h'] = h

        # save results of first layer of the network
        print("h2_l0 info:", h.dtype, h.size())
        if(h.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()
            np.savetxt(self.out_dir + "data/h2_l0.txt", h.cpu().numpy(), delimiter=',', fmt="%.7f")
            print("the tensor is still on GPU after transferring:", h.is_cuda)
            torch.cuda.synchronize()
        else:
            np.savetxt(self.out_dir + "data/h2_l0.txt", h.numpy(), delimiter=',', fmt="%.7f")
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)

    def _save_qat_data(self, h_features):
        """Export int8 weights, features, and scale factors for FPGA deployment."""
        print("Saving QAT quantization data...")

        input_scale = self.fake_quant_input.scale.item()
        weight_scale = self.layers[0].fake_quant_weight.scale.item()
        output_scale = self.layers[0].fake_quant_output.scale.item()

        # Save scale factors
        with open(self.out_dir + "data/qat_scales.txt", 'w') as f:
            f.write(f"input_scale={input_scale}\n")
            f.write(f"weight_scale={weight_scale}\n")
            f.write(f"output_scale={output_scale}\n")
        print(f"  input_scale={input_scale}, weight_scale={weight_scale}, output_scale={output_scale}")

        # Save int8 features
        h_cpu = h_features.detach().cpu()
        features_int8 = torch.round(h_cpu / input_scale).clamp(-128, 127).to(torch.int8)
        np.savetxt(self.out_dir + "data/features_int8.txt",
                   features_int8.numpy(), delimiter=',', fmt='%d')
        print(f"  Saved features_int8.txt: {features_int8.shape}")

        # Save int8 weights
        weight = self.layers[0].conv.weight.detach().cpu()
        weight_fq = self.layers[0].fake_quant_weight(self.layers[0].conv.weight)
        weight_int8 = torch.round(weight_fq.detach().cpu() / weight_scale).clamp(-128, 127).to(torch.int8)
        np.savetxt(self.out_dir + "data/weight_l0_int8.txt",
                   weight_int8.numpy(), delimiter=',', fmt='%d')
        print(f"  Saved weight_l0_int8.txt: {weight_int8.shape}")
    
    def loss(self, pred, label):
        # print("pred", pred, pred.size())
        # print("label", label, label.size())

        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
