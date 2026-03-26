## Background:

Questions to answer:

- What is an FPGA
- What are GNNs
- What is HLS
- What contribution does GNNHLS paper make?

## Contributions:

### Adaptation of GNNHLS code to DRAM based FPGAs

- We adapted the GNNHLS code to work for DRAM-based FPGAs. In the original paper, the FPGA target was the Xilinx Alveo `U280` card, which provides `8GB HBM2` (high bandwidth) and also `32GB DDR` for off-chip storage.
- The FPGA we have access to is Kria `KV260`, which does not provide HBM; instead, it relies on DDR/DRAM for external memory.
- To make the <???>

### Quantization of GCN Kernel

- Due to limited time, we decided to focus on one of the kernels. We decided to choose the simplest kernel, the GCN kernel.
- The improvement we made was to quantize the GCN kernel.
- \*\* We updated the GCN host kernel to support fixed point representation of the weights. <how is this done?>

### Profiling the Range of the Weights and Activations
- We quantized the GCN kernel from FP32 to INT8. <talk about layers>
- First we profiled the range of the weights and activations. We found that the weights and activations were in a pretty small range.
The input features have range -16 to 15.875, witha  mean of 0. The weights for layer 0 have range between -1.22 to 0.93, with mean 0.001. Since the mean is close to 0, it is a good candidate for symmetric quantization. <maybe we can mention the other datasets and thier ranges>

### Choosing the Quantization Parameters
- We used QAT for better accuracy. We used symmetric quantiaztion (weigths are pretty symmetric). We used power of 2 scales to make the scales implementable as bit shifts in hardware.
- We used the following scales:
input_scale=0.125
weight_scale=0.015625
output_scale=0.5

### QAT
- We use QAT over PTQ for bettery accuracy. We inserts fake int8 quantization modules during training/inference on the FPGA-facing tensors: the GCN input activations, the GraphConv (GCN) weights, and the GCN output activation after ReLU. The fake quantization uses symmetric per-tensor scaling and rounds/clamps in the forward pass, while using a Straight-Through Estimator (STE) so gradients flow through the rounding operation during backprop.

After training, the code exports the resulting int8 representations (features_int8.txt, weight_l0_int8.txt) and the learned power-of-2 scales (qat_scales.txt) for FPGA deployment.

### Accuracy Results
- We trained the model for <how many> epochs on the OGBG-MOLHIV dataset.
- After quantization, we found that the accuracy decreased by <how much>?


### Performance Results
- <discuss the speedup improvements over the float32 model>
