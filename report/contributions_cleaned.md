## Background

This project is motivated by four central questions: what an FPGA is, what Graph Neural Networks (GNNs) are, what High-Level Synthesis (HLS) is, and what the main contribution of the GNNHLS paper is. In short, the GNNHLS paper demonstrates that well-optimized HLS kernels can make FPGA-based GNN inference both fast and energy-efficient, and it provides an open framework to evaluate that claim across multiple GNN models and datasets.

## Contributions

### Adaptation of GNNHLS Code to DRAM-Based FPGAs

We adapted the GNNHLS workflow to run on a DRAM-based FPGA platform. The original paper evaluates on the Xilinx Alveo U280, which has HBM2 and DDR memory. Our available platform is the Kria KV260, which relies on DDR/DRAM and does not provide HBM. To support this hardware difference, we adjusted the memory placement assumptions so that tensors and buffers are mapped in a DDR-compatible manner while keeping the host-kernel interface behavior consistent.

### Quantization of GCN Kernel

Because of time constraints, we focused on one kernel and chose the GCN kernel since it is the simplest starting point. Our main improvement was to quantize the GCN computation path from FP32 toward INT8 operation. The quantized path targets the FPGA-facing GCN tensors, while the rest of the model flow remains in floating-point where needed. This approach gives us a practical balance between implementation complexity and measurable benefit.

### Profiling the Range of the Weights and Activations

Before choosing quantization parameters, we profiled tensor ranges to verify that INT8 is feasible and to avoid overly aggressive clipping. In our current run, input features are approximately in the range -16 to 15.875 and are centered near zero. The first-layer GCN weights are also centered near zero, which is favorable for symmetric quantization. These range characteristics indicate that most values can be represented effectively with properly chosen INT8 scales.

### Choosing the Quantization Parameters

We used Quantization-Aware Training to preserve accuracy under quantization and adopted symmetric per-tensor quantization. We also constrained scales to powers of two so that rescaling is hardware-friendly and can be implemented efficiently with shift-based logic. In the current run, the selected scales are input_scale = 0.125, weight_scale = 0.015625, and output_scale = 0.5.

### QAT

We chose QAT instead of post-training quantization because QAT generally provides better accuracy retention. During training and inference simulation, fake INT8 quantization is inserted on the FPGA-facing tensors, including GCN input activations, GCN weights, and GCN output activations after ReLU. The forward pass applies rounding and clamping behavior to emulate INT8 effects, while the backward pass uses a Straight-Through Estimator so gradients can still propagate through quantization operations. After training, the deployment flow exports quantized representations and corresponding scale factors for FPGA execution.

### Accuracy Results

For accuracy evaluation, we use the OGBG-MOLHIV dataset and compare the FP32 baseline with the QAT-INT8 version of the model. This comparison measures how much predictive performance is retained after quantization. The final report should include the exact number of training epochs and the final accuracy difference between FP32 and INT8.

### Performance Results

For performance evaluation, we compare the INT8 GCN path with the FP32 path in terms of speed and efficiency. The main metrics are inference latency, throughput, memory footprint reduction, and FPGA resource/performance impact. The final report should include measured speedup values and concrete hardware results from your latest runs.

