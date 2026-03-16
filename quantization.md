This repository is from a research paper. A portion of the paper is in GNNHLSSupplementalMaterial.pdf.

The goal that I want to accomplish is to quantize the GCN HLS kernel to \_\_ bits.

First I want to understand how the current GCN kernel is trained. Questions include:

- What precision are they using
- How are they training the model?
  - Is it using PyTorch?
  - Is it using DGL?
  - Where do they get the dataset?
- Do they evaluate the accuracy of the model after training?
  - If not, is there a way to do this evaluation? Maybe PyTorch or DGL has an evaluation framework or script or something?

To run training:

```
cd ./benchmarking-gnns
python3 main_infer_OGBG_graph_classification.py \
  --dataset ogbg-molhiv \
  --config 'configs_infer_ogbg/OGBG_graph_classification_GCN_ogbg-molhiv_100k_144.json'
```

Weight, input, and activation quantization:
- Modify the training script to output the weight_l0.txt file in a quantized format (weight quantization)
- Modify the training script to output the features.txt file in a quantized format (input quantization)
- Need to choose a quantization method (i.e. symmetric, asymmetric, etc.)
- Need to choose a format (i.e. ap_fixed<16,6>)
  - Update TYPE in `gnnhls/gcn/defines.h` to match the format

Updating the TYPE in the defines.h file will allow the FPGA to interpret the weights and inputs in the quantized format. The intermediate activations will also be represetned as TYPE (activation quantization).
