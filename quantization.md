This repository is from a research paper. A portion of the paper is in GNNHLSSupplementalMaterial.pdf.

The goal is to quantize the GCN HLS kernel to **int8** using Quantization-Aware Training (QAT).

## Background

The original researchers trained and ran the GCN model in float32. The FPGA HLS kernel (`gnnhls/gcn/kernel/gcn.cpp`) accelerates only the GCN layer (aggregation + weight multiply + ReLU). Everything else (embedding, graph readout, MLP classification) runs on the host CPU.

The training pipeline uses PyTorch + DGL on the OGBG-MOLHIV dataset.

## What Runs Where

| Component | Runs on | Quantized? |
|---|---|---|
| `embedding_h` (nn.Linear) | CPU/GPU | No (but its output becomes the FPGA input) |
| GCN layer: aggregation + weight multiply + ReLU | **FPGA** | **Yes** |
| Graph readout (mean pooling) | CPU/GPU | No |
| `MLPReadout` (classification head) | CPU/GPU | No |

## Running Training

Float32 baseline:

```
cd ./benchmarking-gnns
python3 main_infer_OGBG_graph_classification.py \
  --dataset ogbg-molhiv \
  --config 'configs_infer_ogbg/OGBG_graph_classification_GCN_ogbg-molhiv_100k_144.json'
```

With int8 QAT:

```
cd ./benchmarking-gnns
python3 main_infer_OGBG_graph_classification.py \
  --dataset ogbg-molhiv \
  --config 'configs_infer_ogbg/OGBG_graph_classification_GCN_ogbg-molhiv_100k_144.json' \
  --qat
```

## Quantization-Aware Training (QAT)

### Approach: Int8 Symmetric Quantization with STE

We use **symmetric per-tensor int8 quantization** with power-of-2 scale factors. During training, "fake quantize" nodes simulate the int8 precision loss in the forward pass, while the Straight-Through Estimator (STE) passes gradients through unchanged in the backward pass. This lets the model learn to compensate for quantization error.

The quantize/dequantize operation:

```
scale = abs_max(tensor) / 127, rounded up to nearest power of 2

Quantize:   x_int = clamp(round(x / scale), -128, 127)
Dequantize: x_hat = x_int * scale
```

Power-of-2 scales are used so that scale multiplication is a **bit shift** in hardware (no multiplier needed).

### Where Fake Quantization Is Inserted

Only the tensors that touch the FPGA are quantized:

```
Raw Features  -->  [embedding_h (float32)]  -->  features
                                                    |
                                           fake_quant_input  (input_scale)
                                                    |
                                      GCN Layer weights fake-quantized  (weight_scale)
                                                    |
                                      aggregation + weight multiply  (float32 accumulator)
                                                    |
                                                  ReLU
                                                    |
                                           fake_quant_output  (output_scale)
                                                    |
                                      [readout + MLP (float32)]  -->  class prediction
```

### Files Modified/Created

- **`layers/fake_quantize.py`** (new) — `FakeQuantizeInt8` module. Per-tensor symmetric int8 fake quantization with STE. Supports power-of-2 scale constraint. Uses EMA for activation scales, per-tensor recomputation for weight scales.
- **`layers/gcn_layer.py`** — Added `qat` parameter to `GCNLayer`. When enabled, GraphConv weights are fake-quantized via a data-swap trick that preserves STE gradient flow, and layer outputs are fake-quantized after ReLU.
- **`nets/OGBG_graph_classification/gcn_net.py`** — Added `fake_quant_input` after the embedding output (= FPGA input). Passes `qat` flag to GCN layers. Added `_save_qat_data()` to export int8 values and scale factors during inference.
- **`main_infer_OGBG_graph_classification.py`** — Added `--qat` CLI flag.

### Exported Data (after QAT training + inference)

Standard files (float values, but now only int8-representable):
- `features.txt` — input features to the GCN layer
- `weight_l0.txt` — GCN layer weights
- `h2_l0.txt` — GCN layer output

QAT-specific files:
- `features_int8.txt` — int8 input features for the FPGA
- `weight_l0_int8.txt` — int8 weight matrix for the FPGA
- `qat_scales.txt` — scale factors (input, weight, output)

### Scale Factors

Each scale converts between float and int8: `float_value = int8_value * scale`.

Scales are computed automatically during QAT training:

1. Find the largest absolute value of the tensor (`abs_max`)
2. Compute raw scale: `abs_max / 127` (so that the largest value maps to int8 max)
3. Round up to the nearest power of 2: `2^ceil(log2(raw_scale))`

For weights, the scale is recomputed from the current weights every forward pass. For activations, the scale tracks an exponential moving average (EMA) of the observed abs max across batches. Rounding to a power of 2 means the FPGA can apply the scale with a bit shift instead of a multiply.

Example for weights: `abs_max=0.285 → raw_scale=0.285/127=0.00224 → 2^ceil(log2(0.00224))=2^(-8)=0.00390625`.

Example from a training run:

| Scale | Value | Power of 2 | Int8 step size | Representable range |
|---|---|---|---|---|
| `input_scale` | 0.125 | 2^(-3) | 0.125 | +/-16.0 |
| `weight_scale` | 0.00390625 | 2^(-8) | ~0.004 | +/-0.5 |
| `output_scale` | 0.25 | 2^(-2) | 0.25 | +/-32.0 |

### How the FPGA Would Use This (Hardware Modification)

The plan is to modify the HLS kernel to:
1. Load **int8** weights and features (instead of float/ap\_fixed)
2. Do **integer** multiply-accumulate (int8 x int8 -> int32 accumulator)
3. At the output, multiply by the combined scale factor (or bit shift) to get the real-valued result
4. This is faster and uses less FPGA area than float or `ap_fixed` arithmetic

## Accuracy Results

Trained for 50 epochs on ogbg-molhiv with hidden\_dim=144, L=1.

| | Float32 | QAT Int8 | Difference |
|---|---|---|---|
| **Test Accuracy** | 96.912% | 96.888% | -0.024% |
| **Train Accuracy** | 96.441% | 96.559% | +0.118% |

Int8 quantization causes a negligible accuracy drop of 0.024%, confirming that the GCN layer can run on the FPGA with int8 arithmetic (4x smaller weights and features than float32) with virtually no loss in accuracy.

## Profiling Value Ranges

Before choosing quantization parameters, run:

```
cd ./benchmarking-gnns
python3 profile_ranges.py
```

This analyzes the trained model's weights and activations to determine value ranges. The percentiles tell you "what fraction of values fit in what range" — this matters because outliers stretch the quantization range and reduce precision for the majority of values.

### Weights — `weight_l0.txt`, shape (144, 144)

The 144x144 weight matrix of the GCN layer (20,736 values).

```
Min:     -0.285        Max:     0.241        Abs Max: 0.285
Mean:     0.001        Std:     0.085        Zeros:   0%

Percentiles (absolute value):
  90th: 0.132    95th: 0.140    99th: 0.159    99.9th: 0.209    100th: 0.285
```

**Interpretation:** All weights are tiny (within +/-0.29), nicely centered around zero. The tight distribution with no extreme outliers is ideal for symmetric quantization. With int8 and `weight_scale=0.004`, we get ~73 distinct levels to represent the full range — excellent precision.

### Input Features — `features.txt`, shape (1,049,163 x 144)

Output of `embedding_h` for all ~1M nodes, each with 144 features (~151M values total).

```
Min:    -16.000        Max:     15.875       Abs Max: 16.000
Mean:    -0.003        Std:      1.700       Zeros:   3.6%

Percentiles (absolute value):
  90th: 2.750    95th: 3.375    99th: 4.250    99.9th: 8.500    100th: 16.000
```

**Interpretation:** Much wider range than weights (+/-16), but most values are small (std=1.7). Only 1% of values exceed +/-4.25, and the +/-16 extremes are very rare outliers. With int8 and `input_scale=0.125`, the representable range is +/-15.875 which covers nearly everything. The 99% of typical values within +/-4.25 get mapped to ~34 int8 levels.

### Output Activations — `h2_l0.txt`, shape (1,049,163 x 144)

GCN layer output after ReLU (~151M values).

```
Min:      0.000        Max:     31.750       Abs Max: 31.750
Mean:     1.082        Std:      1.672       Zeros:  57.7%

Percentiles (absolute value):
  90th: 3.750    95th: 4.500    99th: 6.500    99.9th: 10.000   100th: 31.750
```

**Interpretation:** All values are >= 0 (ReLU clips negatives to zero), and **57.7% are exactly zero** — more than half the neurons are inactive. The non-zero values are mostly small (99% under 6.5), with rare outliers up to ~32. With int8 and `output_scale=0.25`, the representable range is +/-32 which covers everything. Note: since outputs are all non-negative, symmetric quantization wastes the negative half of the int8 range [-128, -1] — asymmetric (uint8) would double precision here, but symmetric is chosen for hardware simplicity.

### Summary

| Tensor | Values are... | Int8 Scale | Precision |
|---|---|---|---|
| Weights | Tiny, +/-0.29 | 0.004 (2^-8) | Excellent — ~73 distinct levels |
| Input features | Moderate, +/-16, mostly +/-4 | 0.125 (2^-3) | Good for typical values, coarse at extremes |
| Output activations | 0 to 32, mostly 0–6, 58% zeros | 0.25 (2^-2) | Adequate, but half of int8 range is unused |

The three tensors have very different value distributions, which is why each gets its own scale factor.
