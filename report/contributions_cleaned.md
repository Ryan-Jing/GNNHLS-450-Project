### Symmetric quantization (for GCN weights/activations)

For signed `b`-bit quantization (for example, int8), the integer range is:

`q_min = -2^(b-1), q_max = 2^(b-1) - 1`

In symmetric quantization, zero-point is fixed to `z = 0`, and we use only a scale `s`:

`q = clamp(round(x / s), q_min, q_max)`

`x_hat = s * q`

where `x` is FP32 value, `q` is quantized integer, and `x_hat` is dequantized approximation.

For this implementation, we choose:

`s = max(abs(x)) / 127`

so that the largest-magnitude value maps near `+/-127`.

### When we need to dequantize

In this codebase (QAT with fake quantization), dequantization happens immediately inside the fake-quant module:

`x_hat = q * s`

So `GraphConv`, batchnorm, activation, and dropout still run in floating point during training/inference. The quantization effect is simulated (quantization noise), but arithmetic is not fully integer end-to-end in these two files.

### Why use power-of-2 scale

If `s = 2^k` (or equivalently `1/s = 2^n`), multiply/divide by scale can be implemented as bit shifts in hardware instead of general multipliers. This reduces area/latency and simplifies FPGA/ASIC implementation.

### How to force scale to power-of-2

1. Compute the usual symmetric scale:
   `s_raw = max(abs(x)) / q_max`
2. Project it to the next higher power of 2 (as implemented):
   `s_p2 = 2^(ceil(log2(s_raw)))`
3. Use `s_p2` in quantize/dequantize:
   `q = clamp(round(x / s_p2), q_min, q_max)`
   `x_hat = s_p2 * q`
<compare a power of 2 scale to a non-power of 2 scale>
This introduces a small extra quantization error compared to unconstrained scale, but is often worth it for hardware efficiency.

<PTQ vs QAT>
