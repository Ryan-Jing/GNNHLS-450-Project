"""
Symmetric int8 fake quantization with Straight-Through Estimator (STE).

Used for Quantization-Aware Training (QAT) targeting FPGA deployment.
Simulates int8 precision loss during training so the model learns to
compensate. Scale factors can optionally be constrained to powers of 2,
making them implementable as bit shifts in hardware.
"""

import torch
import torch.nn as nn


class FakeQuantizeInt8(nn.Module):
    """
    Per-tensor symmetric int8 fake quantization.

    Forward:
        scale   = absmax / 127  (optionally rounded up to power of 2)
        x_int   = clamp(round(x / scale), -128, 127)
        x_hat   = x_int * scale          (dequantized)

    Backward:
        STE — gradients pass through round() unchanged;
        clamp gradient is zero outside [-128, 127] (after scaling).

    For weights:  scale is recomputed from the tensor each forward call.
    For activations: scale tracks a running EMA of the observed abs-max.
    """

    def __init__(self, is_weight=False, power_of_2_scale=True, ema_momentum=0.1):
        super().__init__()
        self.is_weight = is_weight
        self.power_of_2_scale = power_of_2_scale
        self.ema_momentum = ema_momentum
        self.quant_min = -128
        self.quant_max = 127

        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('running_max', torch.tensor(0.0))
        self.register_buffer('initialized', torch.tensor(False))

    def _compute_scale(self, abs_max):
        abs_max = torch.max(abs_max, torch.tensor(1e-8, device=abs_max.device))
        scale = abs_max / self.quant_max
        if self.power_of_2_scale:
            scale = 2.0 ** torch.ceil(torch.log2(scale))
        return scale

    def forward(self, x):
        if self.training:
            abs_max = x.detach().abs().max()
            if self.is_weight:
                scale = self._compute_scale(abs_max)
            else:
                if not self.initialized:
                    self.running_max.copy_(abs_max)
                    self.initialized.fill_(True)
                else:
                    self.running_max.lerp_(abs_max, self.ema_momentum)
                scale = self._compute_scale(self.running_max)
            self.scale.copy_(scale)
        else:
            scale = self.scale

        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, self.quant_min, self.quant_max)
        # STE: forward rounds, backward passes through
        x_rounded = x_clamped + (torch.round(x_clamped) - x_clamped).detach()
        return x_rounded * scale

    def extra_repr(self):
        return (f'is_weight={self.is_weight}, '
                f'power_of_2_scale={self.power_of_2_scale}, '
                f'scale={self.scale.item():.6f}')
