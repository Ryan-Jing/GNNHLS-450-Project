"""
Profile the dynamic range of weights and activations from a trained GCN model
to determine appropriate ap_fixed<W, I> parameters for FPGA quantization.
"""
import numpy as np
import torch
import os
import matplotlib

# Use a non-interactive backend so this script can generate PNGs anywhere.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(
    _SCRIPT_DIR, "out_new/OGBG_graph_classification/ogbg-molhiv/GCN/data/"
)

PLOTS_DIR = os.path.join(DATA_DIR, "plots")


def _subsample_flat(x, max_samples=500_000, seed=0):
    """
    Subsample a huge tensor for plotting/estimating distributions.
    Uses random indices with replacement (fast enough for hist/CDF).
    """
    x_flat = x.reshape(-1)
    n = x_flat.size
    if n <= max_samples:
        return x_flat

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=max_samples, endpoint=False)
    return x_flat[idx]


def _quant_scale_power2_symmetric_int8(abs_max, power_of_2_scale=True, quant_max=127):
    """
    Match FakeQuantizeInt8's scale rule (symmetric int8):
      raw_scale = abs_max / 127
      scale = ceil_to_power_of_2(raw_scale)  (optional)
    """
    abs_max = float(abs_max)
    abs_max = max(abs_max, 1e-8)
    raw_scale = abs_max / quant_max

    if not power_of_2_scale:
        return raw_scale

    exp = float(np.ceil(np.log2(raw_scale)))
    return float(2.0**exp)


def _sym_int8_codes_from_scale(x, scale, quant_min=-128, quant_max=127):
    """
    Symmetric int8 code mapping:
      x_int = clamp(round(x / scale), -128, 127)
    """
    x_q = np.round(x / scale)
    x_q = np.clip(x_q, quant_min, quant_max)
    return x_q.astype(np.int32)


def _plot_signed_hist(x_sample, out_path, title, bins=200):
    plt.figure(figsize=(8, 5))
    plt.hist(x_sample, bins=bins, density=True, color="#4C72B0", alpha=0.85)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_abs_cdf(abs_sample, out_path, title, percentile_marks=None):
    abs_sorted = np.sort(abs_sample)
    if abs_sorted.size == 0:
        return

    cdf = np.arange(1, abs_sorted.size + 1, dtype=np.float64) / abs_sorted.size

    plt.figure(figsize=(8, 5))
    plt.plot(abs_sorted, cdf, color="#55A868", linewidth=1.5)
    plt.title(title)
    plt.xlabel("|value|")
    plt.ylabel("CDF P(|x| <= t)")

    if percentile_marks:
        labels = ", ".join([f"{p}th={v:.4g}" for p, v in percentile_marks[:3]])
        for p, v in percentile_marks:
            plt.axvline(v, linestyle="--", linewidth=1.0, alpha=0.75)
        plt.text(
            0.02,
            0.98,
            labels,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_int8_code_hist(codes_sample, out_path, title):
    # Probability per code bucket in [-128, 127].
    codes_sample = codes_sample.reshape(-1)
    if codes_sample.size == 0:
        return

    bin_edges = np.arange(-128, 129)  # 257 edges -> 256 bins
    weights = np.ones_like(codes_sample, dtype=np.float64) / codes_sample.size

    plt.figure(figsize=(8, 5))
    plt.hist(codes_sample, bins=bin_edges, weights=weights, color="#8172B2", alpha=0.9)
    plt.title(title)
    plt.xlabel("int8 code (x_int)")
    plt.ylabel("probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def profile_array(name, data, plot_prefix=None):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape:        {data.shape}")
    print(f"  Min:          {data.min():.6f}")
    print(f"  Max:          {data.max():.6f}")
    print(f"  Abs Max:      {np.abs(data).max():.6f}")
    print(f"  Mean:         {data.mean():.6f}")
    print(f"  Std:          {data.std():.6f}")
    print(f"  % zeros:      {100.0 * np.sum(data == 0) / data.size:.2f}%")

    percentiles = [90, 95, 99, 99.9, 100]
    abs_data = np.abs(data)
    print(f"\n  Absolute value percentiles:")
    percentile_vals = []
    for p in percentiles:
        val = np.percentile(abs_data, p)
        percentile_vals.append(val)
        print(f"    {p:>5}th:  {val:.6f}")

    print(f"\n  Suggested ap_fixed formats (signed):")
    abs_max = np.abs(data).max()
    int8_scale = _quant_scale_power2_symmetric_int8(abs_max)
    int8_repr_range = 127 * int8_scale
    for total_bits in [8, 12, 16, 20, 32]:
        int_bits_needed = int(np.ceil(np.log2(abs_max + 1e-10))) + 1 + 1  # +1 for sign, +1 for headroom
        frac_bits = total_bits - int_bits_needed
        if frac_bits < 1:
            status = "(insufficient precision)"
        else:
            step = 2.0 ** (-frac_bits)
            representable_range = 2.0 ** (int_bits_needed - 1)
            status = f"range=±{representable_range}, step={step:.6f}"
        print(f"    ap_fixed<{total_bits},{int_bits_needed}>: {status}")

    print(f"\n  Symmetric int8 (power-of-2 scale) suggestion:")
    print(f"    abs_max={abs_max:.6f} -> scale={int8_scale:.8f} -> representable ±{int8_repr_range:.6f}")

    if plot_prefix:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        seed = abs(hash(plot_prefix)) % (2**32 - 1)

        x_sample = _subsample_flat(data, max_samples=500_000, seed=seed)
        abs_sample = np.abs(x_sample)

        signed_hist_out = os.path.join(PLOTS_DIR, f"{plot_prefix}_signed_hist.png")
        _plot_signed_hist(
            x_sample,
            signed_hist_out,
            title=f"{plot_prefix}: signed value distribution (sample)",
            bins=200,
        )

        abs_cdf_out = os.path.join(PLOTS_DIR, f"{plot_prefix}_abs_cdf.png")
        percentile_marks = list(zip(percentiles, percentile_vals))
        _plot_abs_cdf(
            abs_sample,
            abs_cdf_out,
            title=f"{plot_prefix}: CDF of |x| (sample)",
            percentile_marks=percentile_marks,
        )

        codes_sample = _sym_int8_codes_from_scale(x_sample, scale=int8_scale)
        int8_codes_out = os.path.join(PLOTS_DIR, f"{plot_prefix}_int8_codes.png")
        _plot_int8_code_hist(
            codes_sample,
            int8_codes_out,
            title=f"{plot_prefix}: symmetric int8 code distribution (sample)",
        )


def main():
    print("Profiling trained GCN model value ranges")
    print(f"Data directory: {DATA_DIR}")

    # Profile weights
    weight_path = os.path.join(DATA_DIR, "weight_l0.txt")
    weights = np.loadtxt(weight_path, delimiter=',')
    profile_array("GCN Layer Weights (weight_l0.txt)", weights, plot_prefix="weight_l0")

    # Profile input features (output of embedding_h = input to FPGA)
    features_path = os.path.join(DATA_DIR, "features.txt")
    features = np.loadtxt(features_path, delimiter=',')
    profile_array("Input Features (features.txt)", features, plot_prefix="features")

    # Profile output activations (output of GCN layer on FPGA)
    h2_path = os.path.join(DATA_DIR, "h2_l0.txt")
    h2 = np.loadtxt(h2_path, delimiter=',')
    profile_array("GCN Layer Output (h2_l0.txt)", h2, plot_prefix="h2_l0")

    # Also profile the saved PyTorch parameters directly
    param_path = os.path.join(DATA_DIR, "param_GCN")
    if os.path.exists(param_path):
        print(f"\n\n{'#'*60}")
        print(f"  PyTorch Model Parameters")
        print(f"{'#'*60}")
        state_dict = torch.load(param_path, map_location='cpu', weights_only=False)
        for name, param in state_dict.items():
            data = param.numpy()
            if data.dtype == bool:
                print(f"\n  Skipping boolean param: {name} (shape={data.shape})")
                continue
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float64)
            profile_array(f"param: {name}", data)


if __name__ == "__main__":
    main()
