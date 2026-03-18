"""
Profile the dynamic range of weights and activations from a trained GCN model
to determine appropriate ap_fixed<W, I> parameters for FPGA quantization.
"""
import numpy as np
import torch
import os

DATA_DIR = "out_new/OGBG_graph_classification/ogbg-molhiv/GCN/data/"

def profile_array(name, data):
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
    for p in percentiles:
        val = np.percentile(abs_data, p)
        print(f"    {p:>5}th:  {val:.6f}")

    print(f"\n  Suggested ap_fixed formats (signed):")
    abs_max = np.abs(data).max()
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


def main():
    print("Profiling trained GCN model value ranges")
    print(f"Data directory: {DATA_DIR}")

    # Profile weights
    weight_path = os.path.join(DATA_DIR, "weight_l0.txt")
    weights = np.loadtxt(weight_path, delimiter=',')
    profile_array("GCN Layer Weights (weight_l0.txt)", weights)

    # Profile input features (output of embedding_h = input to FPGA)
    features_path = os.path.join(DATA_DIR, "features.txt")
    features = np.loadtxt(features_path, delimiter=',')
    profile_array("Input Features (features.txt)", features)

    # Profile output activations (output of GCN layer on FPGA)
    h2_path = os.path.join(DATA_DIR, "h2_l0.txt")
    h2 = np.loadtxt(h2_path, delimiter=',')
    profile_array("GCN Layer Output (h2_l0.txt)", h2)

    # Also profile the saved PyTorch parameters directly
    param_path = os.path.join(DATA_DIR, "param_GCN")
    if os.path.exists(param_path):
        print(f"\n\n{'#'*60}")
        print(f"  PyTorch Model Parameters")
        print(f"{'#'*60}")
        state_dict = torch.load(param_path, map_location='cpu', weights_only=False)
        for name, param in state_dict.items():
            data = param.numpy()
            profile_array(f"param: {name}", data)


if __name__ == "__main__":
    main()
