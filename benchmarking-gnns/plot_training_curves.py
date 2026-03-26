#!/usr/bin/env python3
"""
Plot train and test accuracy vs epoch from two TensorBoard runs (e.g. FP vs QAT).

Training logs scalars: train/_acc, test/_acc (values in [0, 1]).

Dependencies (conda env usually has these; else):
  pip install tensorboard matplotlib

Example:
  python plot_training_curves.py \\
    --fp-logs out_fp/logs \\
    --qat-logs out_qat/logs \\
    -o ../report/plots/accuracy/training_fp_vs_qat.png
"""
from __future__ import annotations

import argparse
import glob
import os

MAX_EPOCHS = 100


def _find_event_file(log_root: str) -> str:
    pattern = os.path.join(log_root, "**", "events.out.tfevents.*")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No TensorBoard event files under {log_root!r} "
            f"(expected **/events.out.tfevents.*)."
        )
    return max(matches, key=os.path.getmtime)


def _load_scalars(event_path: str, tag: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: tensorboard. Install with: pip install tensorboard"
        ) from e

    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(f"Tag {tag!r} not in run {event_path!r}; have: {ea.Tags().get('scalars')}")
    return [(s.step, float(s.value)) for s in ea.Scalars(tag)]


def _clip_first_epochs(points: list[tuple[int, float]], max_epochs: int) -> list[tuple[int, float]]:
    # Epochs are zero-indexed in TensorBoard logs, so first 100 means 0..99.
    return [(step, value) for step, value in points if step < max_epochs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fp-logs",
        default="out_fp/logs",
        help="Root TensorBoard log dir for full-precision (no QAT) run",
    )
    parser.add_argument(
        "--qat-logs",
        default="out_qat/logs",
        help="Root TensorBoard log dir for QAT run",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="plots/training_fp_vs_qat.png",
        help="Output image path (png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: matplotlib. Install with: pip install matplotlib"
        ) from e

    fp_events = _find_event_file(args.fp_logs)
    qat_events = _find_event_file(args.qat_logs)

    series = {
        ("FP", "train"): _clip_first_epochs(_load_scalars(fp_events, "train/_acc"), MAX_EPOCHS),
        ("FP", "test"): _clip_first_epochs(_load_scalars(fp_events, "test/_acc"), MAX_EPOCHS),
        ("QAT", "train"): _clip_first_epochs(_load_scalars(qat_events, "train/_acc"), MAX_EPOCHS),
        ("QAT", "test"): _clip_first_epochs(_load_scalars(qat_events, "test/_acc"), MAX_EPOCHS),
    }

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    styles = {
        ("FP", "train"): dict(color="#0173b2", linestyle="--", linewidth=1.8),
        ("FP", "test"): dict(color="#0173b2", linestyle="-", linewidth=2.0),
        ("QAT", "train"): dict(color="#de8f05", linestyle="--", linewidth=1.8),
        ("QAT", "test"): dict(color="#de8f05", linestyle="-", linewidth=2.0),
    }

    for key, pts in series.items():
        label = f"{key[0]} {key[1]}"
        xs = [p[0] for p in pts]
        ys = [p[1] * 100.0 for p in pts]
        ax.plot(xs, ys, label=label, **styles[key])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("GCN ogbg-molhiv: train vs test accuracy (FP vs QAT)")
    ax.legend(loc="best", framealpha=0.92)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(left=-0.5)

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
