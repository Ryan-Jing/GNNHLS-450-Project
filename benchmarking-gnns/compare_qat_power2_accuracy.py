#!/usr/bin/env python3
"""
Compare GCN QAT accuracy with and without power-of-2 scale.

This script runs two experiments:
1) QAT with unconstrained scale (power_of_2_scale=False)
2) QAT with power-of-2 scale (power_of_2_scale=True)

It then reports the accuracy delta attributable to forcing power-of-2 scale.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _latest_result_file(out_dir: Path) -> Path:
    results_dir = out_dir / "results"
    files = sorted(results_dir.glob("result_*.txt"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No result file found in: {results_dir}")
    return files[-1]


def _parse_test_accuracy_percent(result_file: Path) -> float:
    content = result_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"TEST ACCURACY averaged:\s*([0-9]*\.?[0-9]+)", content)
    if not match:
        raise ValueError(f"Could not parse test accuracy from: {result_file}")
    return float(match.group(1))


def _run_once(
    repo_dir: Path,
    config: Path,
    out_dir: Path,
    power_of_2_scale: bool,
    gpu_id: Optional[int],
    extra_args: list[str],
) -> float:
    cmd = [
        sys.executable,
        "main_infer_OGBG_graph_classification.py",
        "--config",
        str(config),
        "--model",
        "GCN",
        "--qat",
        "--qat_power_of_2_scale",
        "True" if power_of_2_scale else "False",
        "--out_dir",
        str(out_dir) + "/",
    ]
    if gpu_id is not None:
        cmd.extend(["--gpu_id", str(gpu_id)])
    cmd.extend(extra_args)

    print(f"\n[RUN] power_of_2_scale={power_of_2_scale}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_dir, check=True)

    result_file = _latest_result_file(out_dir)
    acc = _parse_test_accuracy_percent(result_file)
    print(f"[RESULT] {result_file.name}: TEST ACCURACY averaged = {acc:.4f}%")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure accuracy impact of forcing QAT scale to power-of-2."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config JSON for main_infer_OGBG_graph_classification.py",
    )
    parser.add_argument(
        "--base_out_dir",
        default="out_qat_power2_compare",
        help="Base output directory where both runs will be written.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="Optional GPU id override passed to training script.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Extra args forwarded to main_infer_OGBG_graph_classification.py. "
            "Example: --extra_args --epochs 20 --batch_size 64"
        ),
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    config = Path(args.config).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = (repo_dir / args.base_out_dir / timestamp).resolve()
    out_non_p2 = base_out / "qat_non_power2"
    out_p2 = base_out / "qat_power2"

    out_non_p2.mkdir(parents=True, exist_ok=True)
    out_p2.mkdir(parents=True, exist_ok=True)

    print("[INFO] Running comparison with shared config:")
    print(f"       config={config}")
    print(f"       run_dir={base_out}")

    acc_non_p2 = _run_once(
        repo_dir=repo_dir,
        config=config,
        out_dir=out_non_p2,
        power_of_2_scale=False,
        gpu_id=args.gpu_id,
        extra_args=args.extra_args,
    )
    acc_p2 = _run_once(
        repo_dir=repo_dir,
        config=config,
        out_dir=out_p2,
        power_of_2_scale=True,
        gpu_id=args.gpu_id,
        extra_args=args.extra_args,
    )

    abs_drop = acc_non_p2 - acc_p2
    rel_drop = (abs_drop / acc_non_p2 * 100.0) if acc_non_p2 != 0 else 0.0

    print("\n=== Power-of-2 Scale Accuracy Comparison ===")
    print(f"QAT no power-of-2 scale : {acc_non_p2:.4f}%")
    print(f"QAT power-of-2 scale    : {acc_p2:.4f}%")
    print(f"Absolute drop (pp)      : {abs_drop:.4f}")
    print(f"Relative drop (%)       : {rel_drop:.4f}")
    print(f"Outputs saved under     : {base_out}")


if __name__ == "__main__":
    main()

