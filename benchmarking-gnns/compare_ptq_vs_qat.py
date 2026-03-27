#!/usr/bin/env python3
"""
PTQ vs QAT comparison for GCN on OGB graph classification.

- QAT: train with fake quantization (--qat in main_infer_OGBG_graph_classification.py).
- PTQ: train full precision (no --qat), then load weights into a QAT-shaped model,
  calibrate activation scales on the training set (EMA in FakeQuantizeInt8), and
  evaluate on the test set.

PTQ is not a separate training objective; it is FP training + post-hoc quantization
at inference time with calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _repo_root() -> Path:
    return _REPO


def gpu_setup(use_gpu: bool, gpu_id: int) -> torch.device:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print("cuda available with GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("cuda not available")
    return torch.device("cpu")


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


def _load_ogbg_bundle(
    config_path: Path,
    out_dir: str,
    gpu_id: Optional[int],
    qat: bool,
    qat_power_of_2_scale: bool,
) -> dict[str, Any]:
    """Build the same net_params / params / device as main_infer_OGBG_graph_classification."""
    with open(config_path) as f:
        config = json.load(f)

    if gpu_id is not None:
        config["gpu"]["id"] = int(gpu_id)
        config["gpu"]["use"] = True

    device = gpu_setup(config["gpu"]["use"], config["gpu"]["id"])
    params = config["params"]
    net_params = dict(config["net_params"])
    net_params["device"] = device
    net_params["gpu_id"] = config["gpu"]["id"]
    net_params["batch_size"] = params["batch_size"]
    net_params["out_dir"] = out_dir
    net_params["qat"] = qat
    net_params["qat_power_of_2_scale"] = qat_power_of_2_scale

    from data.data import LoadData

    dataset_name = config["dataset"]
    dataset = LoadData(dataset_name)

    model_name = config["model"]
    if model_name in ["GCN", "GAT"] and net_params.get("self_loop"):
        print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
        dataset._add_self_loops()
    if net_params.get("pos_enc"):
        print("[!] Adding graph positional encoding.")
        dataset._add_positional_encodings(net_params["pos_enc_dim"])

    net_params["in_dim"] = dataset.all[0][0].ndata["feat"].size(-1)
    net_params["in_dim_edge"] = dataset.all[0][0].edata["feat"].size(-1)
    net_params["n_classes"] = int(dataset.all.num_classes)

    if model_name == "RingGNN":
        num_nodes_train = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        num_nodes_test = [dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))]
        net_params["avg_node_num"] = int(np.ceil(np.mean(num_nodes_train + num_nodes_test)))

    if model_name in ["RingGNN", "3WLGNN"] and net_params.get("pos_enc"):
        net_params["in_dim"] = net_params["pos_enc_dim"]

    drop_last = model_name == "DiffPool"
    train_loader = DataLoader(
        dataset.train,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=drop_last,
        collate_fn=dataset.collate,
    )
    test_loader = DataLoader(
        dataset.test,
        batch_size=params["batch_size"],
        shuffle=False,
        drop_last=drop_last,
        collate_fn=dataset.collate,
    )

    return {
        "config": config,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "device": device,
        "params": params,
        "net_params": net_params,
        "train_loader": train_loader,
        "test_loader": test_loader,
    }


def _reset_fake_quant_buffers(model: torch.nn.Module) -> None:
    from layers.fake_quantize import FakeQuantizeInt8

    for m in model.modules():
        if isinstance(m, FakeQuantizeInt8):
            m.running_max.zero_()
            m.initialized.fill_(False)
            m.scale.fill_(1.0)


def _calibrate_ptq(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    num_batches: int,
) -> None:
    model.train()
    _reset_fake_quant_buffers(model)
    for i, (batch_graphs, _batch_labels) in enumerate(train_loader):
        if i >= num_batches:
            break
        batch_x = batch_graphs.ndata["feat"].to(device)
        batch_e = batch_graphs.edata["feat"].to(device)
        if "pos_enc" in batch_graphs.ndata:
            batch_pos_enc = batch_graphs.ndata["pos_enc"].to(device)
            model(batch_graphs, batch_x, batch_e, batch_pos_enc)
        else:
            model(batch_graphs, batch_x, batch_e)


def evaluate_ptq_accuracy(
    config_path: Path,
    fp_param_path: Path,
    gpu_id: Optional[int],
    calibration_batches: int,
    qat_power_of_2_scale: bool,
) -> float:
    """Load FP checkpoint into QAT-shaped GCN, calibrate fake quant, return test accuracy (%)."""
    from nets.OGBG_graph_classification.load_net import gnn_model
    from train.train_OGBG_graph_classification import evaluate_network_sparse

    bundle = _load_ogbg_bundle(
        config_path,
        out_dir=str(fp_param_path.parent.parent) + "/",
        gpu_id=gpu_id,
        qat=True,
        qat_power_of_2_scale=qat_power_of_2_scale,
    )
    net_params = bundle["net_params"]
    device = bundle["device"]
    model = gnn_model(bundle["model_name"], net_params)
    model = model.to(device)

    ckpt = torch.load(fp_param_path, map_location=device)
    incompatible = model.load_state_dict(ckpt, strict=False)
    if incompatible is not None and hasattr(incompatible, "missing_keys"):
        if incompatible.missing_keys:
            mk = incompatible.missing_keys
            print("[PTQ] Missing keys (expected for FP→QAT load):", mk[:12], "..." if len(mk) > 12 else "")
        if incompatible.unexpected_keys:
            print("[PTQ] Unexpected keys:", incompatible.unexpected_keys)

    _calibrate_ptq(model, device, bundle["train_loader"], calibration_batches)
    model.eval()
    with torch.no_grad():
        _, acc = evaluate_network_sparse(model, device, bundle["test_loader"], epoch=0)
    return 100.0 * acc


def _run_training_subprocess(
    repo_dir: Path,
    config: Path,
    out_dir: Path,
    qat: bool,
    qat_power_of_2_scale: bool,
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
        "--out_dir",
        str(out_dir).rstrip("/") + "/",
    ]
    if qat:
        cmd.append("--qat")
        cmd.extend(["--qat_power_of_2_scale", "True" if qat_power_of_2_scale else "False"])
    if gpu_id is not None:
        cmd.extend(["--gpu_id", str(gpu_id)])
    cmd.extend(extra_args)
    label = "QAT" if qat else "FP (for PTQ)"
    print(f"\n[RUN] {label}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_dir, check=True)
    result_file = _latest_result_file(out_dir)
    acc = _parse_test_accuracy_percent(result_file)
    print(f"[RESULT] {result_file.name}: TEST ACCURACY averaged = {acc:.4f}%")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PTQ (FP train + calibrate + eval) vs QAT (train with fake quant)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config JSON for main_infer_OGBG_graph_classification.py",
    )
    parser.add_argument(
        "--base_out_dir",
        default="out_ptq_vs_qat_compare",
        help="Directory under benchmarking-gnns/ for this experiment run.",
    )
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument(
        "--qat_power_of_2_scale",
        default="True",
        help="Power-of-2 scale for both QAT training and PTQ eval (True/False).",
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=200,
        help="Training batches for PTQ activation calibration (FakeQuantize EMA).",
    )
    parser.add_argument(
        "--skip_fp_train",
        action="store_true",
        help="Do not run FP training; use --fp_param for PTQ eval.",
    )
    parser.add_argument(
        "--skip_qat_train",
        action="store_true",
        help="Do not run QAT training; use --qat_result_acc to fill QAT number.",
    )
    parser.add_argument(
        "--fp_param",
        default=None,
        help="Path to FP checkpoint (param_GCN) if skipping FP train or overriding.",
    )
    parser.add_argument(
        "--qat_result_acc",
        type=float,
        default=None,
        help="Manual QAT test accuracy (%%) if --skip_qat_train.",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Forwarded to main_infer (e.g. --extra_args --epochs 100 --batch_size 32)",
    )
    args = parser.parse_args()

    repo_dir = _repo_root()
    config_path = Path(args.config).resolve()
    p2 = args.qat_power_of_2_scale == "True"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (repo_dir / args.base_out_dir / timestamp).resolve()
    out_fp = base / "fp_train"
    out_qat = base / "qat_train"
    out_fp.mkdir(parents=True, exist_ok=True)
    out_qat.mkdir(parents=True, exist_ok=True)

    print("[INFO] Config:", config_path)
    print("[INFO] Run directory:", base)
    print("[INFO] qat_power_of_2_scale:", p2)

    acc_fp_train: Optional[float] = None
    if not args.skip_fp_train:
        acc_fp_train = _run_training_subprocess(
            repo_dir, config_path, out_fp, qat=False, qat_power_of_2_scale=p2,
            gpu_id=args.gpu_id, extra_args=args.extra_args,
        )

    if args.fp_param:
        fp_path = Path(args.fp_param).resolve()
    else:
        fp_path = out_fp / "data" / "param_GCN"
    if not fp_path.is_file():
        raise FileNotFoundError(
            f"FP checkpoint not found: {fp_path}. Run FP training or pass --fp_param."
        )

    acc_ptq = evaluate_ptq_accuracy(
        config_path=config_path,
        fp_param_path=fp_path,
        gpu_id=args.gpu_id,
        calibration_batches=args.calibration_batches,
        qat_power_of_2_scale=p2,
    )
    print(f"[PTQ] Test accuracy (FP weights + fake quant + calibrate): {acc_ptq:.4f}%")

    acc_qat: Optional[float] = None
    if not args.skip_qat_train:
        acc_qat = _run_training_subprocess(
            repo_dir, config_path, out_qat, qat=True, qat_power_of_2_scale=p2,
            gpu_id=args.gpu_id, extra_args=args.extra_args,
        )
    elif args.qat_result_acc is not None:
        acc_qat = args.qat_result_acc
    else:
        print("[WARN] QAT training skipped and no --qat_result_acc; QAT column empty.")

    print("\n=== PTQ vs QAT (test accuracy %%) ===")
    if acc_fp_train is not None:
        print(f"FP train only (reference):     {acc_fp_train:.4f}%")
    print(f"PTQ (FP ckpt + calibrate):     {acc_ptq:.4f}%")
    if acc_qat is not None:
        print(f"QAT:                           {acc_qat:.4f}%")
        print(f"PTQ − QAT (pp):                {acc_ptq - acc_qat:.4f}")
    print(f"Artifacts: {base}")


if __name__ == "__main__":
    main()
