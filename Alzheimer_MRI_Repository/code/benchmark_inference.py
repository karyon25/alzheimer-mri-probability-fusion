#!/usr/bin/env python3
"""Profile inference cost and throughput for the study's two four-class CNNs
and the deployed EqualSoft ensemble, using trained full-network fine-tuning checkpoints.

This is a separate inference benchmark; it does not run pair-exclusion refits or
fusion evaluation. It measures a model-only forward pass on pre-created tensors
already on the GPU (image decoding, resizing, normalization, and host-to-device
transfer are excluded).

Architecture matches the submitted experiment: torchvision EfficientNet-B2 or
DenseNet121 with a Dropout(0.5) + Linear(4) classifier. Trained weights are
loaded with strict=True, which also verifies that the profiled architecture is
exactly the one that was trained. (Weights do not change parameter count,
operation count, or speed; loading them documents provenance.)

Requirements:
    pip install torch torchvision fvcore

Example (Windows; single GPU, FP32, batch sizes 1 and 32):
    python benchmark_inference.py ^
      --efficientnet-checkpoint "F:\\Dataset_for_Ensemble\\Ensemble_OOF_Project\\full_network_finetuning\\efficientnet_b2\\seed42\\best_val_macro_f1.pt" ^
      --densenet-checkpoint "F:\\Dataset_for_Ensemble\\Ensemble_OOF_Project\\full_network_finetuning\\densenet121\\seed42\\best_val_macro_f1.pt" ^
      --device cuda:0 --batch-sizes 1 32 --precision fp32 ^
      --warmup 50 --iterations 200 --repeats 5 ^
      --output inference_profile.csv

Operation-count convention
--------------------------
fvcore's FlopCountAnalysis counts ONE fused multiply-add as one "flop", i.e.
its total is a MAC count. This script reports both:
    GMAC/image = fvcore total / 1e9
    GOP/image  = 2 x GMAC/image          (1 MAC = 2 operations)
    GOP/s      = GOP/image x images/s
Element-wise ops (activations, SE gating, residual adds, pooling) are not
counted by fvcore; they are listed in the output so the convention can be
stated in the manuscript.

Size reporting
--------------
The training checkpoints also store optimizer/scheduler/scaler/history data, so the
script reports the full training-checkpoint file size separately from the
weights-only serialized state_dict (the value to report as model size).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pickle
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision
from torchvision import models

N_CLASSES = 4
DROPOUT = 0.5
MODEL_NAMES = ("efficientnet_b2", "densenet121")
ENSEMBLE_NAME = "equalsoft_ensemble"


def build_model(model_name: str) -> nn.Module:
    """Construct the same four-class architecture used in the experiment."""
    if model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        in_features = model.classifier[-1].in_features
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, N_CLASSES),
    )
    return model


class EqualSoftEnsemble(nn.Module):
    """Deployed EqualSoft rule: mean of the two softmax probability vectors."""

    def __init__(self, eff: nn.Module, dense: nn.Module):
        super().__init__()
        self.eff = eff
        self.dense = dense

    def forward(self, x):
        p_eff = torch.softmax(self.eff(x).float(), dim=1)
        p_dense = torch.softmax(self.dense(x).float(), dim=1)
        return 0.5 * (p_eff + p_dense)


def safe_torch_load(path: Path) -> Any:
    """Load a local checkpoint, preferring weights-only deserialization."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch without the weights_only argument.
        return torch.load(path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError, AttributeError) as exc:
        # Training checkpoints may contain non-tensor objects (e.g. numpy
        # scalars in history) that weights_only refuses. These are the
        # authors' own local files, so fall back to a full load.
        print(f"  weights_only load refused ({type(exc).__name__}); "
              "falling back to full load of trusted local checkpoint.")
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                checkpoint = value
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError(
            "Checkpoint must be a state_dict or contain one under "
            "'state_dict', 'model_state_dict', or 'model'."
        )
    state = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        state[key.removeprefix("module.")] = value  # DataParallel prefix
    if not state:
        raise ValueError("No tensor weights were found in the checkpoint.")
    return state


def load_model(model_name: str, checkpoint_path: Path) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = build_model(model_name)
    state_dict = extract_state_dict(safe_torch_load(checkpoint_path))
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_dict_tensor_mib(model: nn.Module) -> float:
    """Raw storage for parameters and buffers, excluding training state."""
    size_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return size_bytes / (1024.0 ** 2)


def serialized_weights_mib(model: nn.Module) -> float:
    """Serialized model-only state_dict size (excludes optimizer etc.)."""
    state = {k: t.detach().cpu() for k, t in model.state_dict().items()}
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return len(buffer.getbuffer()) / (1024.0 ** 2)


def precision_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("fp16/bf16 profiling requires a CUDA device.")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def profile_gmac(model: nn.Module, input_size: int) -> tuple[float, str]:
    """Return (GMAC per image, JSON of uncounted ops). fvcore total = MACs."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError("FLOP profiling requires fvcore: pip install fvcore") from exc

    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    analysis = FlopCountAnalysis(model.to("cpu").eval(), dummy)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    total_macs = float(analysis.total())
    if not math.isfinite(total_macs) or total_macs <= 0:
        raise RuntimeError(f"Invalid operation count returned: {total_macs}")
    return total_macs / 1e9, json.dumps(dict(analysis.unsupported_ops()), sort_keys=True)


def benchmark_batch(model, device, input_size, batch_size, precision,
                    warmup, iterations, repeats) -> dict[str, float]:
    """Median over `repeats` timed runs of `iterations` forward passes."""
    x = torch.randn(batch_size, 3, input_size, input_size,
                    device=device, dtype=torch.float32)
    model = model.to(device).eval()
    batch_ms_runs, ips_runs = [], []

    with torch.inference_mode():
        for _ in range(warmup):
            with precision_context(device, precision):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    with precision_context(device, precision):
                        _ = model(x)
                end.record()
                torch.cuda.synchronize(device)
                elapsed_ms = float(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                for _ in range(iterations):
                    with precision_context(device, precision):
                        _ = model(x)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            batch_ms = elapsed_ms / iterations
            batch_ms_runs.append(batch_ms)
            ips_runs.append(batch_size * 1000.0 / batch_ms)

    return {
        "batch_ms_median": statistics.median(batch_ms_runs),
        "images_per_s_median": statistics.median(ips_runs),
        "images_per_s_min": min(ips_runs),
        "images_per_s_max": max(ips_runs),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile model cost and inference throughput.")
    p.add_argument("--efficientnet-checkpoint", required=True, type=Path)
    p.add_argument("--densenet-checkpoint", required=True, type=Path)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    p.add_argument("--data-parallel", action="store_true",
                   help="NOT recommended for the paper table: DataParallel re-replicates "
                        "the model and scatters/gathers tensors on every forward call.")
    p.add_argument("--device-ids", type=int, nargs="+", default=[0, 1])
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("inference_profile.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_size <= 0 or args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("Input size/iterations/repeats must be positive; warmup >= 0.")
    if any(b <= 0 for b in args.batch_sizes):
        raise ValueError("All batch sizes must be positive.")

    device = (torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    if args.data_parallel:
        if device.type != "cuda" or len(args.device_ids) < 2:
            raise ValueError("--data-parallel needs CUDA and at least two device IDs.")
        device = torch.device(f"cuda:{args.device_ids[0]}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if device.type == "cuda":
        ids = args.device_ids if args.data_parallel else [device.index or 0]
        hardware = "; ".join(f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in ids)
    else:
        hardware = platform.processor() or platform.machine() or "CPU"

    print(f"Device: {device} | {hardware}")
    print(f"DataParallel: {args.data_parallel}")
    print(f"PyTorch: {torch.__version__} | torchvision: {torchvision.__version__}")
    print(f"CUDA runtime: {torch.version.cuda} | cuDNN: "
          f"{torch.backends.cudnn.version() if device.type == 'cuda' else 'n/a'} | "
          f"precision: {args.precision}")
    print(f"OS: {platform.platform()} | CPU: {platform.processor()}")
    print("Timing scope: model forward only; input tensors already on the device.")

    # ---- load trained constituents -------------------------------------------
    ckpt_paths = {
        "efficientnet_b2": args.efficientnet_checkpoint.expanduser().resolve(),
        "densenet121": args.densenet_checkpoint.expanduser().resolve(),
    }
    loaded = {n: load_model(n, ckpt_paths[n]) for n in MODEL_NAMES}
    loaded[ENSEMBLE_NAME] = EqualSoftEnsemble(loaded["efficientnet_b2"],
                                              loaded["densenet121"]).eval()

    rows: list[dict[str, Any]] = []
    for name, model in loaded.items():
        params = parameter_count(model)
        if name == ENSEMBLE_NAME:
            train_ckpt_mib = sum(p.stat().st_size for p in ckpt_paths.values()) / 1024.0 ** 2
            ckpt_str = " + ".join(str(p) for p in ckpt_paths.values())
        else:
            train_ckpt_mib = ckpt_paths[name].stat().st_size / 1024.0 ** 2
            ckpt_str = str(ckpt_paths[name])
        state_mib = state_dict_tensor_mib(model)
        weights_mib = serialized_weights_mib(model)
        gmac, unsupported = profile_gmac(model, args.input_size)
        gop = 2.0 * gmac

        bench_model: nn.Module = model.to(device).eval()
        if args.data_parallel:
            bench_model = nn.DataParallel(bench_model, device_ids=args.device_ids,
                                          output_device=args.device_ids[0])

        print(f"\n{name}: parameters={params:,} ({params / 1e6:.3f} M), "
              f"training checkpoint={train_ckpt_mib:.3f} MiB, "
              f"weights-only state_dict={weights_mib:.3f} MiB "
              f"({weights_mib * 1.048576:.3f} MB), "
              f"{gmac:.6f} GMAC/image = {gop:.6f} GOP/image")
        if unsupported != "{}":
            print(f"  fvcore operations not counted: {unsupported}")

        for bs in args.batch_sizes:
            r = benchmark_batch(bench_model, device, args.input_size, bs, args.precision,
                                args.warmup, args.iterations, args.repeats)
            ips = r["images_per_s_median"]
            row = {
                "model": name,
                "checkpoint": ckpt_str,
                "input_size": args.input_size,
                "batch_size": bs,
                "precision": args.precision,
                "parameters": params,
                "parameters_M": params / 1e6,
                "training_checkpoint_file_MiB": train_ckpt_mib,
                "model_state_dict_tensor_MiB": state_mib,
                "weights_only_serialized_MiB": weights_mib,
                "weights_only_serialized_MB": weights_mib * 1.048576,
                "GMAC_per_image": gmac,
                "GOP_per_image": gop,
                "fvcore_unsupported_ops": unsupported,
                "median_batch_latency_ms": r["batch_ms_median"],
                "images_per_second_median": ips,
                "images_per_second_min": r["images_per_s_min"],
                "images_per_second_max": r["images_per_s_max"],
                "GOP_per_second": gop * ips,
                "device": str(device),
                "data_parallel": args.data_parallel,
                "hardware": hardware,
                "pytorch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda_runtime": torch.version.cuda or "none",
                "warmup_iterations": args.warmup,
                "timed_iterations": args.iterations,
                "repeats": args.repeats,
                "timing_scope": "model_forward_only_preloaded_input",
            }
            rows.append(row)
            print(f"  batch={bs:>3} | {r['batch_ms_median']:>9.3f} ms/batch | "
                  f"{ips:>10.2f} images/s (range {r['images_per_s_min']:.0f}-"
                  f"{r['images_per_s_max']:.0f}) | {gop * ips:>10.2f} GOP/s")

        del bench_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results: {args.output.resolve()}")
    print("Convention: GOP = 2 x fvcore MAC count. Report weights-only size as model "
          "size; training-checkpoint size includes optimizer state.")


if __name__ == "__main__":
    main()
