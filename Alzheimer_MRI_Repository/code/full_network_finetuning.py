"""
Full-network fine-tuning of EfficientNet-B2 and DenseNet121
=====================================================================

Purpose
-------
Upgrade the two weak frozen-backbone constituents before regenerating OOF
predictions. This script uses ONLY the frozen training and validation splits.

IMPORTANT
---------
- The existing internal test split is NOT loaded.
- The fixed ten-epoch frozen-backbone checkpoint is used only as initialization.
- Model selection is based ONLY on validation Macro-F1.
- Test behavior must not be used to select LR, epoch, or checkpoint.

Protocol
--------
Starting point:
    frozen-backbone checkpoint (epoch 10)

Fine-tuning:
    all backbone layers unfrozen
    AdamW
    backbone LR = 1e-5
    classifier LR = 1e-4
    weight decay = 1e-4
    batch size = 32 total
    max epochs = 20
    early stopping patience = 6
    ReduceLROnPlateau on validation Macro-F1
    factor = 0.3, patience = 2
    cross-entropy loss
    input = 224x224 RGB
    augmentation = rotation +/-10 deg + zoom 0.90-1.10
    ImageNet normalization
    AMP + DataParallel

Run
---
    python full_network_finetuning.py --model efficientnet_b2
    python full_network_finetuning.py --model densenet121

Resume after interruption:
    python full_network_finetuning.py --model efficientnet_b2 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B2_Weights, DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize


# =============================================================================
# LOCKED CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"F:\Dataset_for_Ensemble\Ensemble_OOF_Project")
SPLIT_DIR = PROJECT_ROOT / "splits"

TRAIN_CSV = SPLIT_DIR / "train_seed42.csv"
VAL_CSV = SPLIT_DIR / "val_seed42.csv"
MANIFEST_CSV = SPLIT_DIR / "split_manifest_seed42.csv"

EXPECTED_SPLIT_FINGERPRINT = (
    "7efad99c096dec1fecee39f260c1984c063c5f17c161e82304824f94c8d62aa7"
)

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = True

MAX_EPOCHS = 20
EARLY_STOP_PATIENCE = 6

BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4

SCHEDULER_FACTOR = 0.3
SCHEDULER_PATIENCE = 2
MIN_LR = 1e-7

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EPS = 1e-12


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility preference.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_manifest_fingerprint(df: pd.DataFrame) -> str:
    required = {"relative_path", "class_name", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    canonical = "\n".join(
        f"{r['relative_path']}|{r['class_name']}|{r['split']}"
        for _, r in df.sort_values("relative_path").iterrows()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_frozen_splits():
    for p in [MANIFEST_CSV, TRAIN_CSV, VAL_CSV]:
        if not p.exists():
            raise FileNotFoundError(p)

    manifest = pd.read_csv(MANIFEST_CSV)
    fp = compute_manifest_fingerprint(manifest)

    print("\nSplit fingerprint:")
    print(fp)

    if fp != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(
            "Frozen split fingerprint mismatch.\n"
            f"Expected: {EXPECTED_SPLIT_FINGERPRINT}\n"
            f"Found:    {fp}"
        )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    if len(train_df) != 27187:
        raise RuntimeError(f"Expected 27187 train images, found {len(train_df)}.")
    if len(val_df) != 3398:
        raise RuntimeError(f"Expected 3398 validation images, found {len(val_df)}.")

    unknown_train = set(train_df["class_name"].astype(str)) - set(CLASS_NAMES)
    unknown_val = set(val_df["class_name"].astype(str)) - set(CLASS_NAMES)

    if unknown_train or unknown_val:
        raise RuntimeError(
            f"Unknown classes. Train={unknown_train}, Val={unknown_val}"
        )

    print("Frozen split verification: PASSED")
    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Test CSV is intentionally NOT loaded.")

    return train_df, val_df


# =============================================================================
# DATA
# =============================================================================

class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(str(row["absolute_path"]))

        if not path.exists():
            raise FileNotFoundError(path)

        with Image.open(path) as img:
            img = img.convert("RGB")

        x = self.transform(img)
        y = CLASS_TO_IDX[str(row["class_name"])]

        return x, torch.tensor(y, dtype=torch.long)


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=10,
            scale=(0.90, 1.10),
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, val_tf


# =============================================================================
# MODEL
# =============================================================================

def build_model(model_name: str):
    # weights=None because frozen-backbone training full checkpoint is loaded next.
    if model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, N_CLASSES),
        )
        head_module = model.classifier

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, N_CLASSES),
        )
        head_module = model.classifier

    else:
        raise ValueError(model_name)

    return model, head_module


def frozen_backbone_checkpoint_path(model_name: str) -> Path:
    return (
        PROJECT_ROOT
        / "frozen_backbone_models"
        / model_name
        / f"seed{SEED}"
        / "final_epoch10.pt"
    )


def load_frozen_backbone_initialization(model, model_name):
    ckpt_path = frozen_backbone_checkpoint_path(model_name)

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if ckpt.get("model_name") != model_name:
        raise RuntimeError(
            f"frozen-backbone training checkpoint model mismatch: {ckpt.get('model_name')}"
        )

    if int(ckpt.get("epoch", -1)) != 10:
        raise RuntimeError("frozen-backbone training initialization is not epoch 10.")

    if ckpt.get("split_fingerprint") != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError("frozen-backbone training checkpoint split fingerprint mismatch.")

    if ckpt.get("validation_used") is not False:
        raise RuntimeError("frozen-backbone training checkpoint unexpectedly used validation.")

    if ckpt.get("test_used") is not False:
        raise RuntimeError("frozen-backbone training checkpoint unexpectedly used test data.")

    model.load_state_dict(ckpt["state_dict"])

    return ckpt_path


def make_optimizer(model, head_module):
    # All backbone and classifier parameters are trainable in fine-tuning.
    for p in model.parameters():
        p.requires_grad = True

    head_param_ids = {id(p) for p in head_module.parameters()}

    backbone_params = [
        p
        for p in model.parameters()
        if id(p) not in head_param_ids
    ]

    head_params = list(head_module.parameters())

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": BACKBONE_LR,
                "name": "backbone",
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
                "name": "head",
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


# =============================================================================
# METRICS
# =============================================================================

def normalize_probs(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def multiclass_brier(y_true, probs):
    onehot = np.eye(N_CLASSES)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, probs, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == 0:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf > edges[i]) & (conf <= edges[i + 1])

        if np.any(mask):
            ece += (
                abs(correct[mask].mean() - conf[mask].mean())
                * mask.mean()
            )

    return float(ece)


def evaluate_from_arrays(y_true, probs):
    probs = normalize_probs(probs)
    pred = probs.argmax(axis=1)

    y_bin = label_binarize(
        y_true,
        classes=np.arange(N_CLASSES),
    )

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "macro_f1": float(
            f1_score(y_true, pred, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y_true, pred, average="weighted")
        ),
        "macro_auc_ovr": float(
            roc_auc_score(
                y_bin,
                probs,
                average="macro",
            )
        ),
        "nll": float(
            log_loss(
                y_true,
                probs,
                labels=np.arange(N_CLASSES),
            )
        ),
        "brier": multiclass_brier(y_true, probs),
        "ece15": expected_calibration_error(
            y_true,
            probs,
            n_bins=15,
        ),
    }


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    amp_enabled,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = y.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

    return {
        "train_loss": running_loss / total,
        "train_accuracy": correct / total,
    }


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
):
    model.eval()

    running_loss = 0.0
    total = 0

    all_y = []
    all_probs = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(
            logits.float(),
            dim=1,
        )

        bs = y.size(0)
        running_loss += loss.item() * bs
        total += bs

        all_y.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y)
    probs = normalize_probs(
        np.concatenate(all_probs, axis=0)
    )

    metrics = evaluate_from_arrays(
        y_true,
        probs,
    )

    metrics["val_loss"] = running_loss / total

    return metrics


# =============================================================================
# CHECKPOINTING
# =============================================================================

def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_epoch,
    best_macro_f1,
    epochs_without_improvement,
    history,
    model_name,
):
    torch.save(
        {
            "model_name": model_name,
            "epoch": int(epoch),
            "state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_epoch": int(best_epoch),
            "best_macro_f1": float(best_macro_f1),
            "epochs_without_improvement": int(
                epochs_without_improvement
            ),
            "history": history,
            "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
            "train_used": True,
            "validation_used": True,
            "test_used": False,
            "protocol": {
                "initialization": "frozen_backbone_epoch10",
                "full_backbone_unfrozen": True,
                "optimizer": "AdamW",
                "backbone_lr": BACKBONE_LR,
                "head_lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
                "max_epochs": MAX_EPOCHS,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "scheduler": "ReduceLROnPlateau",
                "scheduler_factor": SCHEDULER_FACTOR,
                "scheduler_patience": SCHEDULER_PATIENCE,
                "batch_size_total": BATCH_SIZE,
                "input_size": IMAGE_SIZE,
                "augmentation": "RandomAffine(degrees=10, scale=(0.90,1.10))",
            },
        },
        path,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "efficientnet_b2",
            "densenet121",
        ],
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(SEED)

    train_df, val_df = verify_frozen_splits()

    out_dir = (
        PROJECT_ROOT
        / "full_network_finetuning"
        / args.model
        / f"seed{SEED}"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_ckpt_path = out_dir / "last_checkpoint.pt"
    best_ckpt_path = out_dir / "best_val_macro_f1.pt"
    history_csv = out_dir / "training_history.csv"
    summary_json = out_dir / "fine_tuning_summary.json"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    n_gpus = (
        torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0
    )

    amp_enabled = device.type == "cuda"

    print("\n" + "=" * 88)
    print("FULL-NETWORK FINE-TUNING — FULL-NETWORK FINE-TUNING")
    print("=" * 88)
    print("Model:", args.model)
    print("Initialization: frozen-backbone epoch-10 checkpoint")
    print("Device:", device)
    print("Visible CUDA GPUs:", n_gpus)
    print("AMP:", amp_enabled)
    print("Batch size TOTAL:", BATCH_SIZE)
    print("Backbone LR:", BACKBONE_LR)
    print("Head LR:", HEAD_LR)
    print("Weight decay:", WEIGHT_DECAY)
    print("Max fine-tune epochs:", MAX_EPOCHS)
    print("Early stop patience:", EARLY_STOP_PATIENCE)
    print("Selection metric: validation Macro-F1")
    print("Output:", out_dir)

    if torch.cuda.is_available():
        for i in range(n_gpus):
            print(
                f"GPU {i}: "
                f"{torch.cuda.get_device_name(i)}"
            )

    train_tf, val_tf = build_transforms()

    train_ds = MRIDataset(
        train_df,
        train_tf,
    )
    val_ds = MRIDataset(
        val_df,
        val_tf,
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and amp_enabled,
        persistent_workers=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and amp_enabled,
        persistent_workers=False,
        drop_last=False,
    )

    model, head_module = build_model(
        args.model
    )

    frozen_backbone_path = load_frozen_backbone_initialization(
        model,
        args.model,
    )

    print("\nfrozen-backbone training initialization loaded:")
    print(frozen_backbone_path)

    optimizer = make_optimizer(
        model,
        head_module,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 1
    best_epoch = 0
    best_macro_f1 = -np.inf
    epochs_without_improvement = 0
    history = []

    if args.resume:
        if not last_ckpt_path.exists():
            raise FileNotFoundError(
                f"--resume specified but no checkpoint exists:\n"
                f"{last_ckpt_path}"
            )

        resume_ckpt = torch.load(
            last_ckpt_path,
            map_location="cpu",
        )

        model.load_state_dict(
            resume_ckpt["state_dict"]
        )
        optimizer.load_state_dict(
            resume_ckpt["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            resume_ckpt["scheduler_state_dict"]
        )
        scaler.load_state_dict(
            resume_ckpt["scaler_state_dict"]
        )

        start_epoch = (
            int(resume_ckpt["epoch"]) + 1
        )
        best_epoch = int(
            resume_ckpt["best_epoch"]
        )
        best_macro_f1 = float(
            resume_ckpt["best_macro_f1"]
        )
        epochs_without_improvement = int(
            resume_ckpt[
                "epochs_without_improvement"
            ]
        )
        history = resume_ckpt["history"]

        print("\nRESUME ENABLED")
        print(
            "Continuing from fine-tune epoch:",
            start_epoch,
        )
        print(
            "Best epoch so far:",
            best_epoch,
        )
        print(
            "Best validation Macro-F1:",
            f"{best_macro_f1:.6f}",
        )

    model = model.to(device)

    if amp_enabled and n_gpus > 1:
        model = nn.DataParallel(model)
        print(
            f"DataParallel enabled across "
            f"{n_gpus} GPUs."
        )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTotal parameters:    ", f"{total_params:,}")
    print("Trainable parameters:", f"{trainable_params:,}")
    print("Full-network fine-tuning: ENABLED")

    if start_epoch > MAX_EPOCHS:
        print(
            "Training already reached configured "
            "maximum epoch."
        )
        return

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
        t0 = time.time()

        train_m = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            amp_enabled,
        )

        val_m = validate(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled,
        )

        scheduler.step(
            val_m["macro_f1"]
        )

        backbone_lr_now = optimizer.param_groups[0]["lr"]
        head_lr_now = optimizer.param_groups[1]["lr"]

        improved = (
            val_m["macro_f1"]
            > best_macro_f1 + 1e-6
        )

        if improved:
            best_macro_f1 = (
                val_m["macro_f1"]
            )
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_m["train_loss"],
            "train_accuracy": train_m["train_accuracy"],
            "val_loss": val_m["val_loss"],
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_weighted_f1": val_m["weighted_f1"],
            "val_macro_auc_ovr": val_m["macro_auc_ovr"],
            "val_nll": val_m["nll"],
            "val_brier": val_m["brier"],
            "val_ece15": val_m["ece15"],
            "backbone_lr": backbone_lr_now,
            "head_lr": head_lr_now,
            "improved_best": improved,
            "epoch_seconds": time.time() - t0,
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            history_csv,
            index=False,
            encoding="utf-8-sig",
        )

        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_epoch,
            best_macro_f1,
            epochs_without_improvement,
            history,
            args.model,
        )

        if improved:
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_epoch,
                best_macro_f1,
                epochs_without_improvement,
                history,
                args.model,
            )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss={train_m['train_loss']:.4f} | "
            f"train acc={train_m['train_accuracy']:.4f} | "
            f"val loss={val_m['val_loss']:.4f} | "
            f"val acc={val_m['accuracy']:.4f} | "
            f"val bal_acc={val_m['balanced_accuracy']:.4f} | "
            f"val mF1={val_m['macro_f1']:.4f} | "
            f"val AUC={val_m['macro_auc_ovr']:.4f} | "
            f"LR(backbone/head)="
            f"{backbone_lr_now:.2e}/{head_lr_now:.2e} | "
            f"{row['epoch_seconds']:.1f}s"
            + ("  <-- BEST" if improved else "")
        )

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    hist_df = pd.DataFrame(history)

    best_row = hist_df.loc[
        hist_df["epoch"] == best_epoch
    ].iloc[0]

    summary = {
        "model": args.model,
        "initialization_checkpoint": str(frozen_backbone_path),
        "best_checkpoint": str(best_ckpt_path),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(
            best_row["val_macro_f1"]
        ),
        "best_val_accuracy": float(
            best_row["val_accuracy"]
        ),
        "best_val_balanced_accuracy": float(
            best_row["val_balanced_accuracy"]
        ),
        "best_val_macro_auc_ovr": float(
            best_row["val_macro_auc_ovr"]
        ),
        "best_val_nll": float(
            best_row["val_nll"]
        ),
        "best_val_brier": float(
            best_row["val_brier"]
        ),
        "best_val_ece15": float(
            best_row["val_ece15"]
        ),
        "epochs_completed": int(
            hist_df["epoch"].max()
        ),
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "train_used": True,
        "validation_used": True,
        "test_used": False,
        "protocol": {
            "full_backbone_unfrozen": True,
            "optimizer": "AdamW",
            "backbone_lr_initial": BACKBONE_LR,
            "head_lr_initial": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "selection_metric": "validation_macro_f1",
            "batch_size_total": BATCH_SIZE,
            "input_size": IMAGE_SIZE,
            "loss": "CrossEntropyLoss",
            "test_loaded": False,
        },
    }

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 88)
    print("FULL-NETWORK FINE-TUNING FINE-TUNING COMPLETE")
    print("=" * 88)
    print("Model:", args.model)
    print("Best epoch:", best_epoch)
    print(
        "Best validation Macro-F1:",
        f"{summary['best_val_macro_f1']:.4f}",
    )
    print(
        "Best validation Balanced Acc:",
        f"{summary['best_val_balanced_accuracy']:.4f}",
    )
    print(
        "Best validation Accuracy:",
        f"{summary['best_val_accuracy']:.4f}",
    )
    print(
        "Best validation Macro AUC:",
        f"{summary['best_val_macro_auc_ovr']:.4f}",
    )
    print("Best checkpoint:", best_ckpt_path)
    print("\nTEST STATUS: NOT LOADED")
    print(
        "Do not evaluate the internal test until the "
        "full-network fine-tuning training protocol is frozen for both models."
    )


if __name__ == "__main__":
    main()
