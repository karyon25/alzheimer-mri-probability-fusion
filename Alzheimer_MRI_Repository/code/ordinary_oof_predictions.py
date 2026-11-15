"""
Ordinary five-fold out-of-fold (OOF) probability generation
==============================================

Purpose
-------
Generate held-out OOF probabilities under the frozen full-network fine-tuning
protocol, using ONLY train_seed42.csv.

Each fold independently reproduces the development pipeline:

    ImageNet initialization
        -> frozen-backbone warm-up for 10 fixed epochs
        -> full-network fine-tuning for a fixed number of epochs
        -> predict the held-out OOF fold once

Fixed fine-tuning durations selected using the validation partition:
    EfficientNet-B2 : 20 epochs
    DenseNet121     : 17 epochs

Why no validation scheduler inside OOF?
---------------------------------------
The preceding validation-guided fine-tuning runs showed that:
- EfficientNet-B2 used 1e-5 / 1e-4 throughout its selected 20 epochs.
- DenseNet121 reached its selected best checkpoint at epoch 17 before any
  LR reduction occurred.

Therefore the OOF schedule is fixed before OOF generation and does not use
the held-out fold for model/epoch/LR selection.

Data safeguards
---------------
- Uses train_seed42.csv only.
- Does NOT load val_seed42.csv.
- Does NOT load test_seed42.csv.
- Held-out OOF fold is never used during training or checkpoint selection.
- Checkpoints fitted on the full training partition are not reused as OOF
  initialization, avoiding training exposure to the held-out fold.

Examples
--------
Run one fold:
    python ordinary_oof_predictions.py --model efficientnet_b2 --fold 0

Resume an interrupted fold:
    python ordinary_oof_predictions.py --model efficientnet_b2 --fold 0 --resume

Run all incomplete folds sequentially:
    python ordinary_oof_predictions.py --model efficientnet_b2 --resume
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize


# =============================================================================
# LOCKED STUDY CONFIG
# =============================================================================

PROJECT_ROOT = Path(r"F:\Dataset_for_Ensemble\Ensemble_OOF_Project")
SPLIT_DIR = PROJECT_ROOT / "splits"

TRAIN_CSV = SPLIT_DIR / "train_seed42.csv"
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
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
N_CLASSES = len(CLASS_NAMES)

N_SPLITS = 5
SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = True
DROPOUT = 0.5

ROTATION_DEGREES = 10
ZOOM_RANGE = (0.90, 1.10)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Phase 1: reference-aligned frozen-backbone warm-up.
WARMUP_EPOCHS = 10
WARMUP_LR = 1e-4

# Phase 2: locked full-network fine-tuning full-network fine-tuning.
FINETUNE_EPOCHS = {
    "efficientnet_b2": 20,
    "densenet121": 17,
}
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4

EPS = 1e-12


# =============================================================================
# REPRODUCIBILITY / SPLIT
# =============================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def manifest_fingerprint(df: pd.DataFrame) -> str:
    required = {"relative_path", "class_name", "split"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Manifest missing: {sorted(missing)}")

    canonical = "\n".join(
        f"{r['relative_path']}|{r['class_name']}|{r['split']}"
        for _, r in df.sort_values("relative_path").iterrows()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_source():
    for p in [MANIFEST_CSV, TRAIN_CSV]:
        if not p.exists():
            raise FileNotFoundError(p)

    manifest = pd.read_csv(MANIFEST_CSV)
    fp = manifest_fingerprint(manifest)

    print("\nSplit fingerprint:")
    print(fp)

    if fp != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(
            "Frozen split fingerprint mismatch.\n"
            f"Expected: {EXPECTED_SPLIT_FINGERPRINT}\n"
            f"Found:    {fp}"
        )

    df = pd.read_csv(TRAIN_CSV)

    if len(df) != 27187:
        raise RuntimeError(f"Expected 27187 train rows; found {len(df)}.")

    unknown = set(df["class_name"].astype(str)) - set(CLASS_NAMES)
    if unknown:
        raise RuntimeError(f"Unknown classes: {sorted(unknown)}")

    print("Frozen split verification: PASSED")
    print("OOF source images:", len(df))
    print("Validation CSV is intentionally NOT loaded.")
    print("Test CSV is intentionally NOT loaded.")

    return df


def make_folds(train_df: pd.DataFrame) -> pd.DataFrame:
    # Uses the original five-fold OOF partition construction.
    df = train_df.copy().reset_index(drop=True)
    y = df["class_name"].map(CLASS_TO_IDX).to_numpy()

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    fold_ids = np.full(len(df), -1, dtype=int)
    dummy = np.zeros(len(df))

    for fold, (_, hold_idx) in enumerate(skf.split(dummy, y)):
        fold_ids[hold_idx] = fold

    df["oof_fold"] = fold_ids

    if np.any(fold_ids < 0):
        raise RuntimeError("Incomplete OOF fold assignment.")

    return df


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

        return {
            "image": x,
            "label": torch.tensor(y, dtype=torch.long),
            "relative_path": str(row["relative_path"]),
            "absolute_path": str(row["absolute_path"]),
            "class_name": str(row["class_name"]),
            "oof_fold": int(row["oof_fold"]),
        }


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=ROTATION_DEGREES,
            scale=ZOOM_RANGE,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, eval_tf


# =============================================================================
# MODEL / OPTIMIZERS
# =============================================================================

def build_imagenet_model(model_name: str):
    if model_name == "efficientnet_b2":
        model = models.efficientnet_b2(
            weights=EfficientNet_B2_Weights.IMAGENET1K_V1
        )

        for p in model.parameters():
            p.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, N_CLASSES),
        )

        head = model.classifier

    elif model_name == "densenet121":
        model = models.densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        for p in model.parameters():
            p.requires_grad = False

        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, N_CLASSES),
        )

        head = model.classifier

    else:
        raise ValueError(model_name)

    # Newly replaced classifier is trainable by default.
    return model, head


def set_warmup_trainability(model, head):
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True


def set_full_trainability(model):
    for p in model.parameters():
        p.requires_grad = True


def make_warmup_optimizer(model):
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=WARMUP_LR,
    )


def make_finetune_optimizer(model, head):
    head_ids = {id(p) for p in head.parameters()}

    backbone_params = [
        p for p in model.parameters()
        if id(p) not in head_ids
    ]
    head_params = list(head.parameters())

    return torch.optim.AdamW(
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


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# =============================================================================
# TRAINING / METRICS
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

    loss_sum = 0.0
    correct = 0
    n = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

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
        loss_sum += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += bs

    return {
        "loss": loss_sum / n,
        "accuracy": correct / n,
    }


def normalize_probs(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def metrics_from_probs(y_true, probs):
    probs = normalize_probs(probs)
    pred = probs.argmax(axis=1)
    y_bin = label_binarize(y_true, classes=np.arange(N_CLASSES))

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
        "macro_roc_auc_ovr": float(
            roc_auc_score(y_bin, probs, average="macro")
        ),
        "nll": float(
            log_loss(y_true, probs, labels=np.arange(N_CLASSES))
        ),
    }


@torch.no_grad()
def predict_holdout(
    model,
    loader,
    criterion,
    device,
    amp_enabled,
):
    model.eval()

    y_all = []
    p_all = []
    rows = []

    loss_sum = 0.0
    n = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits.float(), dim=1)
        probs_np = probs.cpu().numpy().astype(np.float64)
        probs_np = normalize_probs(probs_np)
        logits_np = logits.float().cpu().numpy()
        y_np = y.cpu().numpy()

        loss_sum += loss.item() * len(y_np)
        n += len(y_np)

        y_all.append(y_np)
        p_all.append(probs_np)

        pred_np = probs_np.argmax(axis=1)

        for i in range(len(y_np)):
            row = {
                "relative_path": batch["relative_path"][i],
                "absolute_path": batch["absolute_path"][i],
                "oof_fold": int(batch["oof_fold"][i]),
                "true_label_idx": int(y_np[i]),
                "true_class": IDX_TO_CLASS[int(y_np[i])],
                "pred_label_idx": int(pred_np[i]),
                "pred_class": IDX_TO_CLASS[int(pred_np[i])],
                "confidence": float(probs_np[i].max()),
            }

            for c, cls in enumerate(CLASS_NAMES):
                row[f"logit_{cls}"] = float(logits_np[i, c])
                row[f"prob_{cls}"] = float(probs_np[i, c])

            rows.append(row)

    y_true = np.concatenate(y_all)
    probs = np.concatenate(p_all, axis=0)

    metrics = metrics_from_probs(y_true, probs)
    metrics["cross_entropy_loss"] = loss_sum / n

    return metrics, rows


# =============================================================================
# CHECKPOINT / RESUME
# =============================================================================

def save_last_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    model_name,
    fold,
    phase,
    phase_epoch,
    history,
):
    torch.save(
        {
            "model_name": model_name,
            "fold": int(fold),
            "fold_seed": int(SEED + fold),
            "phase": phase,
            "phase_epoch": int(phase_epoch),
            "state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
            "validation_used": False,
            "test_used": False,
            "heldout_used_for_selection": False,
        },
        path,
    )


def save_json(path, obj):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# MERGE
# =============================================================================

def merge_if_complete(out_dir: Path, fold_df: pd.DataFrame):
    pred_paths = [
        out_dir / f"fold{fold}" / "heldout_predictions.csv"
        for fold in range(N_SPLITS)
    ]

    if not all(p.exists() for p in pred_paths):
        print("\nNot all 5 folds are complete yet; combined OOF file not created.")
        return

    oof = pd.concat(
        [pd.read_csv(p) for p in pred_paths],
        ignore_index=True,
    )

    if len(oof) != len(fold_df):
        raise RuntimeError(
            f"Combined OOF row count mismatch: {len(oof)} vs {len(fold_df)}"
        )

    if oof["relative_path"].duplicated().any():
        raise RuntimeError("Duplicate OOF predictions detected.")

    expected = set(fold_df["relative_path"].astype(str))
    observed = set(oof["relative_path"].astype(str))

    if expected != observed:
        raise RuntimeError("OOF coverage mismatch.")

    # Restore the exact original train CSV order.
    order = {
        p: i
        for i, p in enumerate(fold_df["relative_path"].astype(str))
    }

    oof["_order"] = oof["relative_path"].map(order)
    oof = (
        oof.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    combined_path = out_dir / "oof_predictions_all_folds.csv"
    oof.to_csv(combined_path, index=False, encoding="utf-8-sig")

    y_true = oof["true_label_idx"].to_numpy(dtype=int)
    probs = oof[
        [f"prob_{c}" for c in CLASS_NAMES]
    ].to_numpy(dtype=np.float64)

    probs = normalize_probs(probs)
    metrics = metrics_from_probs(y_true, probs)

    save_json(
        out_dir / "oof_metrics_all_folds.json",
        metrics,
    )

    print("\n" + "=" * 90)
    print("ALL 5 STRONG-MODEL OOF FOLDS COMPLETE")
    print("=" * 90)
    print("Combined OOF file:", combined_path)
    print(f"OOF Accuracy:     {metrics['accuracy']:.4f}")
    print(f"OOF Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    print(f"OOF Macro-F1:     {metrics['macro_f1']:.4f}")
    print(f"OOF Macro AUC:    {metrics['macro_roc_auc_ovr']:.4f}")
    print(f"OOF NLL:          {metrics['nll']:.4f}")


# =============================================================================
# MAIN FOLD RUN
# =============================================================================

def run_fold(
    model_name,
    fold,
    fold_df,
    out_dir,
    resume,
    device,
    n_gpus,
    amp_enabled,
    single_gpu,
):
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    pred_path = fold_dir / "heldout_predictions.csv"
    final_ckpt_path = fold_dir / "final_finetuned.pt"
    last_ckpt_path = fold_dir / "last_checkpoint.pt"
    history_path = fold_dir / "training_history.csv"

    if pred_path.exists() and final_ckpt_path.exists():
        print(f"\nFold {fold}: completed outputs found; skipping.")
        return

    if last_ckpt_path.exists() and not resume:
        raise RuntimeError(
            f"\nPartial checkpoint exists for fold {fold}:\n{last_ckpt_path}\n"
            "Use --resume to continue instead of overwriting it."
        )

    print("\n" + "-" * 90)
    print(f"STRONG OOF FOLD {fold}")
    print("-" * 90)

    seed_everything(SEED + fold)

    fit_df = fold_df[fold_df["oof_fold"] != fold].copy()
    hold_df = fold_df[fold_df["oof_fold"] == fold].copy()

    print("Fit images:     ", len(fit_df))
    print("Held-out images:", len(hold_df))

    train_tf, eval_tf = build_transforms()

    # Seeded shuffle generator for deterministic data order.
    g = torch.Generator()
    g.manual_seed(SEED + fold)

    fit_loader = DataLoader(
        MRIDataset(fit_df, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )

    hold_loader = DataLoader(
        MRIDataset(hold_df, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )

    model, head = build_imagenet_model(model_name)

    history = []
    resume_phase = None
    resume_phase_epoch = 0
    resume_optimizer_state = None
    resume_scaler_state = None

    if resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location="cpu")

        if ckpt.get("model_name") != model_name or int(ckpt.get("fold")) != fold:
            raise RuntimeError("Resume checkpoint model/fold mismatch.")

        if ckpt.get("split_fingerprint") != EXPECTED_SPLIT_FINGERPRINT:
            raise RuntimeError("Resume checkpoint split fingerprint mismatch.")

        model.load_state_dict(ckpt["state_dict"])

        resume_phase = str(ckpt["phase"])
        resume_phase_epoch = int(ckpt["phase_epoch"])
        resume_optimizer_state = ckpt["optimizer_state_dict"]
        resume_scaler_state = ckpt["scaler_state_dict"]
        history = list(ckpt.get("history", []))

        print(
            f"RESUME: phase={resume_phase}, "
            f"phase_epoch={resume_phase_epoch}"
        )

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # PHASE 1 — Frozen-backbone warm-up
    # -------------------------------------------------------------------------
    if resume_phase is None or resume_phase == "warmup":
        set_warmup_trainability(model, head)

        model = model.to(device)
        optimizer = make_warmup_optimizer(model)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        if resume_phase == "warmup":
            optimizer.load_state_dict(resume_optimizer_state)
            scaler.load_state_dict(resume_scaler_state)

        if device.type == "cuda" and n_gpus > 1 and not single_gpu:
            model = nn.DataParallel(model)
            print(f"DataParallel enabled across {n_gpus} GPUs.")

        start = resume_phase_epoch + 1 if resume_phase == "warmup" else 1

        if start <= WARMUP_EPOCHS:
            print("\nPHASE 1 — FROZEN-BACKBONE WARM-UP")

            for epoch in range(start, WARMUP_EPOCHS + 1):
                t0 = time.time()

                m = train_one_epoch(
                    model,
                    fit_loader,
                    optimizer,
                    criterion,
                    scaler,
                    device,
                    amp_enabled,
                )

                row = {
                    "phase": "warmup",
                    "phase_epoch": epoch,
                    "loss": m["loss"],
                    "accuracy": m["accuracy"],
                    "backbone_lr": 0.0,
                    "head_lr": WARMUP_LR,
                    "epoch_seconds": time.time() - t0,
                }
                history.append(row)

                pd.DataFrame(history).to_csv(
                    history_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                save_last_checkpoint(
                    last_ckpt_path,
                    model,
                    optimizer,
                    scaler,
                    model_name,
                    fold,
                    "warmup",
                    epoch,
                    history,
                )

                print(
                    f"Warm-up {epoch:02d}/{WARMUP_EPOCHS} | "
                    f"loss={m['loss']:.4f} | "
                    f"acc={m['accuracy']:.4f} | "
                    f"{row['epoch_seconds']:.1f}s"
                )

        # Prepare the underlying module for phase 2.
        model = unwrap(model)

    else:
        # Resuming directly inside phase 2.
        set_full_trainability(model)
        model = model.to(device)

    # -------------------------------------------------------------------------
    # PHASE 2 — Full-network fine-tuning
    # -------------------------------------------------------------------------
    fine_epochs = FINETUNE_EPOCHS[model_name]

    if resume_phase == "finetune":
        set_full_trainability(model)

        optimizer = make_finetune_optimizer(model, head)
        optimizer.load_state_dict(resume_optimizer_state)

        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        scaler.load_state_dict(resume_scaler_state)

        fine_start = resume_phase_epoch + 1
    else:
        # Exact full-network fine-tuning transition: new AdamW optimizer and new scaler.
        set_full_trainability(model)

        optimizer = make_finetune_optimizer(model, head)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        fine_start = 1

    if device.type == "cuda" and n_gpus > 1 and not single_gpu:
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)

    print("\nPHASE 2 — FULL-NETWORK FINE-TUNING")
    print("Fixed epochs:", fine_epochs)
    print("Backbone LR:", BACKBONE_LR)
    print("Head LR:", HEAD_LR)
    print("No held-out-fold model selection.")
    print("No held-out-fold LR scheduling.")

    for epoch in range(fine_start, fine_epochs + 1):
        t0 = time.time()

        m = train_one_epoch(
            model,
            fit_loader,
            optimizer,
            criterion,
            scaler,
            device,
            amp_enabled,
        )

        row = {
            "phase": "finetune",
            "phase_epoch": epoch,
            "loss": m["loss"],
            "accuracy": m["accuracy"],
            "backbone_lr": BACKBONE_LR,
            "head_lr": HEAD_LR,
            "epoch_seconds": time.time() - t0,
        }
        history.append(row)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
            encoding="utf-8-sig",
        )

        save_last_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            scaler,
            model_name,
            fold,
            "finetune",
            epoch,
            history,
        )

        print(
            f"Fine-tune {epoch:02d}/{fine_epochs} | "
            f"loss={m['loss']:.4f} | "
            f"acc={m['accuracy']:.4f} | "
            f"{row['epoch_seconds']:.1f}s"
        )

    # -------------------------------------------------------------------------
    # Predict held-out fold ONLY after fixed training is complete.
    # -------------------------------------------------------------------------
    torch.save(
        {
            "model_name": model_name,
            "fold": int(fold),
            "fold_seed": int(SEED + fold),
            "warmup_epochs": WARMUP_EPOCHS,
            "finetune_epochs": fine_epochs,
            "state_dict": unwrap(model).state_dict(),
            "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
            "validation_used": False,
            "test_used": False,
            "heldout_used_for_selection": False,
        },
        final_ckpt_path,
    )

    print("\nTraining schedule complete.")
    print("Now evaluating the held-out OOF fold once...")

    metrics, rows = predict_holdout(
        model,
        hold_loader,
        criterion,
        device,
        amp_enabled,
    )

    pd.DataFrame(rows).to_csv(
        pred_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        fold_dir / "heldout_metrics.json",
        metrics,
    )

    print(
        f"\nFold {fold} held-out | "
        f"acc={metrics['accuracy']:.4f} | "
        f"balAcc={metrics['balanced_accuracy']:.4f} | "
        f"mF1={metrics['macro_f1']:.4f} | "
        f"AUC={metrics['macro_roc_auc_ovr']:.4f}"
    )

    # Completion summary.
    save_json(
        fold_dir / "fold_summary.json",
        {
            "model": model_name,
            "fold": fold,
            "fit_n": len(fit_df),
            "heldout_n": len(hold_df),
            "warmup_epochs": WARMUP_EPOCHS,
            "finetune_epochs": fine_epochs,
            "heldout_used_for_selection": False,
            "validation_used": False,
            "test_used": False,
            "heldout_metrics": metrics,
        },
    )

    del model, optimizer, scaler, fit_loader, hold_loader

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["efficientnet_b2", "densenet121"],
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=list(range(N_SPLITS)),
        default=None,
        help="Run one fold. Omit to process all incomplete folds sequentially.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partial folds and skip completed folds.",
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
    )

    parser.add_argument(
        "--single_gpu",
        action="store_true",
    )

    args = parser.parse_args()

    source_df = verify_source()
    fold_df = make_folds(source_df)

    out_dir = (
        PROJECT_ROOT
        / "ordinary_oof"
        / args.model
        / f"seed{SEED}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_df.to_csv(
        out_dir / "oof_fold_assignments_seed42.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fine_epochs = FINETUNE_EPOCHS[args.model]

    config = {
        "model": args.model,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "source": "train_seed42.csv only",
        "validation_used": False,
        "test_used": False,
        "heldout_used_for_selection": False,
        "pretraining": "ImageNet",
        "warmup": {
            "backbone_frozen": True,
            "epochs": WARMUP_EPOCHS,
            "optimizer": "Adam",
            "learning_rate": WARMUP_LR,
        },
        "finetune": {
            "full_network_unfrozen": True,
            "epochs": fine_epochs,
            "optimizer": "AdamW",
            "backbone_lr": BACKBONE_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "none_in_OOF_fixed_schedule",
        },
        "batch_size_total": BATCH_SIZE,
        "dropout": DROPOUT,
        "input_size": IMAGE_SIZE,
        "augmentation": {
            "rotation_degrees": ROTATION_DEGREES,
            "zoom_range": ZOOM_RANGE,
        },
    }

    save_json(out_dir / "oof_config.json", config)

    print("\nOOF FOLD COUNTS")
    for fold in range(N_SPLITS):
        part = fold_df[fold_df["oof_fold"] == fold]

        print(f"\nFold {fold}: {len(part)}")
        for cls in CLASS_NAMES:
            print(
                f"  {cls:25s}: "
                f"{(part['class_name'] == cls).sum()}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    n_gpus = (
        torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0
    )
    amp_enabled = device.type == "cuda" and not args.no_amp

    print("\n" + "=" * 90)
    print("ORDINARY OOF — STRONG-MODEL 5-FOLD OOF")
    print("=" * 90)
    print("Model:", args.model)
    print("Device:", device)
    print("Visible CUDA GPUs:", n_gpus)
    print("AMP:", amp_enabled)
    print("Batch size TOTAL:", BATCH_SIZE)
    print("Warm-up epochs/fold:", WARMUP_EPOCHS)
    print("Fine-tune epochs/fold:", fine_epochs)
    print("Output:", out_dir)

    if torch.cuda.is_available():
        for i in range(n_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    folds = (
        [args.fold]
        if args.fold is not None
        else list(range(N_SPLITS))
    )

    for fold in folds:
        run_fold(
            model_name=args.model,
            fold=fold,
            fold_df=fold_df,
            out_dir=out_dir,
            resume=args.resume,
            device=device,
            n_gpus=n_gpus,
            amp_enabled=amp_enabled,
            single_gpu=args.single_gpu,
        )

    merge_if_complete(out_dir, fold_df)

    print("\nVALIDATION STATUS: NOT LOADED")
    print("TEST STATUS:       NOT LOADED")


if __name__ == "__main__":
    main()
