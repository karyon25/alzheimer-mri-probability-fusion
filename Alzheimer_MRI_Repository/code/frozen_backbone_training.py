"""
Frozen-backbone training — Fixed ten-epoch constituent training
=========================================================

Purpose
-------
Train the two final constituent models on the frozen 80% training partition
using the frozen initial-training protocol adopted in model development.

IMPORTANT
---------
- Uses ONLY train_seed42.csv.
- Does NOT load validation or test CSVs.
- Trains exactly 10 epochs.
- No validation-based checkpoint selection.
- Saves ONLY the fixed epoch-10 final model for later one-time test evaluation.

Models
------
- EfficientNet-B2
- DenseNet121

Protocol
--------
- ImageNet pretrained torchvision weights
- frozen backbone
- input 224 x 224 RGB
- Dropout(0.5) + 4-class linear head
- Adam, lr=1e-4
- cross-entropy
- total batch size 32
- rotation +/-10 degrees
- zoom 0.90-1.10
- ImageNet normalization
- AMP enabled on CUDA
- DataParallel enabled automatically when 2 GPUs are visible
"""

from __future__ import annotations

import argparse
import csv
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


# ---------------------------------------------------------------------
# LOCKED STUDY CONFIG
# ---------------------------------------------------------------------

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
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DROPOUT = 0.5

NUM_WORKERS = 0  # Windows-safe
PIN_MEMORY = True

ROTATION_DEGREES = 10
ZOOM_RANGE = (0.90, 1.10)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# SPLIT VERIFICATION
# ---------------------------------------------------------------------

def compute_manifest_fingerprint(df: pd.DataFrame) -> str:
    required = {"relative_path", "class_name", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    canonical = "\n".join(
        f"{r['relative_path']}|{r['class_name']}|{r['split']}"
        for _, r in df.sort_values("relative_path").iterrows()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_frozen_split() -> pd.DataFrame:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(MANIFEST_CSV)
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)

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

    if len(train_df) != 27187:
        raise RuntimeError(
            f"Expected 27,187 training images; found {len(train_df)}."
        )

    print("Frozen split verification: PASSED")
    print("Training images:", len(train_df))
    print("Validation CSV is intentionally not loaded.")
    print("Test CSV is intentionally not loaded.")

    return train_df


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------

class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

        required = {
            "relative_path",
            "absolute_path",
            "class_name",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Training CSV missing columns: {sorted(missing)}")

        unknown = sorted(
            set(self.df["class_name"].astype(str)) - set(CLASS_NAMES)
        )
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(str(row["absolute_path"]))

        with Image.open(path) as img:
            img = img.convert("RGB")

        image = self.transform(img)
        label = CLASS_TO_IDX[str(row["class_name"])]

        return image, torch.tensor(label, dtype=torch.long)


def build_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomAffine(
            degrees=ROTATION_DEGREES,
            scale=ZOOM_RANGE,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def build_model(model_name: str) -> nn.Module:
    if model_name == "efficientnet_b2":
        model = models.efficientnet_b2(
            weights=EfficientNet_B2_Weights.IMAGENET1K_V1
        )

        for p in model.parameters():
            p.requires_grad = False

        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, len(CLASS_NAMES)),
        )

    elif model_name == "densenet121":
        model = models.densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        for p in model.parameters():
            p.requires_grad = False

        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT),
            nn.Linear(in_features, len(CLASS_NAMES)),
        )

    else:
        raise ValueError(model_name)

    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

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
    n_seen = 0
    n_correct = 0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        running_loss += (
            loss.item() * bs
        )
        n_seen += bs

        pred = logits.detach().argmax(
            dim=1
        )
        n_correct += int(
            (pred == labels).sum().item()
        )

    return {
        "train_loss":
            running_loss / max(n_seen, 1),
        "train_accuracy":
            n_correct / max(n_seen, 1),
    }


# ---------------------------------------------------------------------
# SAVE HELPERS
# ---------------------------------------------------------------------

def save_json(path: Path, obj):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

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
        "--no_amp",
        action="store_true",
    )

    parser.add_argument(
        "--single_gpu",
        action="store_true",
    )

    args = parser.parse_args()

    seed_everything(SEED)
    train_df = verify_frozen_split()

    output_dir = (
        PROJECT_ROOT
        / "frozen_backbone_models"
        / args.model
        / f"seed{SEED}"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    n_gpus = (
        torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("\n" + "=" * 82)
    print(
        "FROZEN-BACKBONE TRAINING — FINAL FIXED-10-EPOCH "
        "CONSTITUENT MODEL"
    )
    print("=" * 82)
    print("Model:", args.model)
    print("Device:", device)
    print("Visible CUDA GPUs:", n_gpus)
    print("AMP:", amp_enabled)
    print("Batch size TOTAL:", BATCH_SIZE)
    print("Epochs FIXED:", EPOCHS)
    print("Learning rate:", LR)
    print("Output:", output_dir)

    if torch.cuda.is_available():
        for i in range(n_gpus):
            print(
                f"GPU {i}: "
                f"{torch.cuda.get_device_name(i)}"
            )

    transform = build_train_transform()

    train_ds = MRIDataset(
        train_df,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
        persistent_workers=False,
        drop_last=False,
    )

    model = build_model(
        args.model
    )

    total_params, trainable_params = (
        count_parameters(model)
    )

    print(
        f"\nTotal parameters:     "
        f"{total_params:,}"
    )
    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    model = model.to(device)

    if (
        device.type == "cuda"
        and n_gpus > 1
        and not args.single_gpu
    ):
        model = nn.DataParallel(
            model
        )
        print(
            f"DataParallel enabled "
            f"across {n_gpus} GPUs."
        )
    else:
        print(
            "DataParallel disabled."
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=LR,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    config = {
        "training_phase":
            "frozen_backbone_fixed10",
        "model":
            args.model,
        "seed":
            SEED,
        "split_fingerprint":
            EXPECTED_SPLIT_FINGERPRINT,
        "train_csv":
            str(TRAIN_CSV),
        "n_train":
            int(len(train_df)),
        "validation_used":
            False,
        "test_used":
            False,
        "pretraining":
            "ImageNet",
        "backbone_frozen":
            True,
        "input_size":
            [224, 224, 3],
        "optimizer":
            "Adam",
        "learning_rate":
            LR,
        "loss":
            "CrossEntropyLoss",
        "batch_size_total":
            BATCH_SIZE,
        "epochs_fixed":
            EPOCHS,
        "dropout":
            DROPOUT,
        "rotation_degrees":
            ROTATION_DEGREES,
        "zoom_scale":
            list(ZOOM_RANGE),
        "normalization_mean":
            list(IMAGENET_MEAN),
        "normalization_std":
            list(IMAGENET_STD),
        "amp":
            amp_enabled,
        "data_parallel":
            isinstance(
                model,
                nn.DataParallel,
            ),
        "total_parameters":
            total_params,
        "trainable_parameters":
            trainable_params,
        "checkpoint_selection":
            "none_fixed_epoch10",
        "torch_version":
            torch.__version__,
    }

    save_json(
        output_dir / "config.json",
        config,
    )

    history = []
    start_all = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        t0 = time.time()

        result = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        sec = time.time() - t0

        row = {
            "epoch":
                epoch,
            "train_loss":
                result["train_loss"],
            "train_accuracy":
                result["train_accuracy"],
            "seconds":
                sec,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss="
            f"{result['train_loss']:.4f} | "
            f"train acc="
            f"{result['train_accuracy']:.4f} | "
            f"{sec:.1f}s"
        )

        pd.DataFrame(
            history
        ).to_csv(
            output_dir
            / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

    total_seconds = (
        time.time() - start_all
    )

    ckpt_path = (
        output_dir
        / "final_epoch10.pt"
    )

    torch.save(
        {
            "training_phase":
                "frozen_backbone_fixed10",
            "model_name":
                args.model,
            "epoch":
                EPOCHS,
            "seed":
                SEED,
            "class_names":
                CLASS_NAMES,
            "split_fingerprint":
                EXPECTED_SPLIT_FINGERPRINT,
            "validation_used":
                False,
            "test_used":
                False,
            "state_dict":
                unwrap(model).state_dict(),
        },
        ckpt_path,
    )

    summary = {
        "model":
            args.model,
        "final_epoch":
            EPOCHS,
        "final_train_loss":
            history[-1]["train_loss"],
        "final_train_accuracy":
            history[-1]["train_accuracy"],
        "total_training_seconds":
            total_seconds,
        "checkpoint":
            str(ckpt_path),
        "validation_used":
            False,
        "test_used":
            False,
    }

    save_json(
        output_dir
        / "final_training_summary.json",
        summary,
    )

    print("\n" + "=" * 82)
    print("FROZEN-BACKBONE TRAINING COMPLETE")
    print("=" * 82)
    print("Model:", args.model)
    print(
        "Final checkpoint:",
        ckpt_path,
    )
    print(
        f"Final train loss: "
        f"{history[-1]['train_loss']:.4f}"
    )
    print(
        f"Final train accuracy: "
        f"{history[-1]['train_accuracy']:.4f}"
    )
    print(
        "\nVALIDATION STATUS: "
        "NOT USED"
    )
    print(
        "TEST STATUS: "
        "UNTOUCHED"
    )
    print(
        "\nDo not replace this checkpoint "
        "based on future test behavior."
    )


if __name__ == "__main__":
    main()
