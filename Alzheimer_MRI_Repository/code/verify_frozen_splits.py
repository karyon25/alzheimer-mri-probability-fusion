#!/usr/bin/env python3
"""Validate the frozen Alzheimer MRI split files used by the study.

Run from the extracted bundle root:
    python code/verify_frozen_splits.py --split-dir splits
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

EXPECTED_FINGERPRINT = "7efad99c096dec1fecee39f260c1984c063c5f17c161e82304824f94c8d62aa7"
EXPECTED_ROWS = {"manifest": 33984, "train": 27187, "val": 3398, "test": 3399}
REQUIRED = {"relative_path", "absolute_path", "class_name", "split"}


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if len(df) != EXPECTED_ROWS[name]:
        raise ValueError(f"{name}: expected {EXPECTED_ROWS[name]} rows; found {len(df)}")
    return df


def fingerprint(df: pd.DataFrame) -> str:
    required_for_hash = {"relative_path", "class_name", "split"}
    if not required_for_hash.issubset(df.columns):
        raise ValueError("Manifest lacks relative_path, class_name, or split")
    canonical = "\n".join(
        f"{row['relative_path']}|{row['class_name']}|{row['split']}"
        for _, row in df.sort_values("relative_path").iterrows()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    args = parser.parse_args()
    root = args.split_dir

    manifest = load_csv(root / "split_manifest_seed42.csv", "manifest")
    parts = {name: load_csv(root / f"{name}_seed42.csv", name)
             for name in ("train", "val", "test")}

    if manifest["relative_path"].duplicated().any():
        raise ValueError("manifest: duplicate relative_path values")
    got = fingerprint(manifest)
    if got != EXPECTED_FINGERPRINT:
        raise ValueError(f"Fingerprint mismatch: expected {EXPECTED_FINGERPRINT}; found {got}")

    for name, split_df in parts.items():
        if set(split_df["split"].astype(str)) != {name}:
            raise ValueError(f"{name}: unexpected split labels")
        manifest_part = manifest.loc[manifest["split"].astype(str) == name, split_df.columns]
        left = split_df.sort_values("relative_path").reset_index(drop=True)
        right = manifest_part.sort_values("relative_path").reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"{name}: file rows do not exactly match manifest partition")

    summary_path = root / "split_summary_seed42.txt"
    if summary_path.is_file() and got not in summary_path.read_text(encoding="utf-8").lower():
        raise ValueError("split summary does not contain the computed fingerprint")

    print("Frozen split validation: PASS")
    print(f"Manifest fingerprint: {got}")
    print(f"Rows: manifest={len(manifest)}, train={len(parts['train'])}, "
          f"val={len(parts['val'])}, test={len(parts['test'])}")
    print("Train/validation/test records exactly match manifest partitions.")


if __name__ == "__main__":
    main()
