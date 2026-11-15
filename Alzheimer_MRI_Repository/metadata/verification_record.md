# Reproducibility verification record

## Data partitions

- The source manifests contain 33,984 image records: 27,187 training, 3,398 validation, and 3,399 internal-benchmark records.
- The training, validation, and internal-benchmark CSVs exactly match their respective subsets in the complete manifest.
- The complete manifest contains no duplicate `relative_path` values.
- The frozen SHA-256 fingerprint is `7efad99c096dec1fecee39f260c1984c063c5f17c161e82304824f94c8d62aa7`, computed from ordered `relative_path|class_name|split` records.

## Out-of-fold partition verification

- The five-fold assignment uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the original training CSV order.
- Reconstructed assignments were compared with the archived strict-OOF prediction rows, yielding 27,187 matched rows and zero mismatched fold assignments.
- Ten unordered pairs of excluded folds are used per constituent architecture; the per-pair seed rule and seed values are recorded in `seed_configuration.json`.

## Code and profiling artifacts

- The supplied Python source files were checked for syntax correctness.
- GlobalOOF grid search and ClassAwareOOF multi-start coordinate search are implemented in `code/pair_exclusion_fusion_evaluation.py`.
- The inference profile records results for EfficientNet-B2, DenseNet121, and EqualSoft at batch sizes 1 and 32, with single-GPU FP32 timing and counted operation estimates.
- SHA-256 checksums for this distribution are listed in `SHA256SUMS.txt`.

## Scope of verification

These checks establish consistency of the distributed manifests, fold assignments, script syntax, and profiling records. They are not an independent end-to-end repetition of CNN training or inference. The original dataset and generated checkpoints/prediction artifacts are not included in the distribution.
