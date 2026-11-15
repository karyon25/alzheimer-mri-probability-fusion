# Probability-Level CNN Fusion for Four-Class Alzheimer MRI Classification

This repository contains the code and research artifacts for a comparison of equal soft voting (EqualSoft), global out-of-fold performance weighting (GlobalOOF), and class-aware out-of-fold performance weighting (ClassAwareOOF), using EfficientNet-B2 and DenseNet121. The evaluation distinguishes hard-label classification performance from probability quality and incorporates fold-pair exclusion when estimating learned fusion weights.

The experiments concern **four-class, two-dimensional MRI image-label classification** on a publicly distributed, pre-augmented dataset. The partitions do not have verified participant, scan-family, or augmentation-family independence. Fold-pair exclusion controls a specific dependency in fusion-weight estimation; it does not establish patient-level generalization.

## Repository contents

| Location | Contents | Methodological correspondence |
|---|---|---|
| `splits/split_manifest_seed42.csv` | Complete frozen image-level split manifest | Dataset partitioning (Section 3.1) |
| `splits/train_seed42.csv`, `splits/val_seed42.csv`, `splits/test_seed42.csv` | Frozen training, validation, and internal-benchmark partitions | Dataset partitioning (Section 3.1) |
| `splits/oof_fold_assignments_seed42.csv` | Frozen five-fold assignments for the training partition | Ordinary OOF and pair-exclusion evaluation (Sections 3.4 and 3.6) |
| `splits/split_summary_seed42.txt` | Class counts and split fingerprint | Dataset partitioning (Section 3.1) |
| `metadata/seed_configuration.json` | Split, fold, model-fit, and excluded-pair random seeds; fusion-weight search settings | Evaluation implementation (Sections 3.1, 3.4–3.6) |
| `code/frozen_backbone_training.py` | Fixed ten-epoch frozen-backbone training | Constituent training (Section 3.3) |
| `code/full_network_finetuning.py` | Full-network fine-tuning using the frozen training protocol | Constituent training (Section 3.3) |
| `code/ordinary_oof_predictions.py` | Ordinary five-fold OOF probability generation | Ordinary OOF generation (Section 3.4) |
| `code/pair_exclusion_fusion_evaluation.py` | Pair-excluded constituent refits, outer-fold fusion evaluation, and global/class-aware weight searches | Fusion rules and strict pair-exclusion (Sections 3.5–3.6) |
| `code/verify_frozen_splits.py` | Frozen split integrity and fingerprint validator | Partition verification |
| `code/configure_local_paths.py` | Local path configuration utility; not part of the original experiment | Environment configuration |
| `code/benchmark_inference.py`, `results/inference_profile.csv` | FP32 single-GPU inference profiling and recorded measurements | Computational efficiency |
| `metadata/verification_record.md` | Record of artifact-integrity and fold-assignment checks | Reproducibility verification |
| `requirements.txt` | Direct Python package dependencies; not a frozen historical environment | Software dependencies |
| `SHA256SUMS.txt` | SHA-256 hashes of repository files in this distribution | Artifact integrity |

## Dataset and frozen split

Source dataset: [Augmented Alzheimer MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset). MRI images are not included in this repository.

The frozen split contains **33,984 images**: **27,187 training**, **3,398 validation**, and **3,399 internal-benchmark** images. In the original project, the internal-benchmark partition is stored as `test_seed42.csv`. Class names in the data are `MildDemented`, `ModerateDemented`, `NonDemented`, and `VeryMildDemented`.

The manifest records the original `relative_path`, `absolute_path`, `class_name`, and `split` fields. The `absolute_path` entries and the `PROJECT_ROOT` constants in the training scripts record the historical Windows experiment directory and can be configured for a different local environment. The relative paths, class labels, row order, and split membership identify the frozen dataset partitions.

- Partition random seed: **42**.
- Manifest SHA-256 fingerprint: `7efad99c096dec1fecee39f260c1984c063c5f17c161e82304824f94c8d62aa7`.
- Fingerprint construction: SHA-256 over the lexicographically ordered records `relative_path|class_name|split`, joined by newlines, as implemented in `code/verify_frozen_splits.py`.
- OOF partition: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` applied to the training CSV in its preserved row order.
- Ordinary OOF model-fit seeds for folds 0–4: **42, 43, 44, 45, 46**.
- Pair-exclusion refit seeds: `42 + 1000 + 10*a + b` for each unordered excluded pair `a < b`.

The recorded fold assignment file was cross-checked against the archived strict-OOF evaluation rows, with **27,187 of 27,187 assignments matching**. The verification details are in `metadata/verification_record.md`.

## Fusion methods and evaluation

For each outer evaluation fold, the learned fusion combiner is fitted using predictions for the other four folds from constituent models trained without **both** the outer evaluation fold and each corresponding inner prediction fold. Ten distinct unordered fold-pair fits per architecture supply these probabilities. The fitted weights are evaluated on the outer fold using separately generated ordinary OOF constituent predictions. EqualSoft requires no weight fitting.

- **EqualSoft:** arithmetic mean of the two constituent probability vectors.
- **GlobalOOF:** EfficientNet-B2 coefficient searched from **0 to 1** at **0.005** increments, with DenseNet121 receiving the complementary coefficient. The primary optimization objective is **Macro-F1**.
- **ClassAwareOOF:** class-specific EfficientNet-B2 coefficients optimized by multi-start coordinate search; initial states are equal weights and the fitted global solution. The coarse step is **0.05**, followed by a **±0.05** local refinement at **0.005** increments, for up to **three passes per search phase**. Ties are resolved by balanced accuracy and then proximity to equal weighting.

Macro-F1 is the primary hard-label classification endpoint. Negative log-likelihood, multiclass Brier score, and 15-bin expected calibration error separately characterize probability quality. Improvements in these endpoints do not necessarily coincide.

## Running the supplied code

The commands below are executed from the repository root. The original MRI images and compatible Python/GPU dependencies are required for constituent-model training. The `requirements.txt` file enumerates direct imported dependencies but does not encode a complete frozen environment.

```bash
python code/verify_frozen_splits.py --split-dir splits
```

For local dataset and project directories, the path utility can update the historical path fields and root constants in a working copy:

```bash
python code/configure_local_paths.py --dataset-root "PATH_TO_AugmentedAlzheimerDataset"
python code/verify_frozen_splits.py --split-dir splits
```

The model-training workflow uses the frozen-backbone training script, full-network fine-tuning script, and ordinary OOF prediction script before the strict pair-exclusion evaluation script. The latter consumes constituent OOF artifacts generated in the preceding steps. The source headers describe the corresponding execution arguments and required intermediate files. Model checkpoints, intermediate probability files, and MRI images are not redistributed here.

The checksum file represents the published repository distribution. Local path reconfiguration changes some file hashes but not the fingerprinted split identities or fold assignments.

## Inference profiling

`code/benchmark_inference.py` and `results/inference_profile.csv` document the model-forward profiling experiment at 224 × 224 input resolution. Profiling used FP32 inference on one NVIDIA TITAN RTX (`cuda:0`), with measurements at batch sizes 1 and 32, 50 warm-up iterations, 200 timed iterations per repeat, and five repeats. Inputs were already present in GPU memory. Counted MACs were doubled to calculate GOP/image; operations not supported by the counter are reported separately. The resulting effective GOP/s is not peak device throughput. The archived profiling CSV preserves the checkpoint-source paths recorded during the original measurement; these entries are provenance metadata rather than portable file locations.

## Reproducibility scope

The frozen manifests, split fingerprint, source syntax, stored fold assignments, and profiling-table structure were checked when assembling these files. The training and pair-exclusion calculations were **not rerun end-to-end** as part of this repository distribution. The original experiment requires the downloaded dataset, trained constituent models or their regeneration, intermediate OOF predictions, and suitable GPU resources.

The image-level split does not verify independence across participant, examination, adjacent-slice, or augmentation families. Results therefore describe the specified image-level benchmark rather than clinically validated diagnosis of independent patients.
