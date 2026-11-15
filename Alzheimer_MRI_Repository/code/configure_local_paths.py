#!/usr/bin/env python3
"""Optional local path helper, not part of the original experiment.

Run inside a disposable clone after downloading the original Kaggle images.
Rewrites ONLY machine-specific ``absolute_path`` CSV fields and the original
PROJECT_ROOT constants in the four training and evaluation scripts. Split identities, row order,
class labels and fold assignments are unchanged. SHA256SUMS will no longer match
files deliberately reconfigured for a user's local environment.
"""
import argparse
import csv
from pathlib import Path, PureWindowsPath
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True, help='Folder containing the four original class subfolders')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    a = parser.parse_args()
    project_root = a.project_root.expanduser().resolve()
    dataset_root = a.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f'Dataset directory not found: {dataset_root}')
    if not (project_root/'splits/train_seed42.csv').exists():
        parser.error(f'Repository split files not found under: {project_root}')
    sample_count = 0
    for filename in ('split_manifest_seed42.csv', 'train_seed42.csv', 'val_seed42.csv', 'test_seed42.csv'):
        file = project_root/'splits'/filename
        with file.open('r', encoding='utf-8-sig', newline='') as f:
            reader=csv.DictReader(f)
            fields=list(reader.fieldnames or [])
            rows=list(reader)
        if not {'relative_path','absolute_path','class_name','split'}.issubset(fields):
            raise ValueError(f'Unexpected columns in {file}')
        for row in rows:
            rel=PureWindowsPath(row['relative_path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Unsafe relative path: {row["relative_path"]}')
            row['absolute_path']=str(dataset_root.joinpath(*rel.parts))
            sample_count+=1
        with file.open('w', encoding='utf-8-sig', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
        print(f'Updated paths: {file.name} ({len(rows)} rows)')
    edited=[]
    for filename in ('frozen_backbone_training.py', 'full_network_finetuning.py',
                     'ordinary_oof_predictions.py', 'pair_exclusion_fusion_evaluation.py'):
        file=project_root/'code'/filename
        source=file.read_text(encoding='utf-8')
        pattern=r'^PROJECT_ROOT\s*=\s*Path\([^\n]*\)\s*$'
        replacement='PROJECT_ROOT = Path(' + repr(str(project_root)) + ')'
        revised,n=re.subn(pattern,lambda _:replacement,source,count=1,flags=re.MULTILINE)
        if n!=1: raise ValueError(f'Cannot find unique PROJECT_ROOT constant in {file}')
        file.write_text(revised,encoding='utf-8')
        edited.append(filename)
    print('Updated project paths:',', '.join(edited))
    print(f'Updated {sample_count} CSV absolute_path entries in four manifests.')
    print('Next: python code/verify_frozen_splits.py --split-dir splits')
    print('Note: SHA256SUMS.txt describes the original published source, not a locally reconfigured clone.')

if __name__=='__main__':
    main()
