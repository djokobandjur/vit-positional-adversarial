"""
Extract noise ablation and probe analysis data for all 12 ImageNet-100 models.

Generates analysis_data.json containing:
  - noise_ablation: accuracy under Gaussian noise added to PE buffers
    (8 noise levels, 0.0 to 5.0; used for Fig 2 left panel — decoupling thesis)
  - probe: row/column/position probe accuracy (auxiliary, not used in main
    paper narrative)
  - accuracy_no_pe: accuracy with PE zeroed out (bonus)

This script supports ImageNet-100 models only (224x224, patch_size=16).
CIFAR-100 equivalent is not provided in this toolkit; the decoupling
thesis figures rely on ImageNet-100 numbers from this script.

Requires:
  - GPU
  - Trained ImageNet-100 models (4 PE types × 3 seeds)
  - ImageNet-100 val/ directory in ImageFolder format
  - full_scale_experiment.py importable (provides VisionTransformer,
    extract_positional_embedding, probe_analysis, noise_ablation)

Usage:
    python extract_tables_data.py \\
        --models_dir "/path/to/Trained models_ImageNet100" \\
        --val_dir "/path/to/imagenet100/val" \\
        --output_path "/path/to/analysis_data.json"

Optional:
    --batch_size 128       (default: 128)
    --seeds 42 123 456     (default: 42 123 456)
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import from the main experiment script. We assume extract_tables_data.py
# is run from a directory where full_scale_experiment.py is importable
# (either same directory, or full_scale_experiment.py's directory is on
# sys.path). For Colab compatibility, '/content' is added as a fallback.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/content')
from full_scale_experiment import (
    VisionTransformer,
    extract_positional_embedding,
    probe_analysis,
    noise_ablation,
)


PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
DEFAULT_SEEDS = [42, 123, 456]


def build_val_loader(val_dir, batch_size):
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(val_dir, val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Val: {len(val_dataset)} images from {val_dir}")
    return val_loader


def run_extraction(models_dir, val_dir, output_path, batch_size, seeds):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    val_loader = build_val_loader(val_dir, batch_size)

    all_results = {}

    for pe_type in PE_TYPES:
        all_results[pe_type] = {}
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"Processing: {pe_type} seed={seed}")
            print(f"{'='*50}")

            model_path = os.path.join(models_dir, f'{pe_type}_seed{seed}', 'best_model.pth')
            if not os.path.exists(model_path):
                print(f"  SKIP: {model_path} not found")
                continue

            torch.manual_seed(seed)
            model = VisionTransformer(
                img_size=224, patch_size=16, num_classes=100, embed_dim=768,
                depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
            ).to(device)

            state = torch.load(model_path, map_location=device)
            model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
            model.eval()

            result = {}

            # 1. Noise ablation
            # noise_ablation() in full_scale_experiment.py already prints
            # per-level results internally; we just store the dict.
            # Format returned:
            #   {'noise_levels': [0.0, 0.1, ..., 5.0],
            #    'accuracies':   [acc0, acc1, ..., acc7],
            #    'accuracy_no_pe': acc_no_pe}
            print("  Running noise ablation...")
            noise_results = noise_ablation(model, val_loader, device, pe_type)
            result['noise_ablation'] = noise_results

            # 2. Probe analysis (auxiliary, not used in main paper narrative)
            print("  Running probe analysis...")
            pe_matrix = extract_positional_embedding(model, pe_type)
            if pe_matrix is not None:
                probe_results = probe_analysis(pe_matrix, num_patches_per_side=14)
                result['probe'] = probe_results
                for task, vals in probe_results.items():
                    print(f"    Probe {task}: {vals['mean']:.1f}% +/- {vals['std']:.1f}%")

            all_results[pe_type][seed] = result

            # Save incrementally (resume-friendly if Colab session dies)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved to {output_path}")

    print("\n" + "="*50)
    print(f"DONE! All results saved to {output_path}")
    print("="*50)

    return all_results


def print_summary(all_results, seeds):
    """Print formatted summary tables for quick sanity check."""

    print("\n\nNOISE ABLATION SUMMARY (for Table 2):")
    print("-" * 80)
    noise_level_floats = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    noise_level_labels = [f"{x:.1f}x" for x in noise_level_floats] + ['no_pe']
    header = f"{'Level':<10}" + "".join(f"{pe:<15}" for pe in PE_TYPES)
    print(header)
    for label, level_val in zip(noise_level_labels,
                                noise_level_floats + [None]):
        row = f"{label:<10}"
        for pe in PE_TYPES:
            vals = []
            for s in seeds:
                pe_seed = all_results.get(pe, {}).get(s, {})
                na = pe_seed.get('noise_ablation')
                if na is None:
                    continue
                if level_val is None:
                    # 'no_pe' row
                    if 'accuracy_no_pe' in na:
                        vals.append(float(na['accuracy_no_pe']))
                else:
                    # Match parallel-array index by noise level
                    levels_arr = na.get('noise_levels', [])
                    accs_arr = na.get('accuracies', [])
                    for nl, ac in zip(levels_arr, accs_arr):
                        if abs(float(nl) - level_val) < 1e-9:
                            vals.append(float(ac))
                            break
            if vals:
                row += f"{np.mean(vals):.2f}±{np.std(vals):.2f}  "
            else:
                row += f"{'N/A':<15}"
        print(row)

    print("\n\nPROBE ANALYSIS SUMMARY (for Table 3):")
    print("-" * 60)
    for pe in PE_TYPES:
        for task in ['row', 'column', 'position']:
            vals = []
            for s in seeds:
                if s in all_results.get(pe, {}) and 'probe' in all_results[pe][s]:
                    vals.append(all_results[pe][s]['probe'][task]['mean'])
            if vals:
                print(f"  {pe:12s} {task:10s}: {np.mean(vals):.1f} ± {np.std(vals):.1f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract noise ablation and probe analysis data for ImageNet-100 PE models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        required=True,
        help='Directory containing trained ImageNet-100 models, organized as '
             '<models_dir>/<pe_type>_seed<seed>/best_model.pth',
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        required=True,
        help='Path to ImageNet-100 val/ directory in ImageFolder format '
             '(100 class subdirectories, 50 images each)',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for analysis_data.json',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Validation loader batch size (default: 128)',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help='Seeds to process (default: 42 123 456)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = run_extraction(
        models_dir=args.models_dir,
        val_dir=args.val_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        seeds=args.seeds,
    )
    print_summary(results, args.seeds)
