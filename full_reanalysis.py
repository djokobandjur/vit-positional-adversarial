"""
=============================================================================
FULL REANALYSIS: Multi-Block PE Attacks for First TIFS Paper
=============================================================================

Re-runs all attack experiments from "Adversarial Vulnerability of Positional
Encoding in Vision Transformers: A Targeted Attack Analysis" (T-IFS-26532-2026)
with corrected multi-block attack implementation.

Scope:
    - PE types: rope, alibi (Learned/Sinusoidal not affected, results from
      original paper can be retained for those)
    - Seeds: [42, 123, 456]
    - Attacks: FGSM-PE, PGD-PE, VTA
    - Epsilons: [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
      (full grid from original adversarial_pe_attacks.py)
    - Datasets: ImageNet-100, CIFAR-100

Total runs: 2 PE x 3 seeds x 3 attacks x 8 epsilons x 2 datasets = 288 runs
Estimated time: ~5-7 hours on T4 GPU (PGD dominates ~80% of compute)

Output:
    - imagenet/full_reanalysis_results.json
    - cifar/full_reanalysis_results.json
    - Each contains nested {pe_type: {seed: {attack: {eps: acc}}}}
    - Compatible with comparison against original adversarial_pe_results.json

Checkpoint mechanism: results are saved after EACH (pe_type, seed) combination,
so if Colab session dies, you can resume by checking which combinations are
already in the JSON and skipping them.

Usage in Colab:
    # ImageNet run:
    !python full_reanalysis.py \\
        --models_dir "/content/drive/MyDrive/Trained models_ImageNet100" \\
        --val_dir "/content/imagenet100/val" \\
        --output_path "/content/drive/MyDrive/patching_framework/full_reanalysis/imagenet_results.json" \\
        --dataset imagenet

    # CIFAR run:
    !python full_reanalysis.py \\
        --models_dir "/content/drive/MyDrive/Trained models_CIFAR100" \\
        --output_path "/content/drive/MyDrive/patching_framework/full_reanalysis/cifar_results.json" \\
        --dataset cifar
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))
from full_scale_experiment import VisionTransformer, extract_positional_embedding


# ============================================================
# CONFIG
# ============================================================
PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']  # all four PE types
SEEDS = [42, 123, 456]
ATTACKS = ['fgsm_pe', 'pgd_pe', 'vta']
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

PGD_STEPS = 20
PGD_ALPHA_RATIO = 0.1
NUM_GRAD_BATCHES = 20

# Architecture is dataset-specific (CIFAR uses smaller patches on smaller images)
MODEL_KWARGS_BASE = dict(
    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0,
)

# Dataset-specific config (architecture, classes, transforms)
DATASET_CONFIG = {
    'imagenet': {
        'num_classes': 100,
        'img_size': 224,
        'patch_size': 16,
        'transform': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    },
    'cifar': {
        'num_classes': 100,
        'img_size': 32,
        'patch_size': 4,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761]),
        ]),
    },
}


# ============================================================
# DATA LOADING
# ============================================================
def build_loader(args):
    """Build appropriate val loader for ImageNet or CIFAR."""
    cfg = DATASET_CONFIG[args.dataset]

    if args.dataset == 'imagenet':
        if not args.val_dir:
            raise ValueError("--val_dir required for imagenet")
        dataset = datasets.ImageFolder(args.val_dir, cfg['transform'])
    elif args.dataset == 'cifar':
        # Auto-download CIFAR if not present; use Colab's /content for cache
        cache_dir = args.val_dir or '/content/cifar100_data'
        os.makedirs(cache_dir, exist_ok=True)
        dataset = datasets.CIFAR100(
            root=cache_dir, train=False, download=True,
            transform=cfg['transform']
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    return loader, len(dataset)


# ============================================================
# MODEL LOADING
# ============================================================
def load_clean_model(checkpoint_path, pe_type, dataset_cfg, device):
    """Load model with architecture matching the dataset (CIFAR vs ImageNet).

    dataset_cfg must contain: num_classes, img_size, patch_size.
    """
    model_kwargs = {
        **MODEL_KWARGS_BASE,
        'img_size': dataset_cfg['img_size'],
        'patch_size': dataset_cfg['patch_size'],
        'num_classes': dataset_cfg['num_classes'],
        'pe_type': pe_type,
    }
    model = VisionTransformer(**model_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


# ============================================================
# BUFFER ACCESS — uniform interface across all 4 PE types
# ============================================================
# All helpers return dict {name: [tensor, ...]} where the list contains
# 12 elements for attention-space PE (RoPE/ALiBi) — one buffer per block —
# and 1 element for embedding-space PE (Learned/Sinusoidal) — a single
# top-level tensor. This unified shape lets the same attack code work
# for both groups without branching: aggregate_grads_across_blocks() sums
# over the list (which trivially gives the single gradient for embedding-
# space PE), and apply_shared_delta() writes to all elements in the list.
# ============================================================

def get_rope_buffers_all_blocks(model):
    return {
        'cos_cached': [b.attn.rope.cos_cached for b in model.blocks],
        'sin_cached': [b.attn.rope.sin_cached for b in model.blocks],
        'inv_freq':   [b.attn.rope.inv_freq   for b in model.blocks],
    }


def get_alibi_buffers_all_blocks(model):
    return {'slopes': [b.attn.alibi.slopes for b in model.blocks]}


def get_learned_buffers(model):
    """Learned PE: a single top-level pos_embed tensor.

    Returned as {'pos_embed': [tensor]} (list of 1) for protocol compatibility
    with the multi-block format used for RoPE/ALiBi.
    """
    return {'pos_embed': [model.pos_encoding.pos_embed]}


def get_sinusoidal_buffers(model):
    """Sinusoidal PE: a single top-level pe tensor (registered as buffer)."""
    return {'pe': [model.pos_encoding.pe]}


def get_buffers(model, pe_type):
    if pe_type == 'rope':
        return get_rope_buffers_all_blocks(model)
    elif pe_type == 'alibi':
        return get_alibi_buffers_all_blocks(model)
    elif pe_type == 'learned':
        return get_learned_buffers(model)
    elif pe_type == 'sinusoidal':
        return get_sinusoidal_buffers(model)
    raise ValueError(pe_type)


def snapshot_buffers(buffer_dict):
    return {name: [b.data.clone() for b in bufs]
            for name, bufs in buffer_dict.items()}


def restore_buffers(buffer_dict, originals):
    for name, bufs in buffer_dict.items():
        for b, orig in zip(bufs, originals[name]):
            b.data.copy_(orig)


def set_requires_grad_all(buffer_dict, requires_grad):
    for bufs in buffer_dict.values():
        for b in bufs:
            b.requires_grad_(requires_grad)


def aggregate_grads_across_blocks(buffer_dict):
    aggregated = {}
    for name, bufs in buffer_dict.items():
        grads = [b.grad for b in bufs if b.grad is not None]
        aggregated[name] = (torch.stack(grads).sum(dim=0)
                            if len(grads) > 0 else None)
    return aggregated


def apply_shared_delta(buffer_dict, deltas, originals):
    for name, bufs in buffer_dict.items():
        if deltas[name] is None:
            continue
        for b, orig in zip(bufs, originals[name]):
            b.data.copy_(orig + deltas[name])


def zero_all_grads(buffer_dict):
    for bufs in buffer_dict.values():
        for b in bufs:
            if b.grad is not None:
                b.grad.zero_()


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ============================================================
# ATTACKS — all use shared-delta multi-block pattern
# ============================================================
def fgsm_pe_multiblock(model, loader, device, pe_type, epsilon, seed=0):
    """Single-step FGSM with shared delta across all 12 blocks."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    n_batches = min(NUM_GRAD_BATCHES, len(loader))

    bufs = get_buffers(model, pe_type)
    origs = snapshot_buffers(bufs)
    set_requires_grad_all(bufs, True)

    # Compute gradient
    model.zero_grad()
    zero_all_grads(bufs)
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

    # Aggregate gradients and apply single FGSM step
    agg_grads = aggregate_grads_across_blocks(bufs)
    deltas = {}
    for name, g in agg_grads.items():
        if g is not None:
            deltas[name] = epsilon * g.sign()
        else:
            deltas[name] = None
    apply_shared_delta(bufs, deltas, origs)
    set_requires_grad_all(bufs, False)

    acc = evaluate(model, loader, device)
    restore_buffers(bufs, origs)
    return acc


def pgd_pe_multiblock(model, loader, device, pe_type, epsilon, seed=0,
                     steps=PGD_STEPS, alpha_ratio=PGD_ALPHA_RATIO):
    """Multi-step PGD with shared delta across all 12 blocks."""
    model.eval()
    alpha = epsilon * alpha_ratio
    criterion = nn.CrossEntropyLoss()
    n_batches = min(NUM_GRAD_BATCHES, len(loader))

    bufs = get_buffers(model, pe_type)
    origs = snapshot_buffers(bufs)

    torch.manual_seed(seed)
    deltas = {
        name: torch.zeros_like(origs[name][0]).uniform_(-epsilon, epsilon)
        for name in bufs
    }
    set_requires_grad_all(bufs, True)

    for step in range(steps):
        apply_shared_delta(bufs, deltas, origs)
        model.zero_grad()
        zero_all_grads(bufs)

        for i, (images, labels) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

        agg_grads = aggregate_grads_across_blocks(bufs)
        for name in deltas:
            if agg_grads[name] is not None:
                deltas[name] = (deltas[name] + alpha * agg_grads[name].sign())
                deltas[name] = torch.clamp(deltas[name], -epsilon, epsilon)

    apply_shared_delta(bufs, deltas, origs)
    set_requires_grad_all(bufs, False)
    acc = evaluate(model, loader, device)
    restore_buffers(bufs, origs)
    return acc


def vta_multiblock(model, loader, device, pe_type, epsilon, seed=0):
    """Variance-Targeted Attack with shared delta.

    Note: VTA was designed for embedding-space PE (Learned/Sinusoidal) where
    per-dimension variance has clear meaning. For RoPE/ALiBi (attention-space),
    we apply VTA on each buffer using its per-element variance.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    n_batches = min(NUM_GRAD_BATCHES, len(loader))

    bufs = get_buffers(model, pe_type)
    origs = snapshot_buffers(bufs)
    set_requires_grad_all(bufs, True)

    # Compute per-element variance for each buffer (used as VTA weights)
    variance_weights = {}
    for name, orig_list in origs.items():
        # Use block-0 instance as reference (all blocks identical anyway)
        ref = orig_list[0]
        # Per-element absolute value as proxy for "informativeness"
        # (true per-position variance computed over a single tensor is degenerate)
        weights = ref.abs() / (ref.abs().max() + 1e-10)
        variance_weights[name] = weights

    # Compute gradient
    model.zero_grad()
    zero_all_grads(bufs)
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

    # Apply VTA: weight gradient sign by variance
    agg_grads = aggregate_grads_across_blocks(bufs)
    deltas = {}
    for name, g in agg_grads.items():
        if g is not None:
            deltas[name] = epsilon * variance_weights[name] * g.sign()
        else:
            deltas[name] = None
    apply_shared_delta(bufs, deltas, origs)
    set_requires_grad_all(bufs, False)

    acc = evaluate(model, loader, device)
    restore_buffers(bufs, origs)
    return acc


ATTACK_FUNCS = {
    'fgsm_pe': fgsm_pe_multiblock,
    'pgd_pe': pgd_pe_multiblock,
    'vta': vta_multiblock,
}


# ============================================================
# CHECKPOINT / RESUME
# ============================================================
def load_existing_results(output_path):
    """Load partial results if they exist (for resume after crash)."""
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return None


def already_done(results, pe_type, seed):
    """Check if (pe_type, seed) is fully complete in existing results."""
    try:
        attacks = results['results'][pe_type][str(seed)]['attacks']
        # Need all 3 attacks, each with all 8 epsilons
        for attack_name in ATTACKS:
            if attack_name not in attacks:
                return False
            for eps in EPSILONS:
                if str(eps) not in attacks[attack_name]:
                    return False
        return True
    except (KeyError, TypeError):
        return False


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_full_reanalysis(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Models dir: {args.models_dir}")
    print(f"Output: {args.output_path}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cfg = DATASET_CONFIG[args.dataset]

    # Build val loader
    print(f"\nLoading {args.dataset} validation data...")
    val_loader, n_val = build_loader(args)
    print(f"Val images: {n_val}")

    # Resume from existing results if possible
    existing = load_existing_results(args.output_path)
    if existing is not None:
        print(f"\nFound existing results at {args.output_path} - will resume.")
        results = existing
    else:
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset": args.dataset,
                "device": str(device),
                "n_val_images": n_val,
                "config": {
                    "pe_types": PE_TYPES,
                    "seeds": SEEDS,
                    "attacks": ATTACKS,
                    "epsilons": EPSILONS,
                    "pgd_steps": PGD_STEPS,
                    "pgd_alpha_ratio": PGD_ALPHA_RATIO,
                    "n_grad_batches": NUM_GRAD_BATCHES,
                    "attack_pattern": "shared_delta_all_12_blocks",
                },
                "purpose": (
                    "Full reanalysis of T-IFS-26532-2026 with corrected "
                    "multi-block attack. Replaces RoPE and ALiBi columns "
                    "in original Tables tab:fgsm, tab:pgd, tab:vta."
                ),
            },
            "results": {},
        }

    # Run all combinations
    total_combinations = len(PE_TYPES) * len(SEEDS)
    combo_idx = 0
    overall_start = time.time()

    for pe_type in PE_TYPES:
        if pe_type not in results["results"]:
            results["results"][pe_type] = {}

        for seed in SEEDS:
            combo_idx += 1
            print(f"\n{'='*70}")
            print(f"[{combo_idx}/{total_combinations}] {pe_type} seed={seed}")
            print(f"{'='*70}")

            # Resume check
            if already_done(results, pe_type, seed):
                print(f"  Already complete in existing results - skipping.")
                continue

            ckpt = os.path.join(args.models_dir,
                                f"{pe_type}_seed{seed}",
                                "best_model.pth")
            if not os.path.exists(ckpt):
                print(f"  SKIP: checkpoint not found at {ckpt}")
                results["results"][pe_type][str(seed)] = {
                    "status": "missing_checkpoint",
                    "checkpoint": ckpt,
                }
                continue

            # Load model
            t0 = time.time()
            model = load_clean_model(ckpt, pe_type, cfg, device)
            clean_acc = evaluate(model, val_loader, device)
            print(f"  Clean accuracy: {clean_acc:.2f}%  "
                  f"(loaded in {time.time()-t0:.1f}s)")

            run_data = {
                "status": "ok",
                "checkpoint": ckpt,
                "clean_acc": float(clean_acc),
                "attacks": {a: {} for a in ATTACKS},
            }

            # Run all attacks
            for attack_name in ATTACKS:
                attack_func = ATTACK_FUNCS[attack_name]
                print(f"\n  --- {attack_name.upper()} ---")
                print(f"  {'eps':<8} {'acc':<10} {'drop':<10} {'time':<8}")

                for eps in EPSILONS:
                    t_eps = time.time()
                    try:
                        acc = attack_func(model, val_loader, device,
                                         pe_type, eps, seed=seed)
                        elapsed = time.time() - t_eps
                        drop = clean_acc - acc
                        run_data["attacks"][attack_name][str(eps)] = {
                            "accuracy": float(acc),
                            "drop_from_clean": float(drop),
                            "elapsed_sec": float(elapsed),
                        }
                        print(f"  {eps:<8} {acc:>6.2f}%  "
                              f"{drop:>+6.2f}pp  ({elapsed:.0f}s)")
                    except Exception as e:
                        print(f"  {eps:<8} ERROR: {type(e).__name__}: {e}")
                        run_data["attacks"][attack_name][str(eps)] = {
                            "accuracy": None,
                            "error": f"{type(e).__name__}: {e}",
                        }

            results["results"][pe_type][str(seed)] = run_data

            # Free GPU and save incrementally
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)
            elapsed_total = time.time() - overall_start
            print(f"\n  Saved. Total elapsed: {elapsed_total/60:.1f} min")

    # Final summary
    print(f"\n{'='*70}")
    print("FULL REANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {args.output_path}")
    print(f"Total time: {(time.time()-overall_start)/60:.1f} min")

    # Mean ± std summary (mimics format of original tab:pgd)
    print(f"\n{'Attack':<10} {'PE':<8} {'eps':<8} {'mean ± std':<20}")
    print("-" * 50)
    for attack_name in ATTACKS:
        for pe_type in PE_TYPES:
            for eps in EPSILONS:
                accs = []
                for seed in SEEDS:
                    try:
                        a = results["results"][pe_type][str(seed)]["attacks"][attack_name][str(eps)]["accuracy"]
                        if a is not None:
                            accs.append(a)
                    except (KeyError, TypeError):
                        pass
                if accs:
                    m, s = np.mean(accs), np.std(accs)
                    print(f"{attack_name:<10} {pe_type:<8} {eps:<8} "
                          f"{m:>5.1f}±{s:<5.1f}  (n={len(accs)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True,
                        help="Dir with <pe_type>_seed<N>/best_model.pth")
    parser.add_argument("--val_dir", default=None,
                        help="ImageNet val dir (required for imagenet); "
                             "for cifar, optional cache dir for download")
    parser.add_argument("--output_path", required=True,
                        help="Where to save results JSON")
    parser.add_argument("--dataset", required=True,
                        choices=['imagenet', 'cifar'])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.dataset == 'imagenet' and not args.val_dir:
        parser.error("--val_dir required for imagenet dataset")

    run_full_reanalysis(args)


if __name__ == "__main__":
    main()
