"""
=============================================================================
EXPERIMENT 2: ALiBi Structural Ablation — Slopes vs Slopes+RelDist Attack
=============================================================================

Tests whether ALiBi's vulnerability is concentrated in its slope parameters
alone (12 scalars) or extends to the relative distance matrix (rel_dist)
that defines the attention bias geometry.

Background: In our PGD-PE attacks, ALiBi is perturbed only via slopes
(consistent with adversarial_pe_attacks.py). The full ALiBi PE state is:
    bias = -slopes * rel_dist
where slopes is (1, 12, 1, 1) and rel_dist is (1, 1, N, N).

If we perturb ONLY slopes, we are limited to 12 scalar perturbations
(one per attention head). If we also perturb rel_dist, we have N*N
additional dimensions of perturbation.

Three regimes are compared:
    (A) slopes-only (this is what full_reanalysis.py already does)
    (B) rel_dist-only (perturbs the geometric prior, leaves slopes intact)
    (C) slopes + rel_dist (full ALiBi state attack)

The comparison answers:
    - Is ALiBi's apparent fragility a slopes-only phenomenon, or do
      additional dimensions amplify the vulnerability?
    - Does the rel_dist matrix carry exploitable adversarial geometry
      independent of slopes?

Configuration:
    - PE type: alibi only
    - Seeds: [42, 123, 456, 789, 1011, 1213]
    - Attack: PGD-PE only (T=20, alpha=eps/10)
    - Epsilons: standard 8-point grid (matches main paper tables)
    - Datasets: ImageNet-100 and CIFAR-100

Total runs: 3 regimes x 6 seeds x 8 eps x 2 datasets = 288 runs
Estimated time: ~30 min per dataset on Blackwell, ~1 hour total.

Output: experiment2_alibi_ablation_results.json
        Structure: {
          "results": {
            "alibi": {
              "42": {
                "clean_acc": 81.16,
                "regime_slopes_only": {"pgd_pe": {"0.05": {...}, ...}},
                "regime_reldist_only": {"pgd_pe": {...}},
                "regime_both": {"pgd_pe": {...}},
              }, ...
            }
          }
        }

Usage in Colab:
    !python experiment2_alibi_ablation.py \\
        --models_dir "/content/drive/MyDrive/Trained models_ImageNet100" \\
        --val_dir "/content/imagenet100/val" \\
        --output_path "/content/drive/MyDrive/patching_framework/experiment2/imagenet_alibi_ablation.json" \\
        --dataset imagenet
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))
from full_scale_experiment import VisionTransformer


# ============================================================
# CONFIG (self-contained, matches full_reanalysis.py)
# ============================================================
PGD_STEPS = 20
PGD_ALPHA_RATIO = 0.1
NUM_GRAD_BATCHES = 20

MODEL_KWARGS_BASE = dict(
    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0,
)

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
# CONFIG
# ============================================================
DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213]
SEEDS = DEFAULT_SEEDS  # backward-compatible default
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
PGD_STEPS = 20
PGD_ALPHA_RATIO = 0.1
NUM_GRAD_BATCHES = 20

# Three attack regimes
REGIMES = ['slopes_only', 'reldist_only', 'both']


# ============================================================
# ALIBI BUFFER ACCESS — three regime variants
# ============================================================
def get_alibi_buffers_by_regime(model, regime):
    """Returns dict of buffer lists according to attack regime.

    All return values use the {name: [tensor_per_block]} format used by
    full_reanalysis.py, so the same shared-delta multi-block attack
    machinery applies.

    slopes_only:    only the per-head slope parameters
    reldist_only:   only the precomputed relative-distance matrix
    both:           both at once

    Note that rel_dist is NOT instantiated identically per block in this
    implementation -- each block has its own copy. The shared-delta
    pattern still applies.
    """
    if regime == 'slopes_only':
        return {'slopes': [b.attn.alibi.slopes for b in model.blocks]}
    elif regime == 'reldist_only':
        return {'rel_dist': [b.attn.alibi.rel_dist for b in model.blocks]}
    elif regime == 'both':
        return {
            'slopes':   [b.attn.alibi.slopes   for b in model.blocks],
            'rel_dist': [b.attn.alibi.rel_dist for b in model.blocks],
        }
    raise ValueError(f"Unknown regime: {regime}")


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
# PGD-PE FOR ALIBI ABLATION
# ============================================================
def pgd_pe_alibi_regime(model, loader, device, regime, epsilon, seed=0,
                       steps=PGD_STEPS, alpha_ratio=PGD_ALPHA_RATIO):
    """Multi-block PGD-PE with regime-specific buffer selection.

    Identical structure to full_reanalysis.py's pgd_pe_multiblock, but
    uses regime-specific buffer dict from get_alibi_buffers_by_regime.
    """
    model.eval()
    alpha = epsilon * alpha_ratio
    criterion = nn.CrossEntropyLoss()
    n_batches = min(NUM_GRAD_BATCHES, len(loader))

    bufs = get_alibi_buffers_by_regime(model, regime)
    origs = snapshot_buffers(bufs)

    # Initialize shared deltas (one per buffer name)
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

    # Evaluate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, pred = model(images).max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    acc = 100.0 * correct / total

    restore_buffers(bufs, origs)
    return acc


# ============================================================
# DATA / MODEL LOADING
# ============================================================
def build_loader(args):
    cfg = DATASET_CONFIG[args.dataset]
    if args.dataset == 'imagenet':
        if not args.val_dir:
            raise ValueError("--val_dir required for imagenet")
        dataset = datasets.ImageFolder(args.val_dir, cfg['transform'])
    elif args.dataset == 'cifar':
        cache_dir = args.val_dir or '/content/cifar100_data'
        os.makedirs(cache_dir, exist_ok=True)
        dataset = datasets.CIFAR100(
            root=cache_dir, train=False, download=True,
            transform=cfg['transform']
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    return loader, len(dataset)


def load_clean_model(checkpoint_path, dataset_cfg, device):
    model_kwargs = {
        **MODEL_KWARGS_BASE,
        'img_size': dataset_cfg['img_size'],
        'patch_size': dataset_cfg['patch_size'],
        'num_classes': dataset_cfg['num_classes'],
        'pe_type': 'alibi',
    }
    model = VisionTransformer(**model_kwargs).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


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
# RESUME / CHECKPOINT
# ============================================================
def load_existing_results(output_path):
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return None


def already_done(results, seed, regime):
    try:
        regime_data = results['results']['alibi'][str(seed)][f'regime_{regime}']
        if 'pgd_pe' not in regime_data:
            return False
        for eps in EPSILONS:
            if str(eps) not in regime_data['pgd_pe']:
                return False
        return True
    except (KeyError, TypeError):
        return False


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_experiment2(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_path}")
    seeds = list(args.seeds)
    print(f"Seeds: {seeds}  (n={len(seeds)})")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cfg = DATASET_CONFIG[args.dataset]

    print(f"\nLoading {args.dataset} validation data...")
    val_loader, n_val = build_loader(args)
    print(f"Val images: {n_val}")

    existing = load_existing_results(args.output_path)
    if existing is not None:
        print(f"\nFound existing results - resuming.")
        results = existing
        results.setdefault("metadata", {}).setdefault("config", {})["seeds"] = seeds
        results["metadata"]["resumed_with_seeds"] = seeds
    else:
        results = {
            "metadata": {
                "experiment": "Experiment 2: ALiBi structural ablation",
                "timestamp": datetime.now().isoformat(),
                "dataset": args.dataset,
                "device": str(device),
                "n_val_images": n_val,
                "regimes": REGIMES,
                "regime_descriptions": {
                    "slopes_only": "perturb only slopes (12 scalars)",
                    "reldist_only": "perturb only rel_dist matrix (NxN)",
                    "both": "perturb both slopes and rel_dist",
                },
                "config": {
                    "seeds": seeds,
                    "attack": "pgd_pe",
                    "epsilons": EPSILONS,
                    "pgd_steps": PGD_STEPS,
                    "pgd_alpha_ratio": PGD_ALPHA_RATIO,
                    "n_grad_batches": NUM_GRAD_BATCHES,
                    "attack_pattern": "shared_delta_all_12_blocks",
                },
                "purpose": (
                    "Determines whether ALiBi vulnerability is concentrated "
                    "in slopes (12 scalars) or extends to the rel_dist matrix. "
                    "Provides mechanistic data for explaining why ALiBi has "
                    "the lowest attack threshold among all PE types."
                ),
            },
            "results": {"alibi": {}},
        }

    overall_start = time.time()
    total_combinations = len(seeds) * len(REGIMES)
    combo_idx = 0

    for seed in seeds:
        if str(seed) not in results["results"]["alibi"]:
            results["results"]["alibi"][str(seed)] = {}

        ckpt = os.path.join(args.models_dir, f"alibi_seed{seed}", "best_model.pth")
        if not os.path.exists(ckpt):
            print(f"  SKIP seed={seed}: checkpoint not found at {ckpt}")
            continue

        # Load model once per seed; reused across regimes
        t0 = time.time()
        model = load_clean_model(ckpt, cfg, device)
        clean_acc = evaluate(model, val_loader, device)
        results["results"]["alibi"][str(seed)]["clean_acc"] = float(clean_acc)
        results["results"]["alibi"][str(seed)]["checkpoint"] = ckpt
        results["results"]["alibi"][str(seed)]["status"] = "ok"
        print(f"\n{'='*70}")
        print(f"alibi seed={seed}: clean acc = {clean_acc:.2f}%  "
              f"(loaded in {time.time()-t0:.1f}s)")
        print(f"{'='*70}")

        for regime in REGIMES:
            combo_idx += 1
            print(f"\n[{combo_idx}/{total_combinations}] regime={regime}")

            if already_done(results, seed, regime):
                print(f"  Already complete - skipping.")
                continue

            regime_data = {"pgd_pe": {}}
            print(f"  {'eps':<8} {'acc':<10} {'drop':<10} {'time':<8}")

            for eps in EPSILONS:
                t_eps = time.time()
                try:
                    acc = pgd_pe_alibi_regime(
                        model, val_loader, device, regime, eps, seed=seed
                    )
                    elapsed = time.time() - t_eps
                    drop = clean_acc - acc
                    regime_data["pgd_pe"][str(eps)] = {
                        "accuracy": float(acc),
                        "drop_from_clean": float(drop),
                        "elapsed_sec": float(elapsed),
                    }
                    print(f"  {eps:<8} {acc:>6.2f}%  "
                          f"{drop:>+6.2f}pp  ({elapsed:.0f}s)")
                except Exception as e:
                    print(f"  {eps:<8} ERROR: {type(e).__name__}: {e}")
                    regime_data["pgd_pe"][str(eps)] = {
                        "accuracy": None,
                        "error": f"{type(e).__name__}: {e}",
                    }

            results["results"]["alibi"][str(seed)][f"regime_{regime}"] = regime_data

            # Save incrementally
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)
            elapsed_total = time.time() - overall_start
            print(f"  Saved. Total elapsed: {elapsed_total/60:.1f} min")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT 2 COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {args.output_path}")
    print(f"Total time: {(time.time()-overall_start)/60:.1f} min")

    # Print comparison table
    import statistics
    print(f"\nPGD-PE accuracy under three attack regimes (mean ± std over selected seeds):")
    print(f"{'eps':<8} {'slopes_only':<16} {'reldist_only':<16} {'both':<16}")
    print("-" * 65)
    for eps in EPSILONS:
        row = [eps]
        for regime in REGIMES:
            accs = []
            for seed in seeds:
                try:
                    a = results["results"]["alibi"][str(seed)][f"regime_{regime}"]["pgd_pe"][str(eps)]["accuracy"]
                    if a is not None:
                        accs.append(a)
                except (KeyError, TypeError):
                    pass
            if accs:
                m = statistics.mean(accs)
                s = statistics.stdev(accs) if len(accs) > 1 else 0
                row.append(f"{m:>5.1f}±{s:<5.1f}")
            else:
                row.append("---")
        print(f"{row[0]:<8} {row[1]:<16} {row[2]:<16} {row[3]:<16}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset", required=True, choices=['imagenet', 'cifar'])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to process (default: 42 123 456 789 1011 1213)",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.dataset == 'imagenet' and not args.val_dir:
        parser.error("--val_dir required for imagenet")

    run_experiment2(args)


if __name__ == "__main__":
    main()
