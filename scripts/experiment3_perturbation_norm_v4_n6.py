"""
=============================================================================
EXPERIMENT 3 (v4): Perturbation Norm Logging
=============================================================================

Drop-in companion to v3. Reuses v3's PGD-PE attack and model loading verbatim;
the only difference is the post-attack measurement: instead of computing
attention metrics, we log the realized perturbation norms.

Purpose
-------
Test the hypothesis that PGD-PE on a smaller grid more easily saturates the
epsilon ceiling, which would explain the dataset-dependent behavior of
MAD_miss for learned PE on CIFAR. If true, we should see ||delta_final||_inf
closer to epsilon on CIFAR than on ImageNet at the same nominal epsilon.

Metrics logged (per (PE, seed, eps), aggregated over per-block buffers)
----------------------------------------------------------------------
  delta_inf_max       : max over blocks of ||delta||_inf (the binding norm)
  delta_inf_mean      : mean over blocks of ||delta||_inf
  delta_l2_mean       : mean over blocks of ||delta||_2
  delta_inf_to_eps    : delta_inf_max / epsilon  (saturation ratio, in [0, 1])
  numel_at_ceiling    : fraction of delta entries with |delta_i| > 0.99 * eps
                        (across all blocks of the buffer)
  delta_shape         : shape of the perturbation tensor (sanity check)
  buffer_names        : which buffers were perturbed (sanity check)

For learned PE: pos_embed shape is (1, n_patches+1, embed_dim).
For sinusoidal: pe shape is (1, n_patches+1, embed_dim) (frozen, but PGD
  applies delta to the same buffer so we measure it the same way).
For RoPE: cos_cached, sin_cached, inv_freq -- multiple buffers attacked with
  shared delta. We log per-buffer norms.
For ALiBi: slopes shape -- typically (n_heads,). Important for verification.

Output JSON structure mirrors v3 for easy joining:
  results[pe][seed]["attacks"][eps] = {
      "epsilon": float,
      "elapsed_sec": float,
      "norms_per_buffer": {
          buffer_name: {
              "shape": list,
              "delta_inf": float,
              "delta_l2": float,
              "delta_inf_to_eps": float,
              "fraction_at_ceiling": float,
          }
      },
      "delta_inf_max": float,    # max over buffers
      "delta_inf_mean": float,   # mean over buffers
      "delta_l2_mean": float,    # mean over buffers
      "delta_inf_to_eps": float, # max over buffers / eps
      "fraction_at_ceiling_max": float,
  }

Usage in Colab (identical args to v3 except output filename)
------------------------------------------------------------
  !python experiment3_perturbation_norm_v4.py \\
      --models_dir "/content/drive/MyDrive/Trained models_ImageNet100" \\
      --val_dir "/content/imagenet100/val" \\
      --output_path "/content/drive/MyDrive/patching_framework/experiment3/imagenet_perturbation_norms.json" \\
      --dataset imagenet \\
      --batch_size 128

  !python experiment3_perturbation_norm_v4.py \\
      --models_dir "/content/drive/MyDrive/Trained models_CIFAR100" \\
      --output_path "/content/drive/MyDrive/patching_framework/experiment3/cifar_perturbation_norms.json" \\
      --dataset cifar \\
      --batch_size 128

Expected runtime: roughly the same as v3 -- PGD itself dominates wall-clock,
and we still run the attack identically. We just skip the attention-metric
evaluation pass (~10-20 sec/eps), so figure ~10-20% faster overall.
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

# Match v3's import path: full_scale_experiment is in the parent folder.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/content")
from full_scale_experiment import VisionTransformer  # noqa: E402


# ============================================================
# CONFIG (kept identical to v3 -- DO NOT change without reason)
# ============================================================
PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213]
SEEDS = DEFAULT_SEEDS  # backward-compatible default

EPSILONS_PER_PE = {
    'learned':    [0, 0.05, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.20],
    'sinusoidal': [0, 0.05, 0.10, 0.13, 0.15, 0.17, 0.20],
    'rope':       [0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    'alibi':      [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
}

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
        'grid_size': 14,
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
        'grid_size': 8,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761]),
        ]),
    },
}

CEILING_TOLERANCE = 0.99  # |delta| > 0.99 * eps counts as "at the ceiling"


# ============================================================
# BUFFER ACCESS (verbatim from v3)
# ============================================================
def get_buffers(model, pe_type):
    if pe_type == 'rope':
        return {
            'cos_cached': [b.attn.rope.cos_cached for b in model.blocks],
            'sin_cached': [b.attn.rope.sin_cached for b in model.blocks],
            'inv_freq':   [b.attn.rope.inv_freq   for b in model.blocks],
        }
    elif pe_type == 'alibi':
        return {'slopes': [b.attn.alibi.slopes for b in model.blocks]}
    elif pe_type == 'learned':
        return {'pos_embed': [model.pos_encoding.pos_embed]}
    elif pe_type == 'sinusoidal':
        return {'pe': [model.pos_encoding.pe]}
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
# PGD-PE ATTACK (verbatim from v3)
# ============================================================
def pgd_pe_compute_deltas(model, loader, device, pe_type, epsilon, seed=0,
                          steps=PGD_STEPS, alpha_ratio=PGD_ALPHA_RATIO):
    """Returns (deltas, originals); model is left in clean state."""
    model.eval()
    bufs = get_buffers(model, pe_type)
    origs = snapshot_buffers(bufs)

    if epsilon == 0:
        deltas = {name: torch.zeros_like(origs[name][0]) for name in bufs}
        return deltas, origs

    alpha = epsilon * alpha_ratio
    criterion = nn.CrossEntropyLoss()
    n_batches = min(NUM_GRAD_BATCHES, len(loader))

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

    set_requires_grad_all(bufs, False)
    restore_buffers(bufs, origs)
    return deltas, origs


# ============================================================
# NORM MEASUREMENT (the only new logic in v4)
# ============================================================
def measure_perturbation_norms(deltas, epsilon):
    """Compute per-buffer and aggregate norm statistics for the realized
    perturbation. epsilon may be 0 (then ratios are reported as 0.0 to avoid
    division by zero; norms will be 0 anyway since deltas are zero tensors).
    """
    per_buffer = {}
    inf_norms = []
    l2_norms = []
    fractions_at_ceiling = []

    for name, delta in deltas.items():
        d = delta.detach()
        d_inf = float(d.abs().max().item())
        d_l2 = float(d.float().norm().item())  # cast to float32 for safety

        if epsilon > 0:
            d_inf_ratio = d_inf / epsilon
            at_ceiling = (d.abs() > CEILING_TOLERANCE * epsilon).float()
            frac_ceil = float(at_ceiling.mean().item())
        else:
            d_inf_ratio = 0.0
            frac_ceil = 0.0

        per_buffer[name] = {
            "shape": list(d.shape),
            "numel": int(d.numel()),
            "delta_inf": d_inf,
            "delta_l2": d_l2,
            "delta_inf_to_eps": d_inf_ratio,
            "fraction_at_ceiling": frac_ceil,
        }
        inf_norms.append(d_inf)
        l2_norms.append(d_l2)
        fractions_at_ceiling.append(frac_ceil)

    if not inf_norms:
        return {"norms_per_buffer": per_buffer,
                "delta_inf_max": 0.0,
                "delta_inf_mean": 0.0,
                "delta_l2_mean": 0.0,
                "delta_inf_to_eps": 0.0,
                "fraction_at_ceiling_max": 0.0,
                "fraction_at_ceiling_mean": 0.0}

    delta_inf_max = max(inf_norms)
    delta_inf_to_eps = delta_inf_max / epsilon if epsilon > 0 else 0.0

    return {
        "norms_per_buffer": per_buffer,
        "delta_inf_max":     delta_inf_max,
        "delta_inf_mean":    sum(inf_norms) / len(inf_norms),
        "delta_l2_mean":     sum(l2_norms) / len(l2_norms),
        "delta_inf_to_eps":  delta_inf_to_eps,
        "fraction_at_ceiling_max":  max(fractions_at_ceiling),
        "fraction_at_ceiling_mean": sum(fractions_at_ceiling) / len(fractions_at_ceiling),
    }


# ============================================================
# DATA / MODEL LOADING (verbatim from v3)
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


def load_clean_model(checkpoint_path, pe_type, dataset_cfg, device):
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


def load_existing_results(output_path):
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return None


def already_done(results, pe_type, seed):
    try:
        run_data = results['results'][pe_type][str(seed)]
        if run_data.get('status') != 'ok':
            return False
        for eps in EPSILONS_PER_PE[pe_type]:
            if str(eps) not in run_data['attacks']:
                return False
        return True
    except (KeyError, TypeError):
        return False


# ============================================================
# MAIN
# ============================================================
def run_experiment3_v4(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_path}")
    seeds = list(args.seeds)
    print(f"Seeds: {seeds}  (n={len(seeds)})")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cfg = DATASET_CONFIG[args.dataset]
    grid_size = cfg['grid_size']

    print(f"\nLoading {args.dataset} validation data...")
    val_loader, n_val = build_loader(args)
    print(f"Val images: {n_val}, grid: {grid_size}x{grid_size}")

    existing = load_existing_results(args.output_path)
    if existing is not None:
        print(f"\nFound existing results - resuming.")
        results = existing
        results.setdefault("metadata", {}).setdefault("config", {})["seeds"] = seeds
        results["metadata"]["resumed_with_seeds"] = seeds
    else:
        results = {
            "metadata": {
                "experiment": ("Experiment 3 v4: Perturbation norm logging "
                               "(||delta_final||_inf, ||delta_final||_2, "
                               "saturation ratio) under PGD-PE attack. "
                               "Drop-in companion to v3."),
                "timestamp": datetime.now().isoformat(),
                "dataset": args.dataset,
                "device": str(device),
                "n_val_images": n_val,
                "batch_size": args.batch_size,
                "grid_size": grid_size,
                "config": {
                    "pe_types": PE_TYPES,
                    "seeds": seeds,
                    "attack": "pgd_pe",
                    "epsilons_per_pe": EPSILONS_PER_PE,
                    "pgd_steps": PGD_STEPS,
                    "pgd_alpha_ratio": PGD_ALPHA_RATIO,
                    "n_grad_batches": NUM_GRAD_BATCHES,
                    "attack_pattern": "shared_delta_all_12_blocks",
                    "validation_set": "full",
                    "ceiling_tolerance": CEILING_TOLERANCE,
                },
                "metrics_explained": {
                    "delta_inf_max":     "max over buffer keys of ||delta||_inf",
                    "delta_inf_mean":    "mean over buffer keys of ||delta||_inf",
                    "delta_l2_mean":     "mean over buffer keys of ||delta||_2 (Frobenius)",
                    "delta_inf_to_eps":  "delta_inf_max / epsilon; 1.0 = budget fully saturated",
                    "fraction_at_ceiling": (
                        "fraction of perturbation entries within "
                        "CEILING_TOLERANCE * epsilon of the L_inf ball boundary; "
                        "high values indicate PGD pushed against the budget"
                    ),
                },
                "purpose": (
                    "Test whether PGD-PE on smaller grids (CIFAR 8x8) more "
                    "easily saturates the epsilon budget than on larger "
                    "grids (ImageNet 14x14), which would explain the "
                    "dataset-dependent post-collapse behavior of learned PE."
                ),
            },
            "results": {},
        }

    overall_start = time.time()
    total_combinations = len(PE_TYPES) * len(seeds)
    combo_idx = 0

    for pe_type in PE_TYPES:
        if pe_type not in results["results"]:
            results["results"][pe_type] = {}

        for seed in seeds:
            combo_idx += 1
            print(f"\n{'='*78}", flush=True)
            print(f"[{combo_idx}/{total_combinations}] {pe_type} seed={seed}", flush=True)
            print(f"{'='*78}", flush=True)

            if already_done(results, pe_type, seed):
                print(f"  Already complete - skipping.", flush=True)
                continue

            ckpt = os.path.join(args.models_dir,
                                f"{pe_type}_seed{seed}",
                                "best_model.pth")
            if not os.path.exists(ckpt):
                print(f"  SKIP: checkpoint not found at {ckpt}", flush=True)
                results["results"][pe_type][str(seed)] = {
                    "status": "missing_checkpoint", "checkpoint": ckpt,
                }
                continue

            t0 = time.time()
            model = load_clean_model(ckpt, pe_type, cfg, device)
            print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

            run_data = {"status": "ok", "checkpoint": ckpt, "attacks": {}}

            print(f"\n  --- Per-eps perturbation norm logging ---", flush=True)
            print(f"  {'eps':<6} {'inf_max':<9} {'inf/eps':<9} "
                  f"{'l2_mean':<10} {'frac@ceil':<10} {'time':<7}", flush=True)

            for eps in EPSILONS_PER_PE[pe_type]:
                t_eps = time.time()
                try:
                    deltas, origs = pgd_pe_compute_deltas(
                        model, val_loader, device, pe_type, eps, seed=seed
                    )
                    norm_record = measure_perturbation_norms(deltas, eps)
                    elapsed = time.time() - t_eps

                    attack_record = {
                        "epsilon": float(eps),
                        "elapsed_sec": float(elapsed),
                        **{k: (v if not isinstance(v, dict) else v)
                           for k, v in norm_record.items()},
                    }
                    run_data["attacks"][str(eps)] = attack_record

                    print(f"  {eps:<6} "
                          f"{norm_record['delta_inf_max']:<9.5f} "
                          f"{norm_record['delta_inf_to_eps']:<9.4f} "
                          f"{norm_record['delta_l2_mean']:<10.4f} "
                          f"{norm_record['fraction_at_ceiling_max']:<10.4f} "
                          f"({elapsed:.0f}s)", flush=True)

                    # IMPORTANT: free deltas/origs between epsilons
                    del deltas, origs

                except Exception as e:
                    print(f"  {eps:<6} ERROR: {type(e).__name__}: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    run_data["attacks"][str(eps)] = {
                        "error": f"{type(e).__name__}: {e}",
                    }

            results["results"][pe_type][str(seed)] = run_data

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)
            elapsed_total = time.time() - overall_start
            print(f"  Saved. Total elapsed: {elapsed_total/60:.1f} min", flush=True)

    print(f"\n{'='*78}", flush=True)
    print("EXPERIMENT 3 v4 COMPLETE", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"Output: {args.output_path}", flush=True)
    print(f"Total time: {(time.time()-overall_start)/60:.1f} min", flush=True)

    # Compact summary table
    import statistics
    print(f"\nSaturation summary (mean +/- std over selected seeds):", flush=True)
    print(f"{'PE':<11} {'eps':<6} {'delta_inf_max':<16} "
          f"{'inf/eps':<14} {'frac@ceil':<14}", flush=True)
    print("-" * 75, flush=True)
    for pe_type in PE_TYPES:
        for eps in EPSILONS_PER_PE[pe_type]:
            inf_maxs, ratios, fracs = [], [], []
            for seed in seeds:
                try:
                    d = results["results"][pe_type][str(seed)]["attacks"][str(eps)]
                    if "delta_inf_max" in d:
                        inf_maxs.append(d["delta_inf_max"])
                        ratios.append(d["delta_inf_to_eps"])
                        fracs.append(d["fraction_at_ceiling_max"])
                except (KeyError, TypeError):
                    pass
            if inf_maxs:
                fmt = lambda xs, p=4: (f"{statistics.mean(xs):>6.{p}f}+-"
                                       f"{(statistics.stdev(xs) if len(xs)>1 else 0):<6.{p}f}")
                print(f"{pe_type:<11} {eps:<6} {fmt(inf_maxs,5):<16} "
                      f"{fmt(ratios,4):<14} {fmt(fracs,4):<14}", flush=True)
        print(flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset", required=True, choices=['imagenet', 'cifar'])
    parser.add_argument("--batch_size", type=int, default=128)
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

    run_experiment3_v4(args)


if __name__ == "__main__":
    main()
