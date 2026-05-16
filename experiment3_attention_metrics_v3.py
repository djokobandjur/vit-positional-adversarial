"""
=============================================================================
EXPERIMENT 3 (v3): Paired Attention Analysis with SPATIAL Metrics
=============================================================================

Adds spatial metrics to v2's paired attention comparison:

  - mean_jump_distance_all (MAD_all): mean Euclidean distance in 2D patch
    grid between argmax_clean and argmax_attacked, averaged over ALL
    (sample, head, query) triples (including zero-distance cases where
    argmax is preserved). Patch-to-patch only.

  - mean_jump_distance_miss (MAD_miss): same as above but conditioned on
    argmax_clean != argmax_attacked. Answers "when the attack misdirects
    attention, how far does it jump?" PRIMARY SPATIAL METRIC.

  - mass_within_R (L-Err@R): for R in {1, 2, 3}, the cumulative attention
    mass in attacked attention that lies within Chebyshev distance R of
    the clean argmax position. Reports both clean (sanity check) and
    attacked. Per-block, averaged over (sample, head, query).

Patch grid is 14x14 = 196 patches for ImageNet (img_size=224, patch=16),
plus CLS token at index 0 (197 total). Spatial metrics exclude CLS:
  - CLS is excluded as query (we don't compute MAD for CLS query)
  - CLS is excluded as target (if argmax = CLS, that query is excluded)

Configuration changes from v2:
  - Learned epsilons: [0, 0.05, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.20]
    (denser sampling in 30-70% accuracy zone)
  - Sinusoidal epsilons: [0, 0.05, 0.10, 0.13, 0.15, 0.17, 0.20]
  - RoPE, ALiBi unchanged

Output adds (per (PE, seed, eps)):
    per_block_mad_all          [12 floats]
    per_block_mad_miss         [12 floats]
    per_block_miss_rate        [12 floats]  # = 1 - top1_overlap, explicit
    per_block_mass_R1_clean    [12 floats]
    per_block_mass_R1_attacked [12 floats]
    per_block_mass_R2_clean    [12 floats]
    per_block_mass_R2_attacked [12 floats]
    per_block_mass_R3_clean    [12 floats]
    per_block_mass_R3_attacked [12 floats]
    mean_mad_all, mean_mad_miss, mean_miss_rate (scalars)
    mean_mass_R1_clean, mean_mass_R1_attacked, ...

Memory: same dual-pass structure as v2, ~1.5 GB peak attention cache at
batch_size=128. Spatial metric computation adds ~5-10s per epsilon.

Usage in Colab:
    !python experiment3_attention_metrics_v3.py \\
        --models_dir "/content/drive/MyDrive/Trained models_ImageNet100" \\
        --val_dir "/content/imagenet100/val" \\
        --output_path "/content/drive/MyDrive/patching_framework/experiment3/imagenet_spatial_metrics.json" \\
        --dataset imagenet \\
        --batch_size 128
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# FIX: full_scale_experiment.py is in the PARENT folder of experiment3/
sys.path.insert(0, str(Path(__file__).parent.parent))
from full_scale_experiment import VisionTransformer, TransformerBlock


# ============================================================
# CONFIG
# ============================================================
PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS = [42, 123, 456]

# v3: denser grid for learned (most jagged break-zone), modest expansion
# for sinusoidal. RoPE and ALiBi already cover their break zones well.
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
        'grid_size': 14,  # 14x14 = 196 patches + 1 CLS = 197 tokens
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
        'grid_size': 8,  # 8x8 = 64 patches + 1 CLS = 65 tokens
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761]),
        ]),
    },
}

# Radii for L-Err@R metric (Chebyshev distance in patch grid)
LOCALIZATION_RADII = [1, 2, 3]


# ============================================================
# BUFFER ACCESS (unchanged from v2)
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
# PGD-PE ATTACK (unchanged from v2)
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
# SPATIAL UTILITIES
# ============================================================
def build_patch_coords(grid_size, device):
    """Returns 2D coordinates for each token index.
    Index 0 (CLS) -> (-1, -1) sentinel.
    Index i (patch) -> (row, col) in [0, grid_size-1]^2.

    Returns:
        coords: (N, 2) float tensor where N = grid_size^2 + 1
    """
    n_patches = grid_size * grid_size
    n_tokens = n_patches + 1
    coords = torch.zeros(n_tokens, 2, device=device, dtype=torch.float32)
    # CLS sentinel
    coords[0, 0] = -1.0
    coords[0, 1] = -1.0
    # Patches: index i in [1, n_patches] -> patch_idx = i-1
    patch_idx = torch.arange(n_patches, device=device)
    rows = (patch_idx // grid_size).float()
    cols = (patch_idx % grid_size).float()
    coords[1:, 0] = rows
    coords[1:, 1] = cols
    return coords


def build_chebyshev_distance_matrix(grid_size, device):
    """Returns (N, N) matrix where entry [i, j] = Chebyshev distance in
    patch grid between tokens i and j.

    For CLS-involved pairs, distance is set to a large sentinel value
    (effectively infinity) so they're never counted as 'within R'.
    """
    n_patches = grid_size * grid_size
    n_tokens = n_patches + 1

    coords = build_patch_coords(grid_size, device)  # (N, 2)
    # Compute Chebyshev distance between all pairs
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 2)
    cheb = diff.abs().max(dim=-1).values  # (N, N)

    # Mask CLS-involved pairs with a large value (so never within R)
    cls_mask = torch.zeros(n_tokens, n_tokens, dtype=torch.bool, device=device)
    cls_mask[0, :] = True
    cls_mask[:, 0] = True
    cheb = torch.where(cls_mask, torch.full_like(cheb, 1e6), cheb)
    return cheb


# ============================================================
# PAIRED ATTENTION CAPTURE (extended with spatial metrics)
# ============================================================
class PairedAttentionCapture:
    """Same dual-mode interface as v2; computes additional spatial metrics
    in attacked_pass when comparing against cached clean attention.
    """

    def __init__(self, model, grid_size, device):
        self.model = model
        self.depth = len(model.blocks)
        self.grid_size = grid_size
        self.device = device

        # Precomputed spatial helpers
        self.coords = build_patch_coords(grid_size, device)  # (N, 2)
        self.cheb = build_chebyshev_distance_matrix(grid_size, device)  # (N, N)

        # Per-block accumulators
        D = self.depth
        self.entropy_sum  = [0.0] * D
        self.top1_sum     = [0.0] * D
        self.top5_sum     = [0.0] * D
        self.kl_sum       = [0.0] * D
        self.js_sum       = [0.0] * D
        # Spatial: jump distance
        self.mad_all_sum  = [0.0] * D  # sum over (B, H, N_query) of mean dist
        self.mad_miss_sum = [0.0] * D  # sum of (sum of dist on misses)
        self.miss_count   = [0.0] * D  # number of misses (for MAD_miss denom)
        self.valid_count  = [0.0] * D  # number of valid query positions (for MAD_all denom)
        # Spatial: localization mass
        self.mass_clean_sum    = [[0.0]*D for _ in LOCALIZATION_RADII]
        self.mass_attacked_sum = [[0.0]*D for _ in LOCALIZATION_RADII]

        # Counter (only incremented at block 0)
        self.count_sh = 0  # (B, H) pairs

        # Clean cache for paired comparison
        self._clean_cache = [None] * self.depth
        self._mode = None

        # Accuracy
        self.correct_attacked = 0
        self.correct_clean = 0
        self.total = 0

    # ---- forward wrapper installation ----
    def _install_hooks(self):
        self._original_forwards = []
        for block_idx, block in enumerate(self.model.blocks):
            self._original_forwards.append(block.forward)
            block.forward = self._make_wrapper(block, block_idx)

    def _uninstall_hooks(self):
        for block, orig_fwd in zip(self.model.blocks, self._original_forwards):
            block.forward = orig_fwd
        self._original_forwards = []

    def _make_wrapper(self, block, block_idx):
        original_attn_forward = block.attn.forward

        def wrapped(x):
            attn_out, attn_weights = original_attn_forward(
                block.norm1(x), return_attention=True
            )
            x = x + attn_out
            x = x + block.mlp(block.norm2(x))
            self._dispatch(block_idx, attn_weights)
            return x
        return wrapped

    # ---- mode switching ----
    class _PassContext:
        def __init__(self, parent, mode):
            self.parent = parent
            self.mode = mode

        def __enter__(self):
            self.parent._mode = self.mode
            self.parent._install_hooks()
            return self.parent

        def __exit__(self, *args):
            self.parent._uninstall_hooks()
            self.parent._mode = None

    def clean_pass(self):
        return self._PassContext(self, 'clean')

    def attacked_pass(self):
        return self._PassContext(self, 'attacked')

    def _dispatch(self, block_idx, attn_weights):
        if self._mode == 'clean':
            self._clean_cache[block_idx] = attn_weights.detach()
        elif self._mode == 'attacked':
            self._compute_paired_metrics(block_idx, attn_weights.detach())
            self._clean_cache[block_idx] = None

    # ---- core metric computation ----
    def _compute_paired_metrics(self, block_idx, A_att):
        """A_att: (B, H, N, N). Compares against self._clean_cache[block_idx]."""
        A_cln = self._clean_cache[block_idx]
        if A_cln is None:
            return

        with torch.no_grad():
            B, H, N, _ = A_att.shape
            eps_safe = 1e-12

            # ===== Distributional metrics (entropy, KL, JS) =====
            log_att = torch.log(A_att.clamp(min=eps_safe))
            ent_per_row = -(A_att * log_att).sum(dim=-1)  # (B, H, N)
            self.entropy_sum[block_idx] += ent_per_row.mean(dim=-1).sum().item()

            log_cln = torch.log(A_cln.clamp(min=eps_safe))
            kl_per_row = (A_cln * (log_cln - log_att)).sum(dim=-1)
            self.kl_sum[block_idx] += kl_per_row.mean(dim=-1).sum().item()

            M = 0.5 * (A_cln + A_att)
            log_M = torch.log(M.clamp(min=eps_safe))
            js_per_row = 0.5 * ((A_cln * (log_cln - log_M)).sum(dim=-1)
                                + (A_att * (log_att - log_M)).sum(dim=-1))
            self.js_sum[block_idx] += js_per_row.mean(dim=-1).sum().item()

            # ===== Top-k overlap =====
            argmax_cln = A_cln.argmax(dim=-1)  # (B, H, N)
            argmax_att = A_att.argmax(dim=-1)  # (B, H, N)
            match1 = (argmax_cln == argmax_att).float()  # (B, H, N)
            self.top1_sum[block_idx] += match1.mean(dim=-1).sum().item()

            k = min(5, N)
            top5_cln = A_cln.topk(k, dim=-1).indices
            top5_att = A_att.topk(k, dim=-1).indices
            mask_cln = torch.zeros_like(A_cln, dtype=torch.bool)
            mask_att = torch.zeros_like(A_att, dtype=torch.bool)
            mask_cln.scatter_(-1, top5_cln, True)
            mask_att.scatter_(-1, top5_att, True)
            inter5 = (mask_cln & mask_att).sum(dim=-1).float() / k  # (B, H, N)
            self.top5_sum[block_idx] += inter5.mean(dim=-1).sum().item()

            # ===== Spatial metrics (patch-to-patch only) =====
            # Build query mask: query is valid if it's a patch (not CLS)
            # AND its clean argmax is a patch (not CLS)
            # query_idx_valid: queries 1..N-1 (skip CLS at 0)
            # target_clean_valid: argmax_cln != 0 (clean argmax is patch)
            # target_att_valid: argmax_att != 0 (attacked argmax is patch)
            # We only count queries where ALL THREE hold.
            query_mask = torch.zeros(N, dtype=torch.bool, device=A_cln.device)
            query_mask[1:] = True  # patches only as queries
            query_mask = query_mask.view(1, 1, N).expand(B, H, N)

            valid_targets = (argmax_cln != 0) & (argmax_att != 0)  # (B, H, N)
            valid = query_mask & valid_targets  # (B, H, N)

            # ----- Mean Jump Distance -----
            # Get 2D coords for clean and attacked argmax
            coords_cln = self.coords[argmax_cln]  # (B, H, N, 2)
            coords_att = self.coords[argmax_att]  # (B, H, N, 2)
            jump_dist = (coords_cln - coords_att).pow(2).sum(dim=-1).sqrt()  # (B, H, N)

            # Mask invalid positions
            jump_dist_valid = jump_dist * valid.float()
            valid_per_sh = valid.float().sum(dim=-1)  # (B, H), denom for MAD_all
            mad_all_per_sh_sum = jump_dist_valid.sum(dim=-1)  # (B, H)
            # Avoid div by zero
            mad_all_per_sh = mad_all_per_sh_sum / valid_per_sh.clamp(min=1)  # (B, H)
            self.mad_all_sum[block_idx] += mad_all_per_sh.sum().item()
            self.valid_count[block_idx] += (valid_per_sh > 0).sum().item()

            # MAD_miss: only positions where argmax differs (and valid)
            miss_mask = valid & (argmax_cln != argmax_att)  # (B, H, N)
            n_misses_per_sh = miss_mask.float().sum(dim=-1)  # (B, H)
            jump_dist_miss = jump_dist * miss_mask.float()
            mad_miss_per_sh_sum = jump_dist_miss.sum(dim=-1)  # (B, H)
            # Per-(B, H) average over misses (sum / num_misses), then sum over (B, H)
            # We accumulate sums and counts so final mean is exact.
            self.mad_miss_sum[block_idx] += mad_miss_per_sh_sum.sum().item()
            self.miss_count[block_idx]   += n_misses_per_sh.sum().item()

            # ----- Localization Error @ R -----
            # For each (sample, head, query), the clean argmax position determines
            # the target. We sum the attention mass (clean and attacked) over all
            # keys within Chebyshev distance R of that target.
            # Strategy: build a (B, H, N, N) mask "within_R_of_clean_argmax", then
            # multiply with attention and sum over last axis.
            for r_idx, R in enumerate(LOCALIZATION_RADII):
                # cheb[argmax_cln, :] gives Chebyshev distances from target to all keys
                # Shape: (B, H, N) -> we need (B, H, N, N) where last dim is keys
                # cheb is (N, N); index by argmax_cln to get (B, H, N, N)
                target_dists = self.cheb[argmax_cln]  # (B, H, N, N)
                within_R = (target_dists <= R).float()  # (B, H, N, N)

                # Mass within R for clean and attacked attention
                mass_cln = (A_cln * within_R).sum(dim=-1)  # (B, H, N)
                mass_att = (A_att * within_R).sum(dim=-1)  # (B, H, N)

                # Only count valid query positions
                mass_cln_valid = mass_cln * valid.float()
                mass_att_valid = mass_att * valid.float()
                mass_cln_per_sh = mass_cln_valid.sum(dim=-1) / valid_per_sh.clamp(min=1)
                mass_att_per_sh = mass_att_valid.sum(dim=-1) / valid_per_sh.clamp(min=1)

                self.mass_clean_sum[r_idx][block_idx]    += mass_cln_per_sh.sum().item()
                self.mass_attacked_sum[r_idx][block_idx] += mass_att_per_sh.sum().item()

            # Count tracker (only block 0)
            if block_idx == 0:
                self.count_sh += B * H

    # ---- accuracy bookkeeping ----
    def add_clean_correct(self, n):    self.correct_clean += n
    def add_attacked_correct(self, n): self.correct_attacked += n
    def add_total(self, n):            self.total += n

    # ---- output ----
    def get_metrics(self):
        c = max(self.count_sh, 1)
        D = self.depth

        out = {
            'per_block_entropy':      [self.entropy_sum[b] / c for b in range(D)],
            'per_block_top1_overlap': [self.top1_sum[b] / c for b in range(D)],
            'per_block_top5_overlap': [self.top5_sum[b] / c for b in range(D)],
            'per_block_kl':           [self.kl_sum[b] / c for b in range(D)],
            'per_block_js':           [self.js_sum[b] / c for b in range(D)],
            # Spatial
            'per_block_mad_all':      [self.mad_all_sum[b] / max(self.valid_count[b], 1)
                                       for b in range(D)],
            'per_block_mad_miss':     [self.mad_miss_sum[b] / max(self.miss_count[b], 1)
                                       for b in range(D)],
            'per_block_miss_rate':    [1.0 - self.top1_sum[b] / c for b in range(D)],
            # Accuracies
            'clean_accuracy':    100.0 * self.correct_clean / max(self.total, 1),
            'attacked_accuracy': 100.0 * self.correct_attacked / max(self.total, 1),
        }
        # Localization mass curves
        for r_idx, R in enumerate(LOCALIZATION_RADII):
            out[f'per_block_mass_R{R}_clean']    = [self.mass_clean_sum[r_idx][b] / c
                                                   for b in range(D)]
            out[f'per_block_mass_R{R}_attacked'] = [self.mass_attacked_sum[r_idx][b] / c
                                                   for b in range(D)]
        return out


# ============================================================
# PAIRED EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_paired(model, loader, device, pe_type, deltas, originals, grid_size):
    """Dual forward pass per batch, accumulating paired metrics."""
    model.eval()
    capture = PairedAttentionCapture(model, grid_size, device)
    bufs = get_buffers(model, pe_type)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        # Pass 1: clean (buffers already in clean state)
        with capture.clean_pass():
            outputs_clean = model(images)
        _, pred_clean = outputs_clean.max(1)
        capture.add_clean_correct(pred_clean.eq(labels).sum().item())

        # Apply attacked PE
        apply_shared_delta(bufs, deltas, originals)

        # Pass 2: attacked
        with capture.attacked_pass():
            outputs_att = model(images)
        _, pred_att = outputs_att.max(1)
        capture.add_attacked_correct(pred_att.eq(labels).sum().item())
        capture.add_total(B)

        # Restore clean for next batch
        restore_buffers(bufs, originals)

    return capture.get_metrics()


# ============================================================
# DATA / MODEL LOADING (unchanged)
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
def run_experiment3_v3(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_path}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cfg = DATASET_CONFIG[args.dataset]
    grid_size = cfg['grid_size']

    print(f"\nLoading {args.dataset} validation data...")
    val_loader, n_val = build_loader(args)
    print(f"Val images: {n_val}, grid: {grid_size}x{grid_size}")

    existing = load_existing_results(args.output_path)
    if existing is not None:
        print(f"\nFound existing results - resuming.")
        results = existing
    else:
        results = {
            "metadata": {
                "experiment": ("Experiment 3 v3: Paired attention + spatial "
                               "metrics (Mean Jump Distance, L-Err@R) under "
                               "PE attack"),
                "timestamp": datetime.now().isoformat(),
                "dataset": args.dataset,
                "device": str(device),
                "n_val_images": n_val,
                "batch_size": args.batch_size,
                "grid_size": grid_size,
                "config": {
                    "pe_types": PE_TYPES,
                    "seeds": SEEDS,
                    "attack": "pgd_pe",
                    "epsilons_per_pe": EPSILONS_PER_PE,
                    "pgd_steps": PGD_STEPS,
                    "pgd_alpha_ratio": PGD_ALPHA_RATIO,
                    "n_grad_batches": NUM_GRAD_BATCHES,
                    "attack_pattern": "shared_delta_all_12_blocks",
                    "validation_set": "full",
                    "localization_radii": LOCALIZATION_RADII,
                },
                "metrics_explained": {
                    "per_block_mad_all": (
                        "Mean Jump Distance over all (sample, head, query): "
                        "Euclidean distance in 2D patch grid between clean "
                        "and attacked argmax positions. Includes zero-distance "
                        "cases (preserved argmax). Patch-to-patch only "
                        "(CLS excluded as query and as target). For 14x14 "
                        "grid, max distance is sqrt(2*13^2) = 18.4."
                    ),
                    "per_block_mad_miss": (
                        "Mean Jump Distance conditioned on argmax mismatch: "
                        "average distance computed only over (sample, head, "
                        "query) where clean argmax != attacked argmax. "
                        "PRIMARY SPATIAL METRIC. Answers 'when attention "
                        "is misdirected, how far does the focus jump?'"
                    ),
                    "per_block_miss_rate": (
                        "Fraction of (sample, head, query) where clean and "
                        "attacked argmax differ. Equals 1 - top1_overlap; "
                        "reported explicitly for clarity in spatial analysis."
                    ),
                    "per_block_mass_R<r>_clean": (
                        "L-Err@R baseline: cumulative attention mass in "
                        "CLEAN attention falling within Chebyshev distance "
                        "R of the clean argmax position. Reported as sanity "
                        "reference; for clean attention this should be high "
                        "for small R."
                    ),
                    "per_block_mass_R<r>_attacked": (
                        "L-Err@R: cumulative attention mass in ATTACKED "
                        "attention falling within Chebyshev distance R of "
                        "the CLEAN argmax position. Lower values = attack "
                        "displaces mass farther from original target. "
                        "Compare against mass_R<r>_clean to see degradation."
                    ),
                    "per_block_entropy": (
                        "Mean Shannon entropy of attacked attention rows. "
                        "Continuity with v1/v2."
                    ),
                    "per_block_top1_overlap": (
                        "Fraction of positions where argmax key is identical "
                        "in clean and attacked. PRIMARY DISCRETE METRIC."
                    ),
                    "per_block_top5_overlap": (
                        "Mean |top5_clean intersect top5_attacked| / 5."
                    ),
                    "per_block_kl": (
                        "Mean KL(A_clean || A_attacked) per attention row."
                    ),
                    "per_block_js": (
                        "Mean Jensen-Shannon divergence per attention row. "
                        "Bounded by log(2). Cleaner for cross-PE comparison."
                    ),
                },
                "narrative_layers": {
                    "layer_1_statistical": (
                        "'Did something happen?' - accuracy, top5_overlap, "
                        "KL/JS divergence. Symptoms only."
                    ),
                    "layer_2_geometric": (
                        "'Where does the attack send attention?' - Mean Jump "
                        "Distance (especially mad_miss). Distinguishes "
                        "local misdirection (RoPE hypothesis) from global "
                        "misdirection (Learned/Sinusoidal/ALiBi)."
                    ),
                    "layer_3_semantic": (
                        "'Does attention stay near the target region?' - "
                        "L-Err@R curves. Local concentration vs global "
                        "dispersal."
                    ),
                },
                "purpose": (
                    "v2 showed that PE attack redirects rather than diffuses "
                    "attention. v3 measures the SPATIAL CHARACTER of the "
                    "redirection: are misses local (neighbor patch) or "
                    "global (across the image)? Answers the central "
                    "mechanistic question of why RoPE's accuracy degrades "
                    "more gracefully than its top-k overlap suggests."
                ),
            },
            "results": {},
        }

    overall_start = time.time()
    total_combinations = len(PE_TYPES) * len(SEEDS)
    combo_idx = 0

    for pe_type in PE_TYPES:
        if pe_type not in results["results"]:
            results["results"][pe_type] = {}

        for seed in SEEDS:
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

            print(f"\n  --- Per-eps paired analysis ---", flush=True)
            print(f"  {'eps':<6} {'cln':<7} {'att':<7} "
                  f"{'top1':<7} {'MAD_a':<7} {'MAD_m':<7} "
                  f"{'M_R1':<7} {'M_R2':<7} {'KL':<7} {'time':<7}", flush=True)

            for eps in EPSILONS_PER_PE[pe_type]:
                t_eps = time.time()
                try:
                    deltas, origs = pgd_pe_compute_deltas(
                        model, val_loader, device, pe_type, eps, seed=seed
                    )
                    metrics = evaluate_paired(
                        model, val_loader, device, pe_type, deltas, origs,
                        grid_size
                    )
                    elapsed = time.time() - t_eps

                    # Aggregate scalars
                    pb = metrics
                    means = {
                        'mean_entropy':      sum(pb['per_block_entropy']) / 12,
                        'mean_top1_overlap': sum(pb['per_block_top1_overlap']) / 12,
                        'mean_top5_overlap': sum(pb['per_block_top5_overlap']) / 12,
                        'mean_kl':           sum(pb['per_block_kl']) / 12,
                        'mean_js':           sum(pb['per_block_js']) / 12,
                        'mean_mad_all':      sum(pb['per_block_mad_all']) / 12,
                        'mean_mad_miss':     sum(pb['per_block_mad_miss']) / 12,
                        'mean_miss_rate':    sum(pb['per_block_miss_rate']) / 12,
                    }
                    for R in LOCALIZATION_RADII:
                        means[f'mean_mass_R{R}_clean']    = sum(pb[f'per_block_mass_R{R}_clean']) / 12
                        means[f'mean_mass_R{R}_attacked'] = sum(pb[f'per_block_mass_R{R}_attacked']) / 12

                    # Build attack record
                    attack_record = {
                        "clean_accuracy":    float(pb['clean_accuracy']),
                        "attacked_accuracy": float(pb['attacked_accuracy']),
                        "elapsed_sec":       float(elapsed),
                    }
                    # Per-block lists
                    pb_keys = [k for k in pb if k.startswith('per_block_')]
                    for k in pb_keys:
                        attack_record[k] = [float(x) for x in pb[k]]
                    # Mean scalars
                    for k, v in means.items():
                        attack_record[k] = float(v)

                    run_data["attacks"][str(eps)] = attack_record

                    print(f"  {eps:<6} "
                          f"{pb['clean_accuracy']:>5.1f}% "
                          f"{pb['attacked_accuracy']:>5.1f}% "
                          f"{means['mean_top1_overlap']:>5.3f} "
                          f"{means['mean_mad_all']:>5.2f} "
                          f"{means['mean_mad_miss']:>5.2f} "
                          f"{means['mean_mass_R1_attacked']:>5.3f} "
                          f"{means['mean_mass_R2_attacked']:>5.3f} "
                          f"{means['mean_kl']:>5.3f} "
                          f"({elapsed:.0f}s)", flush=True)

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

    # Summary
    print(f"\n{'='*78}", flush=True)
    print("EXPERIMENT 3 v3 COMPLETE", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"Output: {args.output_path}", flush=True)
    print(f"Total time: {(time.time()-overall_start)/60:.1f} min", flush=True)

    import statistics
    print(f"\nPer-PE summary (mean ± std over seeds):", flush=True)
    print(f"{'PE':<11} {'eps':<6} {'att_acc':<14} {'top1':<14} "
          f"{'MAD_miss':<14} {'mass_R1':<14}", flush=True)
    print("-" * 90, flush=True)
    for pe_type in PE_TYPES:
        for eps in EPSILONS_PER_PE[pe_type]:
            accs, t1s, madms, m1s = [], [], [], []
            for seed in SEEDS:
                try:
                    d = results["results"][pe_type][str(seed)]["attacks"][str(eps)]
                    if d.get("attacked_accuracy") is not None:
                        accs.append(d["attacked_accuracy"])
                        t1s.append(d["mean_top1_overlap"])
                        madms.append(d["mean_mad_miss"])
                        m1s.append(d["mean_mass_R1_attacked"])
                except (KeyError, TypeError):
                    pass
            if accs:
                fmt = lambda xs, p=2: (f"{statistics.mean(xs):>5.{p}f}±"
                                       f"{(statistics.stdev(xs) if len(xs)>1 else 0):<5.{p}f}")
                print(f"{pe_type:<11} {eps:<6} {fmt(accs):<14} {fmt(t1s,3):<14} "
                      f"{fmt(madms):<14} {fmt(m1s,3):<14}", flush=True)
        print(flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--val_dir", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset", required=True, choices=['imagenet', 'cifar'])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.dataset == 'imagenet' and not args.val_dir:
        parser.error("--val_dir required for imagenet")

    run_experiment3_v3(args)


if __name__ == "__main__":
    main()
