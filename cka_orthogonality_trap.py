"""
CKA (Centered Kernel Alignment) Analysis for Orthogonality Trap Validation
===========================================================================
Purpose: Validate the Orthogonality Trap hypothesis by computing linear CKA
         between layer activations for different PE strategies.

Hypothesis: If Sinusoidal PE falls into the Orthogonality Trap, then:
  - CKA(Layer 1, Layer 12) will be HIGH for Sinusoidal (positional info passes unchanged)
  - CKA(Layer 1, Layer 12) will be LOW  for RoPE (representations are actively transformed)

Usage (Google Colab):
  1. Upload this script
  2. Set CHECKPOINT_DIR to your model weights path
  3. Set DATASET to 'imagenet100' or 'cifar100'
  4. Run all cells

Requirements:
  pip install torch torchvision matplotlib numpy seaborn
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — EDIT THESE
# ============================================================

CHECKPOINT_DIR = "/content/drive/My Drive/pe_experiment/results"  # Path to model weights
DATASET = "imagenet100"  # "imagenet100" or "cifar100"
DATA_DIR = "/content/imagenet100"

# Dataset-specific settings
if DATASET == "imagenet100":
    IMG_SIZE = 224
    PATCH_SIZE = 16
    NUM_PATCHES = 196  # 14x14
    NUM_CLASSES = 100
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)
else:  # cifar100
    IMG_SIZE = 32
    PATCH_SIZE = 4
    NUM_PATCHES = 64  # 8x8
    NUM_CLASSES = 100
    NORM_MEAN = (0.5071, 0.4867, 0.4408)
    NORM_STD = (0.2675, 0.2565, 0.2761)

# Model settings (ViT-Base)
EMBED_DIM = 768
NUM_LAYERS = 12
NUM_HEADS = 12
MLP_RATIO = 4.0
DROPOUT = 0.1

# CKA settings
NUM_IMAGES = 2000       # Number of validation images to use
BATCH_SIZE = 64         # Batch size for feature extraction
SEED = 42               # Random seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PE strategies to analyse
PE_TYPES = ["sinusoidal", "rope", "learned", "alibi"]

# Which seeds to average over (use one for speed, three for paper)
MODEL_SEEDS = [42, 123, 456] 

# ============================================================
# CKA IMPLEMENTATION (Kornblith et al., ICML 2019)
# ============================================================

def centering_matrix(n):
    """Return the centering matrix H = I - 1/n * 11^T"""
    return torch.eye(n, device=DEVICE) - torch.ones(n, n, device=DEVICE) / n


def linear_hsic(X, Y):
    """
    Compute HSIC with linear kernel.
    X, Y: (n_samples, n_features) tensors
    """
    n = X.shape[0]
    H = centering_matrix(n)
    # K = X @ X^T, L = Y @ Y^T
    K = X @ X.T
    L = Y @ Y.T
    # HSIC = 1/(n-1)^2 * tr(KHLH)
    return (K @ H @ L @ H).trace() / ((n - 1) ** 2)


def linear_cka(X, Y):
    """
    Compute Linear CKA between two sets of representations.
    X, Y: (n_samples, n_features) tensors
    Returns: scalar CKA value in [0, 1]
    """
    # Centre features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = linear_hsic(X, Y)
    hsic_xx = linear_hsic(X, X)
    hsic_yy = linear_hsic(Y, Y)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return torch.tensor(0.0)
    return hsic_xy / denom


def compute_cka_minibatch(X, Y, batch_size=500):
    """
    Memory-efficient CKA for large sample counts.
    Uses the unbiased HSIC estimator with minibatching.
    """
    n = X.shape[0]
    if n <= batch_size:
        return linear_cka(X, Y).item()

    # For large n, subsample to keep memory manageable
    indices = torch.randperm(n)[:batch_size]
    return linear_cka(X[indices], Y[indices]).item()


# ============================================================
# MODEL DEFINITION — imported from your training script
# ============================================================
# We import VisionTransformer directly from your training code.
# Place full_scale_experiment.py in the same directory or add to path.

import sys
sys.path.insert(0, "/content/drive/My Drive/pe_experiment")  # Adjust if needed

from full_scale_experiment import VisionTransformer


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def get_dataloader(dataset_name, data_dir, img_size, norm_mean, norm_std,
                   num_images, batch_size):
    """Create validation dataloader with deterministic subset."""
    transform = transforms.Compose([
        transforms.Resize(img_size) if dataset_name == "cifar100"
            else transforms.Resize(256),
        transforms.CenterCrop(img_size) if dataset_name != "cifar100"
            else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(data_dir, train=False,
                                     download=True, transform=transform)
    else:
        # For ImageNet-100, point to your validation folder
        val_dir = Path(data_dir) / "val"
        dataset = datasets.ImageFolder(str(val_dir), transform=transform)

    # Deterministic subset
    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset))[:num_images].tolist()
    subset = Subset(dataset, indices)

    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


@torch.no_grad()
def extract_all_layer_activations(model, dataloader, num_patches):
    """
    Extract post-block activations from all 12 layers using
    model.forward_layer_activations() from full_scale_experiment.py.
    Returns: dict mapping layer_idx -> (num_images * num_patches, embed_dim)
    
    We exclude the CLS token and concatenate patch activations across images.
    """
    model.eval()
    layer_acts = {i: [] for i in range(NUM_LAYERS)}

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        # forward_layer_activations returns list of 12 tensors,
        # each already detached and on CPU (see line 323 of your code)
        activations = model.forward_layer_activations(images)

        for layer_idx, act in enumerate(activations):
            # act shape: (B, N+1, D) — remove CLS token (index 0)
            patches = act[:, 1:, :]  # (B, N, D)
            # Flatten to (B*N, D)
            patches = patches.reshape(-1, patches.shape[-1])
            layer_acts[layer_idx].append(patches)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx+1) * dataloader.batch_size} images")

    # Concatenate all batches
    for layer_idx in layer_acts:
        layer_acts[layer_idx] = torch.cat(layer_acts[layer_idx], dim=0)
        print(f"  Layer {layer_idx+1}: {layer_acts[layer_idx].shape}")

    return layer_acts


def load_model(pe_type, seed, checkpoint_dir):
    """
    Load a trained model checkpoint.
    Directory structure: {checkpoint_dir}/{pe_type}_seed{seed}/best_model.pth
    """
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        pe_type=pe_type,
    ).to(DEVICE)

    ckpt_path = Path(checkpoint_dir) / f"{pe_type}_seed{seed}" / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at: {ckpt_path}\n"
            f"Expected structure: {checkpoint_dir}/{pe_type}_seed{seed}/best_model.pth"
        )

    print(f"Loading {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location=DEVICE)

    # Handle common checkpoint formats
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ============================================================
# CKA COMPUTATION
# ============================================================

def compute_cka_matrix(layer_acts, layers_to_compare=None):
    """
    Compute full CKA matrix between specified layers.
    layer_acts: dict from extract_all_layer_activations
    layers_to_compare: list of layer indices (0-based), default all 12
    Returns: (n_layers, n_layers) numpy array
    """
    if layers_to_compare is None:
        layers_to_compare = list(range(NUM_LAYERS))

    n = len(layers_to_compare)
    cka_matrix = np.zeros((n, n))

    for i, li in enumerate(layers_to_compare):
        for j, lj in enumerate(layers_to_compare):
            if i == j:
                cka_matrix[i, j] = 1.0
            elif j > i:
                X = layer_acts[li].to(DEVICE)
                Y = layer_acts[lj].to(DEVICE)
                cka_val = compute_cka_minibatch(X, Y, batch_size=1000)
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val  # Symmetric
                print(f"    CKA(L{li+1}, L{lj+1}) = {cka_val:.4f}")

    return cka_matrix


def compute_key_cka_pairs(layer_acts):
    """
    Compute CKA for the specific pairs needed for the paper:
    - CKA(L1, L6): early vs mid
    - CKA(L1, L12): early vs final (key Orthogonality Trap indicator)
    - CKA(L6, L12): mid vs final
    """
    pairs = [(0, 5), (0, 11), (5, 11)]  # 0-indexed
    results = {}

    for (li, lj) in pairs:
        X = layer_acts[li].to(DEVICE)
        Y = layer_acts[lj].to(DEVICE)
        cka_val = compute_cka_minibatch(X, Y, batch_size=1000)
        key = f"L{li+1}-L{lj+1}"
        results[key] = cka_val
        print(f"    CKA({key}) = {cka_val:.4f}")

    return results


# ============================================================
# VISUALIZATION — Publication-quality figures
# ============================================================

def plot_cka_bar_chart(all_results, save_path="cka_orthogonality_trap.pdf"):
    """
    Grouped bar chart: 4 PE types × 3 layer pairs.
    This is the primary figure for the paper.
    """
    pe_labels = {
        "learned": "Learned",
        "sinusoidal": "Sinusoidal",
        "rope": "RoPE",
        "alibi": "ALiBi",
    }

    pairs = ["L1-L6", "L1-L12", "L6-L12"]
    pe_order = ["learned", "sinusoidal", "rope", "alibi"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(pairs))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for idx, pe_type in enumerate(pe_order):
        if pe_type not in all_results:
            continue

        means = []
        stds = []
        for pair in pairs:
            vals = all_results[pe_type].get(pair, [])
            if isinstance(vals, list) and len(vals) > 1:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            elif isinstance(vals, list) and len(vals) == 1:
                means.append(vals[0])
                stds.append(0)
            else:
                means.append(vals if isinstance(vals, (int, float)) else 0)
                stds.append(0)

        bars = ax.bar(x + offsets[idx] * width, means, width,
                      yerr=stds if any(s > 0 for s in stds) else None,
                      label=pe_labels[pe_type], color=colors[idx],
                      edgecolor='white', linewidth=0.5,
                      capsize=3, error_kw={'linewidth': 1})

        # Value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8,
                    fontweight='normal')

    ax.set_xlabel('Layer Pair', fontsize=9)
    ax.set_ylabel('Linear CKA', fontsize=9)
    ax.set_title(f'CKA Between Layer Activations — {DATASET.upper()}',
                 fontsize=9, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_ylim(0, 1.10)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.show()


def plot_cka_heatmaps(cka_matrices, save_path="cka_heatmaps.pdf"):
    """
    Side-by-side CKA heatmaps for all PE types.
    Shows full 12×12 layer-to-layer CKA structure.
    """
    pe_labels = {
        "learned": "Learned",
        "sinusoidal": "Sinusoidal",
        "rope": "RoPE",
        "alibi": "ALiBi",
    }

    available = [pe for pe in ["learned", "sinusoidal", "rope", "alibi"]
                 if pe in cka_matrices]
    n_pe = len(available)

    fig, axes = plt.subplots(1, n_pe, figsize=(4.2 * n_pe, 4),
                              sharey=True)
    if n_pe == 1:
        axes = [axes]

    layer_labels = [str(i+1) for i in range(NUM_LAYERS)]

    for idx, pe_type in enumerate(available):
        ax = axes[idx]
        matrix = cka_matrices[pe_type]

        sns.heatmap(matrix, ax=ax, vmin=0, vmax=1, cmap='viridis',
                    xticklabels=layer_labels, yticklabels=layer_labels,
                    square=True, cbar=(idx == n_pe - 1),
                    cbar_kws={'label': 'Linear CKA', 'shrink': 0.8},
                    linewidths=0.3, linecolor='white')

        ax.set_title(pe_labels[pe_type], fontsize=9, fontweight='normal')
        ax.set_xlabel('Layer', fontsize=9)
        if idx == 0:
            ax.set_ylabel('Layer', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(f'Layer-wise CKA Structure — {DATASET.upper()}',
                 fontsize=9, fontweight='normal', y=1.10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("CKA Analysis for Orthogonality Trap Validation")
    print(f"Dataset: {DATASET} | Device: {DEVICE}")
    print(f"Images: {NUM_IMAGES} | Seeds: {MODEL_SEEDS}")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading validation data...")
    dataloader = get_dataloader(
        DATASET, DATA_DIR, IMG_SIZE, NORM_MEAN, NORM_STD,
        NUM_IMAGES, BATCH_SIZE
    )
    print(f"  Loaded {len(dataloader.dataset)} images")

    # 2. Extract features and compute CKA for each PE type
    all_bar_results = {}       # For bar chart: pe -> {pair: [values]}
    all_heatmap_matrices = {}  # For heatmaps: pe -> 12x12 matrix

    for pe_type in PE_TYPES:
        print(f"\n{'='*60}")
        print(f"[2/4] Processing: {pe_type.upper()}")
        print(f"{'='*60}")

        seed_bar_results = {p: [] for p in ["L1-L6", "L1-L12", "L6-L12"]}
        seed_matrices = []

        for seed in MODEL_SEEDS:
            print(f"\n  --- Seed {seed} ---")

            try:
                model = load_model(pe_type, seed, CHECKPOINT_DIR)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}")
                print(f"  Skipping {pe_type} seed {seed}")
                continue

            print(f"  Extracting activations...")
            layer_acts = extract_all_layer_activations(
                model, dataloader, NUM_PATCHES
            )

            # Key pairs for bar chart
            print(f"  Computing key CKA pairs...")
            pairs = compute_key_cka_pairs(layer_acts)
            for key, val in pairs.items():
                seed_bar_results[key].append(val)

            # Full 12x12 matrix (only for first seed, to save time)
            if seed == MODEL_SEEDS[0]:
                print(f"  Computing full 12x12 CKA matrix...")
                matrix = compute_cka_matrix(layer_acts)
                seed_matrices.append(matrix)

            # Free memory
            del layer_acts, model
            torch.cuda.empty_cache()

        if any(len(v) > 0 for v in seed_bar_results.values()):
            all_bar_results[pe_type] = seed_bar_results
        if seed_matrices:
            all_heatmap_matrices[pe_type] = seed_matrices[0]

    # 3. Generate figures
    print(f"\n{'='*60}")
    print("[3/4] Generating figures...")
    print(f"{'='*60}")

    if all_bar_results:
        results_dir = os.path.join(CHECKPOINT_DIR, "figures")
        os.makedirs(results_dir, exist_ok=True)
        plot_cka_bar_chart(all_bar_results,
                           save_path=os.path.join(results_dir, f"cka_bar_{DATASET}.pdf"))

    if all_heatmap_matrices:
        plot_cka_heatmaps(all_heatmap_matrices,
                           save_path=os.path.join(results_dir, f"cka_heatmaps_{DATASET}.pdf"))

    # 4. Print summary table for the paper
    print(f"\n{'='*60}")
    print("[4/4] Summary for paper")
    print(f"{'='*60}")
    print(f"\n{'PE Type':<12} {'CKA(L1,L6)':<14} {'CKA(L1,L12)':<14} {'CKA(L6,L12)':<14}")
    print("-" * 54)

    for pe_type in ["learned", "sinusoidal", "rope", "alibi"]:
        if pe_type not in all_bar_results:
            continue
        r = all_bar_results[pe_type]
        row = f"{pe_type:<12}"
        for pair in ["L1-L6", "L1-L12", "L6-L12"]:
            vals = r[pair]
            if len(vals) > 1:
                row += f" {np.mean(vals):.4f}±{np.std(vals):.4f} "
            elif len(vals) == 1:
                row += f" {vals[0]:.4f}        "
            else:
                row += f" {'N/A':<13}"
        print(row)

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE:")
    print("=" * 60)
    print("""
If Sinusoidal CKA(L1, L12) >> RoPE CKA(L1, L12):
  → CONFIRMS the Orthogonality Trap hypothesis.
    Sinusoidal's positional code traverses the network unchanged
    (parallel channel), while RoPE actively transforms representations.

If Sinusoidal CKA(L1, L12) ≈ RoPE CKA(L1, L12):
  → DOES NOT CONFIRM the hypothesis.
    Both strategies transform representations similarly;
    the decodability difference must arise from a different mechanism.

Expected values (based on paper's spatial decodability data):
  Sinusoidal CKA(L1, L12): HIGH (≥ 0.7)  — spatial info preserved
  RoPE       CKA(L1, L12): LOW  (≤ 0.4)  — representations transformed
  Learned    CKA(L1, L12): LOW  (≤ 0.3)  — aggressive spatial erasure
  ALiBi      CKA(L1, L12): MODERATE       — depends on resolution
    """)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
