"""
Adversarial Positional Encoding Attacks
========================================
Runs FGSM-PE, PGD-PE, and Variance-Targeted Attack (VTA) on all 12 pre-trained
ViT-Base models. Saves results to JSON for paper tables and figures.

Requirements: GPU, imagenet100_resized on SSD, full_scale_experiment.py copied to /content/
"""

import os, json, sys, time, copy
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer, extract_positional_embedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# CONFIG
# ============================================================
RESULTS = '/content/drive/My Drive/pe_experiment/results'
DATA_DIR = '/content/imagenet100_resized'
SAVE_PATH = os.path.join(RESULTS, 'adversarial_pe_results.json')

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS = [42, 123, 456]
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
PGD_STEPS = 20
PGD_ALPHA_RATIO = 0.1  # alpha = eps * ratio

# ============================================================
# DATA
# ============================================================
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)
print(f"Val: {len(val_dataset)} images")

# ============================================================
# HELPER: Get PE parameter reference
# ============================================================
def get_pe_param(model, pe_type):
    """Returns the PE parameter tensor that we will attack.
    For Learned/Sinusoidal: the pos_embed tensor
    For RoPE: the cos/sin cached tensors
    For ALiBi: the slopes tensor
    """
    if pe_type == 'learned':
        return model.pos_encoding.pos_embed  # (1, N, d)
    elif pe_type == 'sinusoidal':
        return model.pos_encoding.pe  # (1, N, d) - registered buffer
    elif pe_type == 'rope':
        # Attack the cos/sin caches
        rope = model.blocks[0].attn.rope
        return rope.cos_cached, rope.sin_cached
    elif pe_type == 'alibi':
        alibi = model.blocks[0].attn.alibi
        return alibi.slopes  # (num_heads,)
    return None


def get_pe_variance_weights(model, pe_type):
    """Compute per-dimension variance weights for VTA attack."""
    pe_matrix = extract_positional_embedding(model, pe_type)
    if pe_matrix is None:
        return None
    # Variance per dimension across positions
    var_per_dim = np.var(pe_matrix, axis=0)
    # Normalize to [0, 1]
    if var_per_dim.max() > 0:
        weights = var_per_dim / var_per_dim.max()
    else:
        weights = np.ones_like(var_per_dim)
    return weights


# ============================================================
# ATTACK FUNCTIONS
# ============================================================
@torch.no_grad()
def evaluate_clean(model, val_loader, device):
    """Evaluate model accuracy without any attack."""
    model.eval()
    correct, total = 0, 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def fgsm_pe_attack(model, val_loader, device, pe_type, epsilon):
    """FGSM attack on PE parameters.
    Computes gradient of loss w.r.t. PE, then adds epsilon * sign(grad).
    """
    model.eval()

    # For Learned PE: parameter, requires grad already
    # For Sinusoidal/RoPE/ALiBi: buffer, need to enable grad temporarily
    if pe_type == 'learned':
        pe_param = model.pos_encoding.pos_embed
        original_pe = pe_param.data.clone()
    elif pe_type == 'sinusoidal':
        pe_param = model.pos_encoding.pe
        original_pe = pe_param.data.clone()
        pe_param.requires_grad_(True)
    elif pe_type == 'rope':
        cos_c = model.blocks[0].attn.rope.cos_cached
        sin_c = model.blocks[0].attn.rope.sin_cached
        orig_cos = cos_c.data.clone()
        orig_sin = sin_c.data.clone()
        cos_c.requires_grad_(True)
        sin_c.requires_grad_(True)
    elif pe_type == 'alibi':
        slopes = model.blocks[0].attn.alibi.slopes
        orig_slopes = slopes.data.clone()
        slopes.requires_grad_(True)

    # Accumulate gradients over a few batches (not all, for speed)
    criterion = nn.CrossEntropyLoss()
    num_batches = min(20, len(val_loader))  # Use 20 batches for gradient estimation

    for i, (images, labels) in enumerate(val_loader):
        if i >= num_batches:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

    # Apply FGSM perturbation
    if pe_type in ['learned', 'sinusoidal']:
        if pe_param.grad is not None:
            perturbation = epsilon * pe_param.grad.sign()
            pe_param.data = original_pe + perturbation
        else:
            pe_param.data = original_pe
    elif pe_type == 'rope':
        if cos_c.grad is not None:
            cos_c.data = orig_cos + epsilon * cos_c.grad.sign()
            sin_c.data = orig_sin + epsilon * sin_c.grad.sign()
        else:
            cos_c.data = orig_cos
            sin_c.data = orig_sin
    elif pe_type == 'alibi':
        if slopes.grad is not None:
            slopes.data = orig_slopes + epsilon * slopes.grad.sign()
        else:
            slopes.data = orig_slopes

    # Evaluate under attack
    acc = evaluate_clean(model, val_loader, device)

    # Restore original PE
    if pe_type in ['learned', 'sinusoidal']:
        pe_param.data = original_pe
        if pe_type == 'sinusoidal':
            pe_param.requires_grad_(False)
    elif pe_type == 'rope':
        cos_c.data = orig_cos
        sin_c.data = orig_sin
        cos_c.requires_grad_(False)
        sin_c.requires_grad_(False)
    elif pe_type == 'alibi':
        slopes.data = orig_slopes
        slopes.requires_grad_(False)

    # Zero gradients
    model.zero_grad()

    return acc


def pgd_pe_attack(model, val_loader, device, pe_type, epsilon, steps=20, alpha_ratio=0.1):
    """PGD attack on PE parameters. Multi-step version of FGSM-PE."""
    model.eval()
    alpha = epsilon * alpha_ratio
    criterion = nn.CrossEntropyLoss()
    num_batches = min(20, len(val_loader))

    # Save originals and init perturbation
    if pe_type in ['learned', 'sinusoidal']:
        pe_param = model.pos_encoding.pos_embed if pe_type == 'learned' else model.pos_encoding.pe
        original_pe = pe_param.data.clone()
        delta = torch.zeros_like(pe_param.data).uniform_(-epsilon, epsilon)
        if pe_type == 'sinusoidal':
            pe_param.requires_grad_(True)
    elif pe_type == 'rope':
        cos_c = model.blocks[0].attn.rope.cos_cached
        sin_c = model.blocks[0].attn.rope.sin_cached
        orig_cos, orig_sin = cos_c.data.clone(), sin_c.data.clone()
        delta_cos = torch.zeros_like(cos_c.data).uniform_(-epsilon, epsilon)
        delta_sin = torch.zeros_like(sin_c.data).uniform_(-epsilon, epsilon)
        cos_c.requires_grad_(True)
        sin_c.requires_grad_(True)
    elif pe_type == 'alibi':
        slopes = model.blocks[0].attn.alibi.slopes
        orig_slopes = slopes.data.clone()
        delta_s = torch.zeros_like(slopes.data).uniform_(-epsilon, epsilon)
        slopes.requires_grad_(True)

    for step in range(steps):
        # Apply current perturbation
        if pe_type in ['learned', 'sinusoidal']:
            pe_param.data = original_pe + delta
        elif pe_type == 'rope':
            cos_c.data = orig_cos + delta_cos
            sin_c.data = orig_sin + delta_sin
        elif pe_type == 'alibi':
            slopes.data = orig_slopes + delta_s

        # Forward + backward on subset
        model.zero_grad()
        for i, (images, labels) in enumerate(val_loader):
            if i >= num_batches:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

        # Update perturbation
        if pe_type in ['learned', 'sinusoidal']:
            if pe_param.grad is not None:
                delta = delta + alpha * pe_param.grad.sign()
                delta = torch.clamp(delta, -epsilon, epsilon)
        elif pe_type == 'rope':
            if cos_c.grad is not None:
                delta_cos = delta_cos + alpha * cos_c.grad.sign()
                delta_cos = torch.clamp(delta_cos, -epsilon, epsilon)
                delta_sin = delta_sin + alpha * sin_c.grad.sign()
                delta_sin = torch.clamp(delta_sin, -epsilon, epsilon)
        elif pe_type == 'alibi':
            if slopes.grad is not None:
                delta_s = delta_s + alpha * slopes.grad.sign()
                delta_s = torch.clamp(delta_s, -epsilon, epsilon)

    # Apply final perturbation and evaluate
    if pe_type in ['learned', 'sinusoidal']:
        pe_param.data = original_pe + delta
    elif pe_type == 'rope':
        cos_c.data = orig_cos + delta_cos
        sin_c.data = orig_sin + delta_sin
    elif pe_type == 'alibi':
        slopes.data = orig_slopes + delta_s

    acc = evaluate_clean(model, val_loader, device)

    # Restore
    if pe_type in ['learned', 'sinusoidal']:
        pe_param.data = original_pe
        if pe_type == 'sinusoidal':
            pe_param.requires_grad_(False)
    elif pe_type == 'rope':
        cos_c.data, sin_c.data = orig_cos, orig_sin
        cos_c.requires_grad_(False)
        sin_c.requires_grad_(False)
    elif pe_type == 'alibi':
        slopes.data = orig_slopes
        slopes.requires_grad_(False)

    model.zero_grad()
    return acc


def vta_attack(model, val_loader, device, pe_type, epsilon, var_weights):
    """Variance-Targeted Attack: allocates perturbation budget
    proportionally to per-dimension variance.
    Only works for Learned and Sinusoidal PE (additive in embedding space).
    For RoPE/ALiBi, falls back to FGSM-PE.
    """
    if pe_type not in ['learned', 'sinusoidal']:
        return fgsm_pe_attack(model, val_loader, device, pe_type, epsilon)

    model.eval()
    pe_param = model.pos_encoding.pos_embed if pe_type == 'learned' else model.pos_encoding.pe
    original_pe = pe_param.data.clone()

    if pe_type == 'sinusoidal':
        pe_param.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()
    num_batches = min(20, len(val_loader))

    # Get gradient direction
    model.zero_grad()
    for i, (images, labels) in enumerate(val_loader):
        if i >= num_batches:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

    if pe_param.grad is not None:
        # Variance-weighted perturbation
        weights_tensor = torch.tensor(var_weights, dtype=torch.float32, device=device)
        # Broadcast weights across positions: (1, 1, d) for (1, N, d) PE
        weights_tensor = weights_tensor.unsqueeze(0).unsqueeze(0)
        perturbation = epsilon * weights_tensor * pe_param.grad.sign()
        pe_param.data = original_pe + perturbation

    acc = evaluate_clean(model, val_loader, device)

    # Restore
    pe_param.data = original_pe
    if pe_type == 'sinusoidal':
        pe_param.requires_grad_(False)
    model.zero_grad()

    return acc


# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================
all_results = {}

for pe_type in PE_TYPES:
    all_results[pe_type] = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  {pe_type.upper()} PE, seed={seed}")
        print(f"{'='*60}")

        # Load model
        model_path = os.path.join(RESULTS, f'{pe_type}_seed{seed}', 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"  SKIP: model not found at {model_path}")
            continue

        torch.manual_seed(seed)
        model = VisionTransformer(
            img_size=224, patch_size=16, num_classes=100, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
        ).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
        model.eval()

        result = {
            'clean_acc': None,
            'fgsm_pe': {},
            'pgd_pe': {},
            'vta': {},
            'variance_gini': None
        }

        # Clean accuracy
        clean_acc = evaluate_clean(model, val_loader, device)
        result['clean_acc'] = clean_acc
        print(f"  Clean accuracy: {clean_acc:.2f}%")

        # Compute variance weights for VTA
        var_weights = get_pe_variance_weights(model, pe_type)
        if var_weights is not None:
            # Gini coefficient of variance distribution
            sorted_v = np.sort(var_weights)
            n = len(sorted_v)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_v) / (n * np.sum(sorted_v))) - (n + 1) / n
            result['variance_gini'] = float(gini)
            print(f"  Variance Gini coefficient: {gini:.4f}")

        # FGSM-PE attacks
        print(f"\n  --- FGSM-PE ---")
        for eps in EPSILONS:
            t0 = time.time()
            acc = fgsm_pe_attack(model, val_loader, device, pe_type, eps)
            dt = time.time() - t0
            result['fgsm_pe'][str(eps)] = acc
            drop = clean_acc - acc
            print(f"    ε={eps:<6} acc={acc:>6.2f}%  Δ={drop:>+6.2f}pp  ({dt:.1f}s)")

        # PGD-PE attacks
        print(f"\n  --- PGD-PE (T={PGD_STEPS}) ---")
        for eps in EPSILONS:
            t0 = time.time()
            acc = pgd_pe_attack(model, val_loader, device, pe_type, eps,
                               steps=PGD_STEPS, alpha_ratio=PGD_ALPHA_RATIO)
            dt = time.time() - t0
            result['pgd_pe'][str(eps)] = acc
            drop = clean_acc - acc
            print(f"    ε={eps:<6} acc={acc:>6.2f}%  Δ={drop:>+6.2f}pp  ({dt:.1f}s)")

        # VTA attacks (only meaningful for Learned/Sinusoidal)
        print(f"\n  --- VTA ---")
        for eps in EPSILONS:
            t0 = time.time()
            acc = vta_attack(model, val_loader, device, pe_type, eps, var_weights)
            dt = time.time() - t0
            result['vta'][str(eps)] = acc
            drop = clean_acc - acc
            fgsm_drop = clean_acc - result['fgsm_pe'][str(eps)]
            vta_gain = drop / max(fgsm_drop, 0.01)
            print(f"    ε={eps:<6} acc={acc:>6.2f}%  Δ={drop:>+6.2f}pp  VTA_gain={vta_gain:.2f}x  ({dt:.1f}s)")

        all_results[pe_type][str(seed)] = result

        # Save incrementally
        with open(SAVE_PATH, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  ✅ Saved to {SAVE_PATH}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("EXPERIMENT COMPLETE — SUMMARY")
print("="*80)

print(f"\n{'PE Type':<14} {'Gini':>6} {'ε_crit (FGSM)':>14} {'ε_crit (PGD)':>14}")
print("-"*50)

for pe_type in PE_TYPES:
    ginis, crit_fgsm, crit_pgd = [], [], []
    for seed in [str(s) for s in SEEDS]:
        if seed not in all_results.get(pe_type, {}):
            continue
        r = all_results[pe_type][seed]
        ginis.append(r['variance_gini'] or 0)
        clean = r['clean_acc']
        half = clean / 2

        # Find critical epsilon (acc < 50% of baseline)
        cf, cp = '>1.0', '>1.0'
        for eps in EPSILONS:
            if r['fgsm_pe'].get(str(eps), 100) < half:
                cf = str(eps)
                break
        for eps in EPSILONS:
            if r['pgd_pe'].get(str(eps), 100) < half:
                cp = str(eps)
                break
        crit_fgsm.append(cf)
        crit_pgd.append(cp)

    print(f"{pe_type:<14} {np.mean(ginis):>6.3f} {crit_fgsm[0] if crit_fgsm else 'N/A':>14} {crit_pgd[0] if crit_pgd else 'N/A':>14}")

print(f"\nResults saved to: {SAVE_PATH}")
print("Send this file for paper table generation.")
