"""
CIFAR-100 Training + Adversarial Attacks
=========================================
Trains 12 ViT models (4 PE types × 3 seeds) on CIFAR-100,
then runs FGSM-PE, PGD-PE, and VTA attacks.

Usage in Colab:
  1. Mount Drive
  2. Download CIFAR-100 (automatic via torchvision)
  3. Copy full_scale_experiment.py to /content/
  4. Copy this file to /content/
  5. Run: !python /content/cifar100_experiment.py

Total time: ~20-30h on T4/A100 (training ~18-24h + attacks ~5-6h)
"""

import os, sys, json, time, copy, argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import VisionTransformer from the main script
sys.path.insert(0, '/content')
from full_scale_experiment import VisionTransformer, extract_positional_embedding

# ============================================================
# CIFAR-100 CONFIG (differs from ImageNet-100)
# ============================================================
CIFAR_CONFIG = {
    'img_size': 32,
    'patch_size': 4,       # 32/4 = 8×8 = 64 patches
    'num_classes': 100,
    'embed_dim': 768,      # SAME as ImageNet for fair comparison
    'depth': 12,           # SAME as ImageNet
    'num_heads': 12,       # SAME as ImageNet
    'mlp_ratio': 4.0,
    'dropout': 0.1,
    'epochs': 300,
    'batch_size': 128,     # smaller batch due to larger model
    'lr': 3e-4,            # SAME as ImageNet
    'warmup_epochs': 20,
    'weight_decay': 0.1,
    'label_smoothing': 0.1,
    'mixup_alpha': 0.8,
}

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
SEEDS = [42, 123, 456]
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
PGD_STEPS = 20

DRIVE_BASE = '/content/drive/My Drive/pe_experiment'
RESULTS_DIR = os.path.join(DRIVE_BASE, 'results_cifar100')
DATA_DIR = '/content/cifar100'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# DATA LOADERS
# ============================================================
CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR_STD = [0.2675, 0.2565, 0.2761]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

train_dataset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform)
val_dataset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=CIFAR_CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CIFAR_CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")


# ============================================================
# TRAINING
# ============================================================
def mixup_data(x, y, alpha=0.8):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

def train_model(pe_type, seed):
    """Train one ViT model on CIFAR-100."""
    save_dir = os.path.join(RESULTS_DIR, f'{pe_type}_seed{seed}')
    os.makedirs(save_dir, exist_ok=True)


    cfg = CIFAR_CONFIG
    
    # Check if fully trained (history has all epochs)
    hist_path = os.path.join(save_dir, 'training_history.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            h = json.load(f)
        if len(h['val_acc']) >= cfg['epochs']:
            print(f"  Already trained ({len(h['val_acc'])} epochs)! Skipping.")
            return

    torch.manual_seed(seed)
    np.random.seed(seed)

    
    model = VisionTransformer(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'],
        num_classes=cfg['num_classes'], embed_dim=cfg['embed_dim'],
        depth=cfg['depth'], num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'], dropout=cfg['dropout'],
        pe_type=pe_type
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    def lr_lambda(epoch):
        if epoch < cfg['warmup_epochs']:
            return epoch / cfg['warmup_epochs']
        progress = (epoch - cfg['warmup_epochs']) / (cfg['epochs'] - cfg['warmup_epochs'])
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'epoch_time': []}
    best_acc = 0
    start_epoch = 1

    # Resume from checkpoint if exists
    ckpt_path = os.path.join(save_dir, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_acc = ckpt['best_acc']
        history = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"  Resumed from epoch {ckpt['epoch']}, best_acc={best_acc:.2f}%")

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        model.train()
        t0 = time.time()
        running_loss = 0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            mixed_x, y_a, y_b, lam = mixup_data(images, labels, cfg['mixup_alpha'])
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        train_loss = running_loss / max(num_batches, 1)

        val_acc = evaluate(model, val_loader, device)
        val_loss = 0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{cfg['epochs']}: loss={train_loss:.3f} acc={val_acc:.2f}% best={best_acc:.2f}% ({epoch_time:.1f}s)")

        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, os.path.join(save_dir, 'checkpoint.pth'))

    # Save history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"  ✅ Best accuracy: {best_acc:.2f}%")
    return best_acc


# ============================================================
# ADVERSARIAL ATTACKS (same as adversarial_pe_attacks.py but for CIFAR models)
# ============================================================
def get_pe_variance_weights(model, pe_type):
    pe_matrix = extract_positional_embedding(model, pe_type)
    if pe_matrix is None:
        return np.ones(CIFAR_CONFIG['embed_dim'])
    var_per_dim = np.var(pe_matrix, axis=0)
    if var_per_dim.max() > 0:
        return var_per_dim / var_per_dim.max()
    return np.ones_like(var_per_dim)


def fgsm_pe_attack(model, loader, device, pe_type, epsilon):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    num_batches = min(20, len(loader))

    if pe_type == 'learned':
        pe_param = model.pos_encoding.pos_embed
        original = pe_param.data.clone()
    elif pe_type == 'sinusoidal':
        pe_param = model.pos_encoding.pe
        original = pe_param.data.clone()
        pe_param.requires_grad_(True)
    elif pe_type == 'rope':
        cos_c = model.blocks[0].attn.rope.cos_cached
        sin_c = model.blocks[0].attn.rope.sin_cached
        orig_cos, orig_sin = cos_c.data.clone(), sin_c.data.clone()
        cos_c.requires_grad_(True); sin_c.requires_grad_(True)
    elif pe_type == 'alibi':
        slopes = model.blocks[0].attn.alibi.slopes
        orig_slopes = slopes.data.clone()
        slopes.requires_grad_(True)

    model.zero_grad()
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches: break
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()

    # Apply perturbation
    if pe_type in ['learned', 'sinusoidal']:
        if pe_param.grad is not None:
            pe_param.data = original + epsilon * pe_param.grad.sign()
    elif pe_type == 'rope':
        if cos_c.grad is not None:
            cos_c.data = orig_cos + epsilon * cos_c.grad.sign()
            sin_c.data = orig_sin + epsilon * sin_c.grad.sign()
    elif pe_type == 'alibi':
        if slopes.grad is not None:
            slopes.data = orig_slopes + epsilon * slopes.grad.sign()

    acc = evaluate(model, loader, device)

    # Restore
    if pe_type in ['learned', 'sinusoidal']:
        pe_param.data = original
        if pe_type == 'sinusoidal': pe_param.requires_grad_(False)
    elif pe_type == 'rope':
        cos_c.data, sin_c.data = orig_cos, orig_sin
        cos_c.requires_grad_(False); sin_c.requires_grad_(False)
    elif pe_type == 'alibi':
        slopes.data = orig_slopes; slopes.requires_grad_(False)
    model.zero_grad()
    return acc


def pgd_pe_attack(model, loader, device, pe_type, epsilon, steps=20):
    model.eval()
    alpha = epsilon * 0.1
    criterion = nn.CrossEntropyLoss()
    num_batches = min(20, len(loader))

    if pe_type in ['learned', 'sinusoidal']:
        pe_param = model.pos_encoding.pos_embed if pe_type == 'learned' else model.pos_encoding.pe
        original = pe_param.data.clone()
        delta = torch.zeros_like(pe_param.data).uniform_(-epsilon, epsilon)
        if pe_type == 'sinusoidal': pe_param.requires_grad_(True)
    elif pe_type == 'rope':
        cos_c = model.blocks[0].attn.rope.cos_cached
        sin_c = model.blocks[0].attn.rope.sin_cached
        orig_cos, orig_sin = cos_c.data.clone(), sin_c.data.clone()
        delta_cos = torch.zeros_like(cos_c.data).uniform_(-epsilon, epsilon)
        delta_sin = torch.zeros_like(sin_c.data).uniform_(-epsilon, epsilon)
        cos_c.requires_grad_(True); sin_c.requires_grad_(True)
    elif pe_type == 'alibi':
        slopes = model.blocks[0].attn.alibi.slopes
        orig_slopes = slopes.data.clone()
        delta_s = torch.zeros_like(slopes.data).uniform_(-epsilon, epsilon)
        slopes.requires_grad_(True)

    for step in range(steps):
        if pe_type in ['learned', 'sinusoidal']:
            pe_param.data = original + delta
        elif pe_type == 'rope':
            cos_c.data = orig_cos + delta_cos; sin_c.data = orig_sin + delta_sin
        elif pe_type == 'alibi':
            slopes.data = orig_slopes + delta_s

        model.zero_grad()
        for i, (images, labels) in enumerate(loader):
            if i >= num_batches: break
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            loss.backward()

        if pe_type in ['learned', 'sinusoidal']:
            if pe_param.grad is not None:
                delta = torch.clamp(delta + alpha * pe_param.grad.sign(), -epsilon, epsilon)
        elif pe_type == 'rope':
            if cos_c.grad is not None:
                delta_cos = torch.clamp(delta_cos + alpha * cos_c.grad.sign(), -epsilon, epsilon)
                delta_sin = torch.clamp(delta_sin + alpha * sin_c.grad.sign(), -epsilon, epsilon)
        elif pe_type == 'alibi':
            if slopes.grad is not None:
                delta_s = torch.clamp(delta_s + alpha * slopes.grad.sign(), -epsilon, epsilon)

    # Final perturbation
    if pe_type in ['learned', 'sinusoidal']:
        pe_param.data = original + delta
    elif pe_type == 'rope':
        cos_c.data = orig_cos + delta_cos; sin_c.data = orig_sin + delta_sin
    elif pe_type == 'alibi':
        slopes.data = orig_slopes + delta_s

    acc = evaluate(model, loader, device)

    # Restore
    if pe_type in ['learned', 'sinusoidal']:
        pe_param.data = original
        if pe_type == 'sinusoidal': pe_param.requires_grad_(False)
    elif pe_type == 'rope':
        cos_c.data, sin_c.data = orig_cos, orig_sin
        cos_c.requires_grad_(False); sin_c.requires_grad_(False)
    elif pe_type == 'alibi':
        slopes.data = orig_slopes; slopes.requires_grad_(False)
    model.zero_grad()
    return acc


def vta_attack(model, loader, device, pe_type, epsilon, var_weights):
    if pe_type not in ['learned', 'sinusoidal']:
        return fgsm_pe_attack(model, loader, device, pe_type, epsilon)

    model.eval()
    pe_param = model.pos_encoding.pos_embed if pe_type == 'learned' else model.pos_encoding.pe
    original = pe_param.data.clone()
    if pe_type == 'sinusoidal': pe_param.requires_grad_(True)

    criterion = nn.CrossEntropyLoss()
    num_batches = min(20, len(loader))
    model.zero_grad()
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches: break
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        loss.backward()

    if pe_param.grad is not None:
        w = torch.tensor(var_weights, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        pe_param.data = original + epsilon * w * pe_param.grad.sign()

    acc = evaluate(model, loader, device)
    pe_param.data = original
    if pe_type == 'sinusoidal': pe_param.requires_grad_(False)
    model.zero_grad()
    return acc


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ========== PHASE 1: TRAINING ==========
    print("\n" + "="*60)
    print("  PHASE 1: Training 12 models on CIFAR-100")
    print("="*60)

    for pe_type in PE_TYPES:
        for seed in SEEDS:
            print(f"\n{'='*50}")
            print(f"  {pe_type.upper()} PE, seed={seed}")
            print(f"{'='*50}")
            train_model(pe_type, seed)

    # ========== PHASE 2: ATTACKS ==========
    print("\n" + "="*60)
    print("  PHASE 2: Adversarial attacks on all 12 models")
    print("="*60)

    attack_results = {}
    save_path = os.path.join(RESULTS_DIR, 'adversarial_pe_results_cifar100.json')

    for pe_type in PE_TYPES:
        attack_results[pe_type] = {}
        for seed in SEEDS:
            print(f"\n{'='*50}")
            print(f"  ATTACK: {pe_type.upper()} PE, seed={seed}")
            print(f"{'='*50}")

            model_path = os.path.join(RESULTS_DIR, f'{pe_type}_seed{seed}', 'best_model.pth')
            if not os.path.exists(model_path):
                print(f"  SKIP: model not found")
                continue

            cfg = CIFAR_CONFIG
            torch.manual_seed(seed)
            model = VisionTransformer(
                img_size=cfg['img_size'], patch_size=cfg['patch_size'],
                num_classes=cfg['num_classes'], embed_dim=cfg['embed_dim'],
                depth=cfg['depth'], num_heads=cfg['num_heads'],
                mlp_ratio=cfg['mlp_ratio'], dropout=cfg['dropout'],
                pe_type=pe_type
            ).to(device)
            state = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
            model.eval()

            result = {'clean_acc': None, 'fgsm_pe': {}, 'pgd_pe': {}, 'vta': {}, 'variance_gini': None}

            # Clean
            clean_acc = evaluate(model, val_loader, device)
            result['clean_acc'] = clean_acc
            print(f"  Clean: {clean_acc:.2f}%")

            # Variance weights
            var_weights = get_pe_variance_weights(model, pe_type)
            sorted_v = np.sort(var_weights)
            n = len(sorted_v)
            gini = (2 * np.sum(np.arange(1, n+1) * sorted_v) / (n * np.sum(sorted_v))) - (n+1)/n
            result['variance_gini'] = float(gini)

            # FGSM-PE
            print(f"\n  --- FGSM-PE ---")
            for eps in EPSILONS:
                t0 = time.time()
                acc = fgsm_pe_attack(model, val_loader, device, pe_type, eps)
                result['fgsm_pe'][str(eps)] = acc
                print(f"    ε={eps:<6} acc={acc:>6.2f}% Δ={clean_acc-acc:>+6.2f}pp ({time.time()-t0:.1f}s)")

            # PGD-PE
            print(f"\n  --- PGD-PE ---")
            for eps in EPSILONS:
                t0 = time.time()
                acc = pgd_pe_attack(model, val_loader, device, pe_type, eps, PGD_STEPS)
                result['pgd_pe'][str(eps)] = acc
                print(f"    ε={eps:<6} acc={acc:>6.2f}% Δ={clean_acc-acc:>+6.2f}pp ({time.time()-t0:.1f}s)")

            # VTA
            print(f"\n  --- VTA ---")
            for eps in EPSILONS:
                t0 = time.time()
                acc = vta_attack(model, val_loader, device, pe_type, eps, var_weights)
                result['vta'][str(eps)] = acc
                print(f"    ε={eps:<6} acc={acc:>6.2f}% Δ={clean_acc-acc:>+6.2f}pp ({time.time()-t0:.1f}s)")

            attack_results[pe_type][str(seed)] = result

            # Save incrementally
            with open(save_path, 'w') as f:
                json.dump(attack_results, f, indent=2)
            print(f"\n  ✅ Saved to {save_path}")

    print("\n" + "="*60)
    print("  ALL DONE!")
    print("="*60)
