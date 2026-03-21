#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Full-Scale Positional Encoding Analysis for Vision Transformers
=============================================================================

Quantifying the Information Content of Positional Encoding in ViT:
Learned vs. Sinusoidal vs. RoPE vs. ALiBi

Dataset:    ImageNet-100 (100 classes, 224x224, ~130K train / ~5K val)
Model:      ViT-Base (12 layers, 768 dim, 12 heads, patch 16x16 → 196 patches)
Seeds:      3 independent runs per configuration
PE types:   Sinusoidal, Learned, RoPE, ALiBi

Requirements:
    pip install torch torchvision timm matplotlib numpy scipy scikit-learn tqdm

Hardware:   1x H100/A100 GPU (~2-3h per run)
            Total: 12 runs × ~3h = ~36 GPU-hours

Usage:
    python full_scale_experiment.py --data_dir /path/to/imagenet100 --output_dir ./results

    python full_scale_experiment.py --data_dir /path/to/imagenet100 --mode train --pe_type learned --seed 42
=============================================================================
"""

import os
import io
import json
import argparse
import math
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# ============================================================================
# 1. MODEL DEFINITIONS
# ============================================================================

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


# --- Positional Encoding Variants ---

class LearnedPE(nn.Module):
    """Learnable positional encoding."""
    def __init__(self, num_positions, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_positions, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embed


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, num_positions, embed_dim):
        super().__init__()
        pe = torch.zeros(num_positions, embed_dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, num_positions, embed_dim)

    def forward(self, x):
        return x + self.pe


class RoPE(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Applied within the attention mechanism, not as additive PE.
    Returns position-encoded Q and K inside the attention block.
    """
    def __init__(self, num_positions, head_dim):
        super().__init__()
        # Precompute rotation frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute sin/cos for all positions
        t = torch.arange(num_positions, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # (num_positions, head_dim/2)
        self.register_buffer('cos_cached', freqs.cos().unsqueeze(0).unsqueeze(0))  # (1, 1, N, D/2)
        self.register_buffer('sin_cached', freqs.sin().unsqueeze(0).unsqueeze(0))

    def _rotate_half(self, x):
        """Split x into two halves and rotate."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len):
        """Apply rotary embeddings to Q and K.
        q, k: (B, num_heads, seq_len, head_dim)
        """
        cos = self.cos_cached[:, :, :seq_len, :]  # (1, 1, seq_len, D/2)
        sin = self.sin_cached[:, :, :seq_len, :]

        # Expand cos/sin to match head_dim (repeat for both halves)
        cos = torch.cat([cos, cos], dim=-1)  # (1, 1, seq_len, D)
        sin = torch.cat([sin, sin], dim=-1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class ALiBi(nn.Module):
    """Attention with Linear Biases (Press et al., 2022).

    Adds a linear bias to attention scores based on relative distance.
    No positional embedding is added to the input.
    """
    def __init__(self, num_heads, num_positions):
        super().__init__()
        # Compute slopes for each head (geometric series)
        # Following the original paper: slopes = 2^(-8/n * (1..n))
        ratio = 2 ** (-8.0 / num_heads)
        slopes = torch.tensor([ratio ** i for i in range(1, num_heads + 1)])
        self.register_buffer('slopes', slopes.view(1, num_heads, 1, 1))

        # Precompute relative distance matrix
        positions = torch.arange(num_positions)
        rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N)
        rel_dist = rel_dist.abs().float()
        self.register_buffer('rel_dist', rel_dist.unsqueeze(0).unsqueeze(0))  # (1, 1, N, N)

    def get_bias(self, seq_len):
        """Return attention bias: (1, num_heads, seq_len, seq_len)."""
        bias = -self.slopes * self.rel_dist[:, :, :seq_len, :seq_len]
        return bias


# --- Attention Block ---

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, pe_type='learned', num_positions=197):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pe_type = pe_type

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # RoPE and ALiBi are applied within attention
        if pe_type == 'rope':
            self.rope = RoPE(num_positions, self.head_dim)
        elif pe_type == 'alibi':
            self.alibi = ALiBi(num_heads, num_positions)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        if self.pe_type == 'rope':
            q, k = self.rope(q, k, N)

        # Priprema maske za ALiBi ako je potreban
        attn_bias = self.alibi.get_bias(N) if self.pe_type == 'alibi' else None

        # KLJUČNA PROMENA: Flash Attention (5-10x brže od ručnog koda)
        if not return_attention:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_bias, 
                dropout_p=0.1 if self.training else 0.0
            )
            attn_weights = None
        else:
            # Fallback na spori metod samo ako nam trebaju težine za analizu
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.pe_type == 'alibi':
                attn = attn + attn_bias
            attn_weights = attn.softmax(dim=-1)
            x = (attn_weights @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        if return_attention:
            return x, attn_weights
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1,
                 pe_type='learned', num_positions=197):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, pe_type, num_positions)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT-Base with configurable positional encoding."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1,
                 pe_type='learned'):
        super().__init__()
        self.pe_type = pe_type
        self.embed_dim = embed_dim
        self.depth = depth
        num_patches = (img_size // patch_size) ** 2  # 196
        num_positions = num_patches + 1  # +1 for CLS

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding (additive types)
        if pe_type == 'learned':
            self.pos_encoding = LearnedPE(num_positions, embed_dim)
        elif pe_type == 'sinusoidal':
            self.pos_encoding = SinusoidalPE(num_positions, embed_dim)
        elif pe_type in ('rope', 'alibi'):
            self.pos_encoding = nn.Identity()  # No additive PE
        else:
            raise ValueError(f"Unknown PE type: {pe_type}")

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, pe_type, num_positions)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 197, 768)
        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        if return_features:
            return x
        return self.head(x[:, 0])  # CLS token

    def forward_with_attention(self, x):
        """Forward pass returning attention weights from all layers."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)

        attentions = []
        for block in self.blocks:
            x, attn = block(x, return_attention=True)
            attentions.append(attn.detach().cpu())

        x = self.norm(x)
        return self.head(x[:, 0]), attentions

    def forward_layer_activations(self, x):
        """Forward pass returning activations after each layer."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)

        activations = []
        for block in self.blocks:
            x = block(x)
            activations.append(x.detach().cpu())

        x = self.norm(x)
        return activations


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def get_imagenet_loaders(data_dir, batch_size=256, num_workers=8):
    """ImageNet-100 data loaders using standard ImageFolder.
    
    Expected structure:
        data_dir/train/n01440764/*.JPEG
        data_dir/val/n01440764/*.JPEG
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=True if num_workers > 0 else False,
                              prefetch_factor=4 if num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True if num_workers > 0 else False,
                            prefetch_factor=4 if num_workers > 0 else None)

    return train_loader, val_loader


# ============================================================================
# 3. TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, config, device, output_dir):
    """Train ViT with DeiT-style recipe. Supports checkpoint-resume."""

    # Performance optimizations
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
    )

    # Cosine schedule with warmup
    total_steps = config['epochs'] * len(train_loader)
    warmup_steps = config['warmup_epochs'] * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))

    # Mixup / CutMix
    use_mixup = config.get('use_mixup', True)
    mixup_alpha = config.get('mixup_alpha', 0.8)
    cutmix_alpha = config.get('cutmix_alpha', 1.0)

    # ========== CHECKPOINT RESUME ==========
    start_epoch = 0
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'lr': [], 'epoch_time': []
    }

    checkpoint_path = os.path.join(output_dir, 'last_checkpoint.pth')
    history_path = os.path.join(output_dir, 'training_history.json')
    final_model_path = os.path.join(output_dir, 'final_model.pth')

    # Skip if already fully trained
    if os.path.exists(final_model_path):
        print(f"  SKIP: final_model.pth already exists. Training is complete.")
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
        return history, best_val_acc

    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"  RESUMING from checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
     
    # Restore random states for reproducibility
        if 'rng_state' in ckpt:
            rng_state = ckpt['rng_state']
            # Ako nije tenzor, pretvaramo ga, a zatim forsiramo uint8 (ByteTensor)
            if not isinstance(rng_state, torch.Tensor):
                rng_state = torch.tensor(rng_state)
            torch.set_rng_state(rng_state.to(torch.uint8).cpu())

        if 'cuda_rng_state' in ckpt and torch.cuda.is_available():
            cuda_rng_state = ckpt['cuda_rng_state']
            if not isinstance(cuda_rng_state, torch.Tensor):
                cuda_rng_state = torch.tensor(cuda_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state.to(torch.uint8).cpu())
        if 'np_rng_state' in ckpt:
            np.random.set_state(ckpt['np_rng_state'])
        print(f"  Resumed at epoch {start_epoch} (best_val_acc={best_val_acc:.2f}%)")
    else:
        print(f"  Starting training from scratch.")

    # ========== TRAINING LOOP ==========
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Pratimo da li je ovaj konkretni batch pomešan (Mixup)
            did_mixup = False

            with torch.amp.autocast('cuda'):
                if use_mixup and np.random.rand() < 0.5:
                    did_mixup = True
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    idx = torch.randperm(images.size(0), device=device)
                    # Mixup radimo PRE modela (pravilno)
                    mixed_images = lam * images + (1 - lam) * images[idx]
                    outputs = model(mixed_images)
                    loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[idx])
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            # --- Optimizacija ---
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # --- Metrike (FIX) ---
            running_loss += loss.item()
            
            # Računamo accuracy samo ako batch NIJE bio mixup-ovan
            # tako ćemo imati realnu sliku treninga
            if not did_mixup:
                with torch.no_grad():
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{running_loss/(batch_idx+1):.3f}",
                             acc=f"{100.*correct/max(total, 1):.2f}%", # Dodajemo Acc u progress bar
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / max(total, 1)

        # --- Validate ---
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.get_last_lr()[0])
        history['epoch_time'].append(epoch_time)

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, time={epoch_time:.0f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

        # Save resumable checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'rng_state': torch.get_rng_state(),
                'np_rng_state': np.random.get_state(),
            }
            if torch.cuda.is_available():
                ckpt['cuda_rng_state'] = torch.cuda.get_rng_state()
           

            # --- Prvo snimimo sve lokalno na SSD ---
            torch.save(ckpt, checkpoint_path)
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"   Checkpoint saved locally at epoch {epoch+1}")

            # 2. ODMAH ZATIM radimo AUTOMATSKI BACKUP NA DRIVE
            import shutil

            # BACKUP NA DRIVE (unutar petlje)
            try:
                import shutil
                drive_path = os.path.join("/content/drive/MyDrive/re_experiment/results", os.path.basename(output_dir))
                os.makedirs(drive_path, exist_ok=True)
                # Kopiramo kritične fajlove (checkpoint i istoriju)
                shutil.copy2(checkpoint_path, os.path.join(drive_path, 'last_checkpoint.pth'))
                shutil.copy2(history_path, os.path.join(drive_path, 'training_history.json'))
                print(f"[Drive Backup] Epoch {epoch+1} synced.")
            except Exception as e:
                print(f"[Drive Backup] Error: {e}")
           
        # Save snapshot every 10 epochs (for MI evolution analysis)
        if (epoch + 1) % 10 == 0:
            snapshot_name = f'checkpoint_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), os.path.join(output_dir, snapshot_name))
            # Opciono: kopiraj i snapshot na Drive odmah
            try:
                shutil.copy2(os.path.join(output_dir, snapshot_name), os.path.join(drive_path, snapshot_name))
            except: pass

    # ========================================================
    # KRAJ PETLJE (Ovde izlaziš iz 'for epoch in range' bloka)
    # ========================================================

    # --- 3. FINALNO SNIMANJE I SINHRONIZACIJA ---
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Finalni rsync ili cp da bi bio 100% siguran da je sve na Drive-u
    try:
        final_drive_target = os.path.join("/content/drive/MyDrive/re_experiment/results", os.path.basename(output_dir))
        os.system(f"cp -r {output_dir}/* {final_drive_target}/")
        print(f"\n[FINAL] Sve je uspešno prebačeno na: {final_drive_target}")
    except:
        print("\n[WARNING] Finalno kopiranje nije uspelo, proveri Drive prostor.")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")
    return history, best_val_acc


@torch.inference_mode() # Brže od no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in val_loader:
        # non_blocking=True omogućava brži prenos na GPU
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Flash Attention će se automatski koristiti i ovde
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100.0 * correct / total


# ============================================================================
# 4. ANALYSIS FUNCTIONS
# ============================================================================

@torch.no_grad()
def extract_positional_embedding(model, pe_type):
    """Extract PE matrix (num_positions, embed_dim) from model.

    For RoPE: construct an effective PE matrix by applying rotary encoding
    to an identity-like input, producing a (num_positions, embed_dim) matrix
    that captures the positional signal injected into Q/K.

    For ALiBi: construct an effective PE matrix from the attention bias,
    representing how each position biases attention scores.
    The matrix is (num_positions, num_positions * num_heads) where each row
    is the flattened bias pattern for that position across all heads.
    For embedding-level analyses we reshape to (num_positions, embed_dim)
    by repeating/truncating as needed.
    """
    if pe_type == 'learned':
        return model.pos_encoding.pos_embed.squeeze(0).cpu().numpy()
    elif pe_type == 'sinusoidal':
        return model.pos_encoding.pe.squeeze(0).cpu().numpy()
    elif pe_type == 'rope':
        return _extract_rope_matrix(model)
    elif pe_type == 'alibi':
        return _extract_alibi_matrix(model)
    return None


def _extract_rope_matrix(model):
    """Construct effective RoPE positional matrix.

    For each position, we apply the RoPE rotation to a unit vector
    and concatenate across all heads, giving a (num_positions, embed_dim)
    matrix that captures the full rotary positional signal.
    """
    # Get RoPE from first layer (all layers share same frequencies)
    rope = model.blocks[0].attn.rope
    num_heads = model.blocks[0].attn.num_heads
    head_dim = model.blocks[0].attn.head_dim

    cos = rope.cos_cached.squeeze(0).squeeze(0).cpu().numpy()  # (N, head_dim/2)
    sin = rope.sin_cached.squeeze(0).squeeze(0).cpu().numpy()

    num_positions = cos.shape[0]

    # Build effective PE: for each position, the rotation applied to a unit
    # query/key vector. We interleave cos and sin to get (N, head_dim) per head,
    # then concatenate across heads.
    pe_per_head = np.zeros((num_positions, head_dim))
    for pos in range(num_positions):
        # The rotation of a unit vector [1,0,1,0,...] gives [cos, sin, cos, sin,...]
        for i in range(head_dim // 2):
            pe_per_head[pos, 2*i] = cos[pos, i]
            pe_per_head[pos, 2*i + 1] = sin[pos, i]

    # Tile across heads to get (N, embed_dim)
    pe_matrix = np.tile(pe_per_head, (1, num_heads))  # (N, head_dim * num_heads = embed_dim)

    # Add CLS row (position 0 is already included)
    return pe_matrix


def _extract_alibi_matrix(model):
    """Construct effective ALiBi positional matrix.

    ALiBi encodes position through attention biases, not embeddings.
    We construct a (num_positions, embed_dim) matrix where each position's
    representation is derived from its bias pattern across all heads.

    For each head h and position i, ALiBi adds bias[i,j] = -slope_h * |i-j|
    to attention scores. We take the bias row for position i (across all j)
    for each head and concatenate, then project to embed_dim via PCA-like
    tiling to enable comparable analyses.
    """
    alibi = model.blocks[0].attn.alibi
    num_heads = model.blocks[0].attn.num_heads
    embed_dim = model.embed_dim

    slopes = alibi.slopes.squeeze().cpu().numpy()  # (num_heads,)
    rel_dist = alibi.rel_dist.squeeze(0).squeeze(0).cpu().numpy()  # (N, N)
    num_positions = rel_dist.shape[0]

    # Bias matrix per head: (num_heads, N, N)
    # For position i, take row i from each head's bias matrix
    # This gives us: for each position, how it biases attention to all other positions
    pe_matrix = np.zeros((num_positions, num_heads * num_positions))
    for h in range(num_heads):
        bias_h = -slopes[h] * rel_dist  # (N, N)
        pe_matrix[:, h*num_positions:(h+1)*num_positions] = bias_h

    # Project to embed_dim for comparable analyses
    # Use SVD to get the most informative embed_dim dimensions
    if pe_matrix.shape[1] > embed_dim:
        from sklearn.decomposition import PCA
        # n_components can't exceed min(n_samples, n_features)
        max_components = min(embed_dim, pe_matrix.shape[0], pe_matrix.shape[1])
        pca = PCA(n_components=max_components)
        pe_matrix = pca.fit_transform(pe_matrix)
        if max_components < embed_dim:
            # Pad remaining dimensions with zeros
            pad = np.zeros((pe_matrix.shape[0], embed_dim - max_components))
            pe_matrix = np.concatenate([pe_matrix, pad], axis=1)
    elif pe_matrix.shape[1] < embed_dim:
        # Pad with zeros
        pad = np.zeros((num_positions, embed_dim - pe_matrix.shape[1]))
        pe_matrix = np.concatenate([pe_matrix, pad], axis=1)

    return pe_matrix


def compute_cosine_similarity(pe_matrix):
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(pe_matrix, axis=1, keepdims=True)
    return (pe_matrix @ pe_matrix.T) / (norms @ norms.T + 1e-8)


def compute_dimension_entropy(pe_matrix, n_bins=64):
    """Compute Shannon entropy per dimension."""
    from scipy.stats import entropy as sp_entropy
    entropies = []
    for d in range(pe_matrix.shape[1]):
        vals = pe_matrix[:, d]
        hist, _ = np.histogram(vals, bins=n_bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        entropies.append(sp_entropy(hist, base=2))
    return np.array(entropies)


def compute_dimension_variance(pe_matrix):
    """Compute variance per dimension across positions."""
    return np.var(pe_matrix[1:], axis=0)  # Exclude CLS


def pca_projection(pe_matrix, n_components=2):
    """PCA projection of patch positions (excluding CLS)."""
    from sklearn.decomposition import PCA
    patches = pe_matrix[1:]  # Exclude CLS
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(patches)
    return proj, pca.explained_variance_ratio_.sum()


def tsne_projection(pe_matrix, n_components=2, perplexity=30):
    """t-SNE projection of patch positions."""
    from sklearn.manifold import TSNE
    patches = pe_matrix[1:]
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(patches)


@torch.no_grad()
def compute_mi_per_layer(model, val_loader, device, n_batches=50):
    """Compute mutual information between position and attention per layer.

    Returns MI in bits (not nats).
    """
    from scipy.stats import entropy as sp_entropy

    model.eval()
    num_layers = model.depth
    # Accumulate attention distributions per position per layer
    # attention shape: (B, num_heads, N, N) -> average over heads
    mi_per_layer = [[] for _ in range(num_layers)]

    for batch_idx, (images, _) in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        images = images.to(device)
        _, attentions = model.forward_with_attention(images)

        for layer_idx, attn in enumerate(attentions):
            # attn: (B, num_heads, N, N)
            # Average over heads, then compute MI(position, attention_distribution)
            avg_attn = attn.mean(dim=1).numpy()  # (B, N, N)
            mi_per_layer[layer_idx].append(avg_attn)

    # Compute MI for each layer
    mi_values = []
    for layer_idx in range(num_layers):
        all_attn = np.concatenate(mi_per_layer[layer_idx], axis=0)  # (total_B, N, N)
        N = all_attn.shape[1]

        # MI(position, attention) via discretization
        n_bins = 32
        mi_sum = 0.0
        for pos in range(N):
            attn_at_pos = all_attn[:, pos, :]  # (total_B, N) - attention FROM position pos
            # Discretize each attention target
            for target in range(N):
                vals = attn_at_pos[:, target]
                hist, _ = np.histogram(vals, bins=n_bins, density=True)
                hist = hist[hist > 0]; hist = hist / hist.sum()
                mi_sum += sp_entropy(hist, base=2)

        # Normalize
        marginal_ent = mi_sum / (N * N)
        mi_values.append(marginal_ent)

    return mi_values


@torch.no_grad()
def compute_attention_entropy_per_layer(model, val_loader, device, n_batches=50):
    """Compute entropy of attention distributions per layer."""
    model.eval()
    num_layers = model.depth
    ent_per_layer = [0.0] * num_layers
    count = 0

    for batch_idx, (images, _) in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        images = images.to(device)
        _, attentions = model.forward_with_attention(images)

        for layer_idx, attn in enumerate(attentions):
            # attn: (B, num_heads, N, N)
            # Entropy of each attention distribution
            attn_np = attn.numpy()
            attn_np = np.clip(attn_np, 1e-10, 1.0)
            ent = -(attn_np * np.log2(attn_np)).sum(axis=-1).mean()
            ent_per_layer[layer_idx] += ent

        count += 1

    return [e / count for e in ent_per_layer]


@torch.no_grad()
def compute_layer_entropy(model, val_loader, device, n_batches=50):
    """Compute activation entropy per layer (all tokens + CLS token)."""
    model.eval()
    num_layers = model.depth
    all_ent = [0.0] * num_layers
    cls_ent = [0.0] * num_layers
    count = 0

    for batch_idx, (images, _) in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        images = images.to(device)
        activations = model.forward_layer_activations(images)

        for layer_idx, act in enumerate(activations):
            act_np = act.numpy()  # (B, N, D)
            # Discretize and compute entropy
            n_bins = 64
            # All tokens
            flat = act_np.reshape(-1, act_np.shape[-1])  # (B*N, D)
            dim_ent = []
            for d in range(flat.shape[1]):
                hist, _ = np.histogram(flat[:, d], bins=n_bins, density=True)
                hist = hist[hist > 0]; hist = hist / hist.sum()
                from scipy.stats import entropy as sp_entropy
                dim_ent.append(sp_entropy(hist, base=2))
            all_ent[layer_idx] += np.mean(dim_ent)

            # CLS token only
            cls_flat = act_np[:, 0, :]  # (B, D)
            cls_dim_ent = []
            for d in range(cls_flat.shape[1]):
                hist, _ = np.histogram(cls_flat[:, d], bins=n_bins, density=True)
                hist = hist[hist > 0]; hist = hist / hist.sum()
                cls_dim_ent.append(sp_entropy(hist, base=2))
            cls_ent[layer_idx] += np.mean(cls_dim_ent)

        count += 1

    return [e / count for e in all_ent], [e / count for e in cls_ent]


def probe_analysis(pe_matrix, num_patches_per_side=14):
    """Train linear probes to predict row, column, exact position from PE."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    patches = pe_matrix[1:]  # Exclude CLS, shape (196, 768)
    num_patches = patches.shape[0]

    positions = np.arange(num_patches)
    rows = positions // num_patches_per_side
    cols = positions % num_patches_per_side

    results = {}
    for task_name, labels in [('row', rows), ('column', cols), ('position', positions)]:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        # Use min(5, smallest_class) to avoid cv error
        min_class_count = min(np.bincount(labels))
        n_splits = min(5, min_class_count)
        if n_splits < 2:
            # Too few samples per class for CV, train on all and report training acc
            clf.fit(patches, labels)
            acc = clf.score(patches, labels)
            results[task_name] = {'mean': float(acc * 100), 'std': 0.0}
            print(f"    Probe {task_name}: {acc*100:.1f}% (no CV, n_per_class={min_class_count})")
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, patches, labels, cv=cv, scoring='accuracy')
            results[task_name] = {
                'mean': float(scores.mean() * 100),
                'std': float(scores.std() * 100),
            }
            print(f"    Probe {task_name}: {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%")

    return results


@torch.no_grad()
def noise_ablation(model, val_loader, device, pe_type):
    """Noise ablation experiment: add increasing noise to PE.

    For RoPE: perturb the cached sin/cos rotation matrices.
    For ALiBi: perturb the slope parameters.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    results = {'noise_levels': noise_levels, 'accuracies': []}

    # Get PE std for noise scaling
    pe = extract_positional_embedding(model, pe_type)
    pe_std = pe.std() if pe is not None else 1.0

    for noise_level in noise_levels:
        originals = {}

        if pe_type == 'learned':
            originals['pe'] = model.pos_encoding.pos_embed.data.clone()
            noise = torch.randn_like(originals['pe']) * pe_std * noise_level
            model.pos_encoding.pos_embed.data = originals['pe'] + noise

        elif pe_type == 'sinusoidal':
            originals['pe'] = model.pos_encoding.pe.data.clone()
            noise = torch.randn_like(originals['pe']) * pe_std * noise_level
            model.pos_encoding.pe.data = originals['pe'] + noise

        elif pe_type == 'rope':
            # Perturb RoPE sin/cos in ALL layers
            for i, block in enumerate(model.blocks):
                rope = block.attn.rope
                originals[f'cos_{i}'] = rope.cos_cached.data.clone()
                originals[f'sin_{i}'] = rope.sin_cached.data.clone()
                cos_std = rope.cos_cached.data.std()
                rope.cos_cached.data += torch.randn_like(rope.cos_cached.data) * cos_std * noise_level
                rope.sin_cached.data += torch.randn_like(rope.sin_cached.data) * cos_std * noise_level

        elif pe_type == 'alibi':
            # Perturb ALiBi slopes in ALL layers
            for i, block in enumerate(model.blocks):
                alibi = block.attn.alibi
                originals[f'slopes_{i}'] = alibi.slopes.data.clone()
                slope_std = alibi.slopes.data.std()
                alibi.slopes.data += torch.randn_like(alibi.slopes.data) * slope_std * noise_level

        _, acc = evaluate(model, val_loader, criterion, device)
        results['accuracies'].append(acc)
        print(f"    Noise {noise_level:.1f}x sigma_PE: accuracy = {acc:.2f}%")

        # Restore originals
        if pe_type == 'learned':
            model.pos_encoding.pos_embed.data = originals['pe']
        elif pe_type == 'sinusoidal':
            model.pos_encoding.pe.data = originals['pe']
        elif pe_type == 'rope':
            for i, block in enumerate(model.blocks):
                block.attn.rope.cos_cached.data = originals[f'cos_{i}']
                block.attn.rope.sin_cached.data = originals[f'sin_{i}']
        elif pe_type == 'alibi':
            for i, block in enumerate(model.blocks):
                block.attn.alibi.slopes.data = originals[f'slopes_{i}']

    # Test without PE
    originals = {}
    if pe_type == 'learned':
        originals['pe'] = model.pos_encoding.pos_embed.data.clone()
        model.pos_encoding.pos_embed.data.zero_()
    elif pe_type == 'sinusoidal':
        originals['pe'] = model.pos_encoding.pe.data.clone()
        model.pos_encoding.pe.data.zero_()
    elif pe_type == 'rope':
        # Zero out all rotations (set cos=1, sin=0 → identity rotation)
        for i, block in enumerate(model.blocks):
            rope = block.attn.rope
            originals[f'cos_{i}'] = rope.cos_cached.data.clone()
            originals[f'sin_{i}'] = rope.sin_cached.data.clone()
            rope.cos_cached.data.fill_(1.0)
            rope.sin_cached.data.fill_(0.0)
    elif pe_type == 'alibi':
        # Zero out all slopes (no positional bias)
        for i, block in enumerate(model.blocks):
            alibi = block.attn.alibi
            originals[f'slopes_{i}'] = alibi.slopes.data.clone()
            alibi.slopes.data.zero_()

    _, acc_no_pe = evaluate(model, val_loader, criterion, device)
    results['accuracy_no_pe'] = acc_no_pe
    print(f"    Without PE: accuracy = {acc_no_pe:.2f}%")

    # Restore
    if pe_type == 'learned':
        model.pos_encoding.pos_embed.data = originals['pe']
    elif pe_type == 'sinusoidal':
        model.pos_encoding.pe.data = originals['pe']
    elif pe_type == 'rope':
        for i, block in enumerate(model.blocks):
            block.attn.rope.cos_cached.data = originals[f'cos_{i}']
            block.attn.rope.sin_cached.data = originals[f'sin_{i}']
    elif pe_type == 'alibi':
        for i, block in enumerate(model.blocks):
            block.attn.alibi.slopes.data = originals[f'slopes_{i}']

    return results


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_training_curves(all_histories, output_dir):
    """Plot training curves for all PE types (averaged over seeds)."""
    import matplotlib.pyplot as plt

    COLOR_MAP = {
        'learned': '#7B68EE',
        'sinusoidal': '#00CED1',
        'rope': '#FF6347',
        'alibi': '#32CD32',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for pe_type, runs in all_histories.items():
        color = COLOR_MAP.get(pe_type, '#888888')
         # Average over seeds — pad shorter runs to max length
        max_epochs = max(len(r['val_acc']) for r in runs)
        min_epochs = min(len(r['val_acc']) for r in runs)
        # Use min_epochs so all runs have same length
        val_accs = np.array([r['val_acc'][:min_epochs] for r in runs])
        val_losses = np.array([r['val_loss'][:min_epochs] for r in runs])
        epochs = np.arange(1, min_epochs + 1)
        
        mean_acc = val_accs.mean(axis=0)
        std_acc = val_accs.std(axis=0)
        mean_loss = val_losses.mean(axis=0)
        std_loss = val_losses.std(axis=0)

        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"

        ax1.plot(epochs, mean_acc, color=color, linewidth=2, label=label)
        ax1.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.15)

        ax2.plot(epochs, mean_loss, color=color, linewidth=2, label=label)
        ax2.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.15)

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation accuracy (%)')
    ax1.set_title('Accuracy during training \u2014 ImageNet-1K (ViT-Base)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation loss')
    ax2.set_title('Validation loss during training \u2014 ImageNet-1K (ViT-Base)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_training_comparison.png'), dpi=200)
    plt.close()


def plot_cosine_similarity(all_pe_matrices, output_dir):
    """Plot cosine similarity matrices for all PE types."""
    import matplotlib.pyplot as plt

    pe_types_with_matrix = {k: v for k, v in all_pe_matrices.items() if v is not None}
    n = len(pe_types_with_matrix)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (pe_type, pe_matrix) in zip(axes, pe_types_with_matrix.items()):
        cos_sim = compute_cosine_similarity(pe_matrix)
        im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"
        ax.set_title(f'{label} \u2014 ImageNet-1K')
        ax.set_xlabel('Position'); ax.set_ylabel('Position')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_cosine_similarity.png'), dpi=200)
    plt.close()


def plot_pca_tsne(all_pe_matrices, output_dir):
    """Plot PCA and t-SNE projections."""
    import matplotlib.pyplot as plt

    pe_types_with_matrix = {k: v for k, v in all_pe_matrices.items() if v is not None}
    n = len(pe_types_with_matrix)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    positions = np.arange(196)  # 14x14 patches

    for col, (pe_type, pe_matrix) in enumerate(pe_types_with_matrix.items()):
        proj_pca, explained_var = pca_projection(pe_matrix)
        proj_tsne = tsne_projection(pe_matrix)
        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"

        sc1 = axes[0, col].scatter(proj_pca[:, 0], proj_pca[:, 1], c=positions,
                                    cmap='viridis', s=30, edgecolors='k', linewidth=0.3)
        axes[0, col].set_title(f'{label} \u2014 PCA ({explained_var*100:.1f}% var.)')
        axes[0, col].set_xlabel('PC1'); axes[0, col].set_ylabel('PC2')
        plt.colorbar(sc1, ax=axes[0, col], label='Position')

        sc2 = axes[1, col].scatter(proj_tsne[:, 0], proj_tsne[:, 1], c=positions,
                                    cmap='viridis', s=30, edgecolors='k', linewidth=0.3)
        axes[1, col].set_title(f'{label} \u2014 t-SNE')
        axes[1, col].set_xlabel('t-SNE 1'); axes[1, col].set_ylabel('t-SNE 2')
        plt.colorbar(sc2, ax=axes[1, col], label='Position')

    fig.suptitle('PCA and t-SNE Projections \u2014 ImageNet-1K (ViT-Base)', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_pca_tsne.png'), dpi=200)
    plt.close()


def plot_dimension_entropy(all_pe_matrices, output_dir):
    """Plot entropy per dimension."""
    import matplotlib.pyplot as plt

    COLOR_MAP = {'learned': '#7B68EE', 'sinusoidal': '#00CED1', 'rope': '#FF6347', 'alibi': '#32CD32'}

    pe_types_with_matrix = {k: v for k, v in all_pe_matrices.items() if v is not None}
    if not pe_types_with_matrix:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    dims = np.arange(pe_types_with_matrix[list(pe_types_with_matrix.keys())[0]].shape[1])

    for pe_type, pe_matrix in pe_types_with_matrix.items():
        ent = compute_dimension_entropy(pe_matrix[1:])  # Exclude CLS
        color = COLOR_MAP.get(pe_type, '#888')
        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"
        ax1.plot(dims, ent, color=color, alpha=0.7, linewidth=0.8,
                label=f'{label} (mean: {ent.mean():.2f})')
        ax2.hist(ent, bins=30, alpha=0.5, color=color, label=label)

    ax1.set_xlabel('Dimension'); ax1.set_ylabel('Shannon entropy (bit)')
    ax1.set_title('Entropy per embedding dimension \u2014 ImageNet-1K')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Shannon entropy (bit)'); ax2.set_ylabel('Number of dimensions')
    ax2.set_title('Entropy distribution \u2014 ImageNet-1K')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_dimension_entropy.png'), dpi=200)
    plt.close()


def plot_variance_per_dim(all_pe_matrices, output_dir):
    """Plot variance per dimension."""
    import matplotlib.pyplot as plt

    COLOR_MAP = {'learned': '#7B68EE', 'sinusoidal': '#00CED1', 'rope': '#FF6347', 'alibi': '#32CD32'}

    pe_types_with_matrix = {k: v for k, v in all_pe_matrices.items() if v is not None}
    if not pe_types_with_matrix:
        return

    n = len(pe_types_with_matrix)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (pe_type, pe_matrix) in zip(axes, pe_types_with_matrix.items()):
        var = compute_dimension_variance(pe_matrix)
        color = COLOR_MAP.get(pe_type, '#888')
        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"
        ax.bar(np.arange(len(var)), var, color=color, alpha=0.7, width=1.0)
        ax.set_xlabel('Dimension'); ax.set_ylabel('Variance')
        ax.set_title(f'{label} variance per dimension \u2014 ImageNet-1K')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_variance_per_dim.png'), dpi=200)
    plt.close()


def plot_mi_per_layer(all_mi, output_dir):
    """Plot MI per layer for all PE types."""
    import matplotlib.pyplot as plt

    COLOR_MAP = {'learned': '#7B68EE', 'sinusoidal': '#00CED1', 'rope': '#FF6347', 'alibi': '#32CD32'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    layers = list(range(1, 13))

    for pe_type, mi_data in all_mi.items():
        color = COLOR_MAP.get(pe_type, '#888')
        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"

        # MI - average over seeds
        mi_vals = np.array(mi_data['mi'])  # shape (n_seeds, 12)
        mean_mi = mi_vals.mean(axis=0)
        std_mi = mi_vals.std(axis=0)

        ax1.plot(layers, mean_mi, 'o-', color=color, linewidth=2, markersize=6, label=label)
        ax1.fill_between(layers, mean_mi - std_mi, mean_mi + std_mi, color=color, alpha=0.15)

        # Attention entropy
        att_vals = np.array(mi_data['attn_ent'])
        mean_att = att_vals.mean(axis=0)
        std_att = att_vals.std(axis=0)

        ax2.plot(layers, mean_att, 'o-', color=color, linewidth=2, markersize=6, label=label)
        ax2.fill_between(layers, mean_att - std_att, mean_att + std_att, color=color, alpha=0.15)

    ax1.set_xlabel('Layer'); ax1.set_ylabel('Mutual information (bit)')
    ax1.set_title('MI(position, attention) per layer \u2014 ImageNet-1K')
    ax1.legend(); ax1.set_xticks(layers); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Layer'); ax2.set_ylabel('Shannon entropy (bit)')
    ax2.set_title('Attention distribution entropy per layer \u2014 ImageNet-1K')
    ax2.legend(); ax2.set_xticks(layers); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_mutual_information.png'), dpi=200)
    plt.close()


def plot_noise_ablation(all_ablation, output_dir):
    """Plot noise ablation results."""
    import matplotlib.pyplot as plt

    COLOR_MAP = {'learned': '#7B68EE', 'sinusoidal': '#00CED1', 'rope': '#FF6347', 'alibi': '#32CD32'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for pe_type, data in all_ablation.items():
        color = COLOR_MAP.get(pe_type, '#888')
        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"

        noise = data['noise_levels']
        accs = np.array(data['accuracies'])  # (n_seeds, n_noise_levels)
        mean_acc = accs.mean(axis=0)
        std_acc = accs.std(axis=0)

        # Filter out None values
        valid = [i for i, v in enumerate(mean_acc) if not np.isnan(v)]
        ax1.plot([noise[i] for i in valid], [mean_acc[i] for i in valid],
                'o-', color=color, linewidth=2, markersize=6, label=label)

    ax1.axhline(y=0.1, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Chance (0.1%)')
    ax1.set_xlabel('Noise level (\u00D7 \u03C3_PE)')
    ax1.set_ylabel('Test accuracy (%)')
    ax1.set_title('Noise robustness \u2014 ImageNet-1K')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Bar chart: with PE vs without PE
    pe_types_with_removal = {k: v for k, v in all_ablation.items()
                              if v.get('acc_no_pe') and len(v['acc_no_pe']) > 0
                              and v['acc_no_pe'][0] is not None}
    if pe_types_with_removal:
        x = np.arange(len(pe_types_with_removal))
        w = 0.35
        with_pe = [v['accuracies'][0].mean() if isinstance(v['accuracies'][0], np.ndarray)
                   else np.mean(v['accuracies'][0]) for v in pe_types_with_removal.values()]
        without_pe = [np.mean(v['acc_no_pe']) for v in pe_types_with_removal.values()]
        labels = [k.upper() if k in ('rope', 'alibi') else f"{k.capitalize()} PE"
                 for k in pe_types_with_removal.keys()]
        colors = [COLOR_MAP.get(k, '#888') for k in pe_types_with_removal.keys()]

        for i, (wp, wop, c, l) in enumerate(zip(with_pe, without_pe, colors, labels)):
            ax2.bar(i - 0.15, wp, 0.3, color=c, alpha=0.9, label=f'{l} (with)')
            ax2.bar(i + 0.15, wop, 0.3, color=c, alpha=0.4, label=f'{l} (without)')
            delta = wp - wop
            ax2.annotate(f'\u0394={delta:.1f}%', xy=(i, wop + 1), fontsize=9, color='red',
                        fontweight='bold', ha='center')

        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Test accuracy (%)')
        ax2.set_title('Effect of PE removal \u2014 ImageNet-1K')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_noise_ablation.png'), dpi=200)
    plt.close()


def plot_probe_analysis(all_probe, output_dir):
    """Plot probe analysis results."""
    import matplotlib.pyplot as plt

    COLOR_MAP = {'learned': '#7B68EE', 'sinusoidal': '#00CED1', 'rope': '#FF6347', 'alibi': '#32CD32'}

    pe_types_with_probe = {k: v for k, v in all_probe.items() if v is not None}
    if not pe_types_with_probe:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    tasks = ['Row', 'Column', 'Exact position']
    x = np.arange(len(tasks))
    n = len(pe_types_with_probe)
    width = 0.8 / n

    for i, (pe_type, data) in enumerate(pe_types_with_probe.items()):
        color = COLOR_MAP.get(pe_type, '#888')
        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"
        vals = [data['row']['mean'], data['column']['mean'], data['position']['mean']]
        errs = [data['row']['std'], data['column']['std'], data['position']['std']]
        bars = ax.bar(x + i * width - (n-1)*width/2, vals, width, yerr=errs,
                     color=color, alpha=0.8, label=label, capsize=3)

    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel('Probe accuracy (%)'); ax.set_ylim(0, 110)
    ax.set_title('Probe analysis \u2014 ImageNet-1K (ViT-Base)')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_probe_analysis.png'), dpi=200)
    plt.close()


def plot_layer_entropy(all_layer_ent, output_dir):
    """Plot activation entropy through layers."""
    import matplotlib.pyplot as plt

    COLOR_MAP = {'learned': '#7B68EE', 'sinusoidal': '#00CED1', 'rope': '#FF6347', 'alibi': '#32CD32'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    layers = list(range(1, 13))

    for pe_type, data in all_layer_ent.items():
        color = COLOR_MAP.get(pe_type, '#888')
        label = pe_type.upper() if pe_type in ('rope', 'alibi') else f"{pe_type.capitalize()} PE"

        all_e = np.array(data['all_tokens'])
        cls_e = np.array(data['cls_token'])

        ax1.plot(layers, all_e.mean(axis=0), 'o-', color=color, linewidth=2, markersize=6, label=label)
        ax1.fill_between(layers, all_e.mean(0)-all_e.std(0), all_e.mean(0)+all_e.std(0), color=color, alpha=0.15)

        ax2.plot(layers, cls_e.mean(axis=0), 'o-', color=color, linewidth=2, markersize=6, label=label)
        ax2.fill_between(layers, cls_e.mean(0)-cls_e.std(0), cls_e.mean(0)+cls_e.std(0), color=color, alpha=0.15)

    ax1.set_xlabel('Layer'); ax1.set_ylabel('Mean entropy (bit)')
    ax1.set_title('Activation entropy across layers\n(ImageNet-1K \u2014 all tokens)')
    ax1.legend(); ax1.set_xticks(layers); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Layer'); ax2.set_ylabel('CLS token entropy (bit)')
    ax2.set_title('CLS token entropy across layers\n(ImageNet-1K \u2014 information bottleneck)')
    ax2.legend(); ax2.set_xticks(layers); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_layer_entropy.png'), dpi=200)
    plt.close()


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Full-scale ViT PE experiment')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet-100')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'analyze', 'all'])
    parser.add_argument('--pe_type', type=str, default=None,
                       choices=['learned', 'sinusoidal', 'rope', 'alibi'],
                       help='Single PE type (default: all)')
    parser.add_argument('--seed', type=int, default=None, help='Single seed (default: all)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes')
    args = parser.parse_args()

    # Configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': args.num_classes,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'lr': 3e-4,
        'weight_decay': 0.1,
        'warmup_epochs': 20,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'label_smoothing': 0.1,
        'use_mixup': True,
        'mixup_alpha': 0.8,
    }

    pe_types = [args.pe_type] if args.pe_type else ['learned', 'sinusoidal', 'rope', 'alibi']
    seeds = [args.seed] if args.seed else [42, 123, 456]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PE types: {pe_types}")
    print(f"Seeds: {seeds}")
    print(f"Config: {json.dumps(config, indent=2)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Data
    print(f"\nLoading ImageNet-100...")
    train_loader, val_loader = get_imagenet_loaders(
        args.data_dir, batch_size=config['batch_size'], num_workers=args.num_workers)
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val: {len(val_loader.dataset)} images")

    # ========== TRAINING ==========
    if args.mode in ('train', 'all'):
        print("\n" + "="*60)
        print("TRAINING PHASE")
        print("="*60)

        for pe_type in pe_types:
            for seed in seeds:
                run_dir = os.path.join(args.output_dir, f'{pe_type}_seed{seed}')
                os.makedirs(run_dir, exist_ok=True)

                print(f"\n--- Training: {pe_type} PE, seed={seed} ---")

                # Set seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                torch.cuda.manual_seed_all(seed)

                # Create model
                model = VisionTransformer(
                    img_size=config['img_size'], patch_size=config['patch_size'],
                    num_classes=config['num_classes'], embed_dim=config['embed_dim'],
                    depth=config['depth'], num_heads=config['num_heads'],
                    mlp_ratio=config['mlp_ratio'], dropout=config['dropout'],
                    pe_type=pe_type
                ).to(device)

                n_params = sum(p.numel() for p in model.parameters())
                print(f"  Model parameters: {n_params/1e6:.1f}M")

                # Compile model for faster training
                if hasattr(torch, 'compile'):
                    print("  Compiling model with torch.compile...")
                    model = torch.compile(model)

                # Train
                history, best_acc = train_model(model, train_loader, val_loader,
                                                config, device, run_dir)

    # ========== ANALYSIS ==========
    if args.mode in ('analyze', 'all'):
        print("\n" + "="*60)
        print("ANALYSIS PHASE")
        print("="*60)

        # Collect results across all runs
        all_histories = {}
        all_pe_matrices = {}
        all_mi = {}
        all_ablation = {}
        all_probe = {}
        all_layer_ent = {}

        for pe_type in pe_types:
            all_histories[pe_type] = []
            all_mi[pe_type] = {'mi': [], 'attn_ent': []}
            all_ablation[pe_type] = {'noise_levels': [], 'accuracies': [], 'acc_no_pe': []}
            all_layer_ent[pe_type] = {'all_tokens': [], 'cls_token': []}

            for seed in seeds:
                run_dir = os.path.join(args.output_dir, f'{pe_type}_seed{seed}')

                # Load history
                hist_path = os.path.join(run_dir, 'training_history.json')
                if os.path.exists(hist_path):
                    with open(hist_path) as f:
                        all_histories[pe_type].append(json.load(f))

                # Load best model
                model_path = os.path.join(run_dir, 'best_model.pth')
                if not os.path.exists(model_path):
                    print(f"  Skipping {pe_type} seed={seed} (no model found)")
                    continue

                print(f"\n--- Analyzing: {pe_type} PE, seed={seed} ---")

                torch.manual_seed(seed)
                model = VisionTransformer(
                    img_size=config['img_size'], patch_size=config['patch_size'],
                    num_classes=config['num_classes'], embed_dim=config['embed_dim'],
                    depth=config['depth'], num_heads=config['num_heads'],
                    mlp_ratio=config['mlp_ratio'], dropout=config['dropout'],
                    pe_type=pe_type
                ).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))

                # 1. PE matrix
                pe_matrix = extract_positional_embedding(model, pe_type)
                if pe_matrix is not None and pe_type not in all_pe_matrices:
                    all_pe_matrices[pe_type] = pe_matrix

                # 2. Probe analysis (first seed only)
                if pe_matrix is not None and pe_type not in all_probe:
                    print("  Running probe analysis...")
                    all_probe[pe_type] = probe_analysis(pe_matrix, num_patches_per_side=14)

                # 3. MI per layer
                print("  Computing MI per layer...")
                mi = compute_mi_per_layer(model, val_loader, device)
                all_mi[pe_type]['mi'].append(mi)

                # 4. Attention entropy
                print("  Computing attention entropy...")
                attn_ent = compute_attention_entropy_per_layer(model, val_loader, device)
                all_mi[pe_type]['attn_ent'].append(attn_ent)

                # 5. Noise ablation
                print("  Running noise ablation...")
                ablation = noise_ablation(model, val_loader, device, pe_type)
                all_ablation[pe_type]['noise_levels'] = ablation['noise_levels']
                all_ablation[pe_type]['accuracies'].append(
                    [a if a is not None else float('nan') for a in ablation['accuracies']])
                all_ablation[pe_type]['acc_no_pe'].append(ablation['accuracy_no_pe'])

                # 6. Layer entropy
                print("  Computing layer entropy...")
                all_e, cls_e = compute_layer_entropy(model, val_loader, device)
                all_layer_ent[pe_type]['all_tokens'].append(all_e)
                all_layer_ent[pe_type]['cls_token'].append(cls_e)

        # ========== GENERATE FIGURES ==========
        print("\n" + "="*60)
        print("GENERATING FIGURES")
        print("="*60)

        fig_dir = os.path.join(args.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        if all_histories:
            print("  Figure 1: Training curves...")
            plot_training_curves(all_histories, fig_dir)

        if all_pe_matrices:
            print("  Figure 2: Cosine similarity...")
            plot_cosine_similarity(all_pe_matrices, fig_dir)

            print("  Figure 3: PCA and t-SNE...")
            plot_pca_tsne(all_pe_matrices, fig_dir)

            print("  Figure 4: Dimension entropy...")
            plot_dimension_entropy(all_pe_matrices, fig_dir)

            print("  Figure 5: Variance per dimension...")
            plot_variance_per_dim(all_pe_matrices, fig_dir)

        if all_mi:
            print("  Figure 6: MI per layer...")
            plot_mi_per_layer(all_mi, fig_dir)

        if all_ablation:
            print("  Figure 7: Noise ablation...")
            plot_noise_ablation(all_ablation, fig_dir)

        if all_probe:
            print("  Figure 8: Probe analysis...")
            plot_probe_analysis(all_probe, fig_dir)

        if all_layer_ent:
            print("  Figure 9: Layer entropy...")
            plot_layer_entropy(all_layer_ent, fig_dir)

        # Save all results
        results = {
            'config': config,
            'pe_types': pe_types,
            'seeds': seeds,
        }
        with open(os.path.join(args.output_dir, 'experiment_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nAll done! Figures saved to {fig_dir}/")
        print("Generated figures:")
        for f_name in sorted(os.listdir(fig_dir)):
            print(f"  {f_name}")


if __name__ == '__main__':
    main()
