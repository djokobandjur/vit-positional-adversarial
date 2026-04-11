"""
Generate All 17 Figures for TIFS Paper
======================================
Reproduces all figures from adversarial_pe_results.json (ImageNet-100),
adversarial_pe_results_cifar100.json (CIFAR-100), and
analysis_data.json (noise ablation data).

Usage:
    python generate_figures.py

Requires: matplotlib, numpy, json
Outputs:  fig1_fgsm_pe.png through fig17_attack_heatmap.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 15


imagenet_path = '/content/drive/MyDrive/pe_experiment/results/adversarial_pe_results.json'
cifar100_path = '/content/drive/MyDrive/pe_experiment/results_cifar100/adversarial_pe_results_cifar100.json'

OUTPUT_DIR = '/content/drive/MyDrive/pe_experiment/adversarial_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================
with open('adversarial_pe_results.json') as f:
    imagenet = json.load(f)
with open('adversarial_pe_results_cifar100.json') as f:
    cifar = json.load(f)

PE_TYPES = ['learned', 'sinusoidal', 'rope', 'alibi']
PE_LABELS = ['Learned PE', 'Sinusoidal PE', 'RoPE', 'ALiBi']
PE_COLORS = {'learned': '#7B68EE', 'sinusoidal': '#00CED1', 'rope': '#FF6347', 'alibi': '#32CD32'}
SEEDS = ['42', '123', '456']
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
EPS_STR = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.2', '0.5', '1.0']

def get_mean_std(data, pe, attack, eps_list=EPS_STR):
    vals = []
    for eps in eps_list:
        seed_vals = [data[pe][s][attack][eps] for s in SEEDS]
        vals.append((np.mean(seed_vals), np.std(seed_vals)))
    return np.array([v[0] for v in vals]), np.array([v[1] for v in vals])

def get_clean(data, pe):
    return np.mean([data[pe][s]['clean_acc'] for s in SEEDS])

# ============================================================
# FIG 1: FGSM-PE Attack (ImageNet-100)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for pe, label in zip(PE_TYPES, PE_LABELS):
    mean, std = get_mean_std(imagenet, pe, 'fgsm_pe')
    ax.plot(EPSILONS, mean, 'o-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=8)
    ax.fill_between(EPSILONS, mean - std, mean + std, alpha=0.15, color=PE_COLORS[pe])
ax.set_xscale('log')
ax.set_xlabel('Perturbation budget ε (L∞)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('FGSM-PE Attack on Positional Encoding')
ax.legend(loc='lower left', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig1_fgsm_pe.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 1 saved")

# ============================================================
# FIG 2: PGD-PE Attack (ImageNet-100)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
markers = {'learned': 's', 'sinusoidal': 's', 'rope': 's', 'alibi': 's'}
for pe, label in zip(PE_TYPES, PE_LABELS):
    mean, std = get_mean_std(imagenet, pe, 'pgd_pe')
    ax.plot(EPSILONS, mean, marker='s', linestyle='-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=8)
    ax.fill_between(EPSILONS, mean - std, mean + std, alpha=0.15, color=PE_COLORS[pe])
ax.set_xscale('log')
ax.set_xlabel('Perturbation budget ε (L∞)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('PGD-PE Attack on Positional Encoding (T=20 steps)')
ax.legend(loc='lower left', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig2_pgd_pe.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 2 saved")

# ============================================================
# FIG 3: All Attacks Comparison (ImageNet-100, 2x2 grid)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
attacks = ['fgsm_pe', 'pgd_pe', 'vta']
attack_labels = ['FGSM-PE', 'PGD-PE (T=20)', 'VTA (Variance-Targeted)']
attack_colors = ['#4169E1', '#FF8C00', '#2E8B57']

for idx, (pe, label) in enumerate(zip(PE_TYPES, PE_LABELS)):
    ax = axes[idx // 2, idx % 2]
    clean = get_clean(imagenet, pe)
    ax.axhline(y=clean, color='gray', linestyle='--', alpha=0.5, label=f'Clean ({clean:.1f}%)')
    for atk, atk_label, atk_color in zip(attacks, attack_labels, attack_colors):
        mean, std = get_mean_std(imagenet, pe, atk)
        ax.plot(EPSILONS, mean, 'o-', label=atk_label, color=atk_color, linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('ε')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(label, fontweight='bold', color=PE_COLORS[pe])
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(-2, 90)
    ax.grid(True, alpha=0.3)

fig.suptitle('Comparison of Three PE Attack Strategies', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig3_all_attacks_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 3 saved")

# ============================================================
# FIG 4: Robustness Inversion (Random Noise vs Adversarial)
# ============================================================
# Random noise data from analysis_data.json
# Load actual noise ablation data
with open('analysis_data.json') as f:
    analysis = json.load(f)

noise_levels = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
noise_data = {}
for pe in PE_TYPES:
    noise_data[pe] = [np.mean([analysis[pe][s]['noise_ablation']['accuracies'][i] for s in SEEDS]) 
                      for i in range(len(noise_levels))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

# Left: Random noise
for pe, label in zip(PE_TYPES, PE_LABELS):
    ax1.plot(noise_levels, noise_data[pe], 'o-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=7)
ax1.set_xlabel('Random noise level (× σ_PE)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Random Gaussian Noise on PE')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right: PGD adversarial
for pe, label in zip(PE_TYPES, PE_LABELS):
    mean, _ = get_mean_std(imagenet, pe, 'pgd_pe')
    ax2.plot(EPSILONS, mean, 's-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=7)
ax2.set_xscale('log')
ax2.set_xlabel('PGD-PE budget ε (L∞)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('PGD Adversarial Attack on PE')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

fig.suptitle('The Robustness Inversion: Random Noise vs. Adversarial Attack', fontsize=15, fontweight='normal')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig4_robustness_inversion.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 4 saved")

# ============================================================
# FIG 5: Gini vs Critical Epsilon
# ============================================================
gini_values = {'learned': 0.457, 'sinusoidal': 0.523, 'rope': 0.513, 'alibi': 0.999}
crit_eps = {'learned': 0.2, 'sinusoidal': 0.5, 'rope': 2.0, 'alibi': 1.33}

fig, ax = plt.subplots(figsize=(8, 6))
for pe, label in zip(PE_TYPES, PE_LABELS):
    ax.scatter(gini_values[pe], crit_eps[pe], s=300, color=PE_COLORS[pe], zorder=5, edgecolors='black', linewidth=0.5)
    ax.annotate(label, (gini_values[pe], crit_eps[pe]), textcoords="offset points",
                xytext=(0, 15), ha='center', fontsize=12, fontweight='bold', color=PE_COLORS[pe])
ax.set_xlabel('Gini Coefficient of PE Variance Distribution')
ax.set_ylabel('Critical ε_crit (PGD-PE)')
ax.set_title('Variance Concentration vs. Adversarial Vulnerability')
ax.set_xlim(0.3, 1.1)
ax.set_ylim(0, 2.2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig5_gini_vs_crit.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 5 saved")

# ============================================================
# FIG 6: VTA Gain
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(EPSILONS))
width = 0.2

for i, (pe, label) in enumerate(zip(PE_TYPES, PE_LABELS)):
    gains = []
    for eps in EPS_STR:
        fgsm_vals = [imagenet[pe][s]['fgsm_pe'][eps] for s in SEEDS]
        vta_vals = [imagenet[pe][s]['vta'][eps] for s in SEEDS]
        clean = get_clean(imagenet, pe)
        fgsm_drop = clean - np.mean(fgsm_vals)
        vta_drop = clean - np.mean(vta_vals)
        gain = vta_drop / max(fgsm_drop, 0.01)
        gains.append(gain)
    ax.bar(x + i * width, gains, width, label=label, color=PE_COLORS[pe])

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='VTA = FGSM (gain=1.0)')
ax.set_xlabel('Perturbation budget ε')
ax.set_ylabel('VTA Gain (drop_VTA / drop_FGSM)')
ax.set_title('Effectiveness of Variance-Targeted Attack vs. FGSM-PE')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(EPS_STR)
ax.legend(fontsize=10)
ax.set_ylim(0, 5)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig6_vta_gain.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 6 saved")

# ============================================================
# FIG 7: PGD-PE on CIFAR-100
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for pe, label in zip(PE_TYPES, PE_LABELS):
    mean, std = get_mean_std(cifar, pe, 'pgd_pe')
    ax.plot(EPSILONS, mean, 's-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=8)
    ax.fill_between(EPSILONS, mean - std, mean + std, alpha=0.15, color=PE_COLORS[pe])
ax.set_xscale('log')
ax.set_xlabel('Perturbation budget ε (L∞)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('PGD-PE Attack on CIFAR-100 (T=20 steps)')
ax.legend(loc='lower left', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig7_cifar100_pgd.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 7 saved")

# ============================================================
# FIG 8: Cross-Dataset Validation (side by side)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

for pe, label in zip(PE_TYPES, PE_LABELS):
    mean, _ = get_mean_std(imagenet, pe, 'pgd_pe')
    ax1.plot(EPSILONS, mean, 's-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=7)
ax1.set_xscale('log')
ax1.set_xlabel('ε (L∞)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('ImageNet-100 (ViT-Base, 224×224)')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

for pe, label in zip(PE_TYPES, PE_LABELS):
    mean, _ = get_mean_std(cifar, pe, 'pgd_pe')
    ax2.plot(EPSILONS, mean, 's-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=7)
ax2.set_xscale('log')
ax2.set_xlabel('ε (L∞)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('CIFAR-100 (ViT-Base, 32×32)')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.suptitle('Cross-Dataset Validation: PGD-PE Attack Vulnerability', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig8_cross_dataset.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 8 saved")

# ============================================================
# FIG 9: CIFAR-100 All Attacks (2x2 grid)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (pe, label) in enumerate(zip(PE_TYPES, PE_LABELS)):
    ax = axes[idx // 2, idx % 2]
    clean = get_clean(cifar, pe)
    ax.axhline(y=clean, color='gray', linestyle='--', alpha=0.5, label=f'Clean ({clean:.1f}%)')
    for atk, atk_label, atk_color in zip(attacks, attack_labels, attack_colors):
        mean, _ = get_mean_std(cifar, pe, atk)
        ax.plot(EPSILONS, mean, 'o-', label=atk_label, color=atk_color, linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('ε')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(label, fontweight='bold', color=PE_COLORS[pe])
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(-2, 80)
    ax.grid(True, alpha=0.3)

fig.suptitle('CIFAR-100: Three PE Attack Strategies Compared', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig9_cifar100_all_attacks.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 9 saved")

# ============================================================
# FIG 10: Critical Epsilon Comparison (ImageNet vs CIFAR)
# ============================================================
crit_eps_in = {'learned': 0.2, 'sinusoidal': 0.5, 'rope': 1.5, 'alibi': 1.17}
crit_eps_ci = {'learned': 0.05, 'sinusoidal': 0.2, 'rope': 1.5, 'alibi': 0.67}

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(4)
width = 0.35

bars1 = ax.bar(x - width/2, [crit_eps_in[pe] for pe in PE_TYPES], width,
               label='ImageNet-100', color=[PE_COLORS[pe] for pe in PE_TYPES], edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, [crit_eps_ci[pe] for pe in PE_TYPES], width,
               label='CIFAR-100', color=[PE_COLORS[pe] for pe in PE_TYPES], alpha=0.5,
               edgecolor='black', linewidth=0.5, hatch='//')

ax.axhline(y=1.5, color='red', linestyle=':', linewidth=1.5)
ax.text(3.5, 1.52, '> 1.0 (immune)', color='red', fontsize=10, ha='right')
ax.set_ylabel('Critical ε_crit (PGD-PE)')
ax.set_xlabel('PE Type')
ax.set_title('Critical ε Comparison: ImageNet-100 vs CIFAR-100')
ax.set_xticks(x)
ax.set_xticklabels(PE_LABELS)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig10_crit_eps_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 10 saved")

# ============================================================
# FIG 11: Normalized Degradation
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

for pe, label in zip(PE_TYPES, PE_LABELS):
    clean = get_clean(imagenet, pe)
    mean, _ = get_mean_std(imagenet, pe, 'pgd_pe')
    norm = (mean / clean) * 100
    ax1.plot(EPSILONS, norm, 's-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=7)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax1.set_xscale('log')
ax1.set_xlabel('Perturbation budget ε')
ax1.set_ylabel('Accuracy retained (%)')
ax1.set_title('ImageNet-100')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

for pe, label in zip(PE_TYPES, PE_LABELS):
    clean = get_clean(cifar, pe)
    mean, _ = get_mean_std(cifar, pe, 'pgd_pe')
    norm = (mean / clean) * 100
    ax2.plot(EPSILONS, norm, 's-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=7)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax2.set_xscale('log')
ax2.set_xlabel('Perturbation budget ε')
ax2.set_ylabel('Accuracy retained (%)')
ax2.set_title('CIFAR-100')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.suptitle('Normalized Degradation: Fraction of Clean Accuracy Retained Under PGD-PE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig11_normalized_degradation.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 11 saved")

# ============================================================
# FIG 12: Degradation Rate
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

for ds, data, ax, title in [(imagenet, imagenet, ax1, 'ImageNet-100'), (cifar, cifar, ax2, 'CIFAR-100')]:
    for pe, label in zip(PE_TYPES, PE_LABELS):
        mean, _ = get_mean_std(data, pe, 'pgd_pe')
        # Compute degradation rate: pp / Δlog10(ε)
        rates = []
        midpoints = []
        for i in range(len(EPSILONS) - 1):
            delta_acc = mean[i] - mean[i+1]
            delta_log = np.log10(EPSILONS[i+1]) - np.log10(EPSILONS[i])
            rates.append(delta_acc / delta_log)
            midpoints.append(np.sqrt(EPSILONS[i] * EPSILONS[i+1]))
        ax.plot(midpoints, rates, 'o-', label=label, color=PE_COLORS[pe], linewidth=2, markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('ε (midpoint)')
    ax.set_ylabel('Degradation rate (pp / Δlog₁₀ ε)')
    ax.set_title(f'{title}: Rate of Accuracy Loss')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle('Degradation Rate Analysis: Where Does Each PE Method Break?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig12_degradation_rate.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 12 saved")

# ============================================================
# FIG 13: Inflection Point Analysis
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))

for ds_name, data, marker in [('ImageNet-100', imagenet, 's'), ('CIFAR-100', cifar, 'D')]:
    for pe, label in zip(PE_TYPES, PE_LABELS):
        mean, _ = get_mean_std(data, pe, 'pgd_pe')
        rates = []
        midpoints = []
        for i in range(len(EPSILONS) - 1):
            delta_acc = mean[i] - mean[i+1]
            delta_log = np.log10(EPSILONS[i+1]) - np.log10(EPSILONS[i])
            rates.append(delta_acc / delta_log)
            midpoints.append(np.sqrt(EPSILONS[i] * EPSILONS[i+1]))
        peak_idx = np.argmax(rates)
        ax.scatter(midpoints[peak_idx], rates[peak_idx], s=250, color=PE_COLORS[pe],
                   marker=marker, edgecolors='black', linewidth=0.5, zorder=5)

# Legend markers
ax.scatter([], [], s=150, marker='s', color='gray', label='ImageNet-100')
ax.scatter([], [], s=150, marker='D', color='gray', label='CIFAR-100')
for pe, label in zip(PE_TYPES, PE_LABELS):
    ax.scatter([], [], s=150, color=PE_COLORS[pe], label=label)

ax.set_xscale('log')
ax.set_xlabel('Inflection point ε* (maximum degradation rate)')
ax.set_ylabel('Peak degradation rate (pp / Δlog₁₀ ε)')
ax.set_title('Inflection Point Analysis: Critical Vulnerability Threshold per PE Type')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig13_inflection_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 13 saved")

# ============================================================
# FIG 14: Variance Structure vs Vulnerability
# ============================================================
gini_in = {'learned': 0.457, 'sinusoidal': 0.523, 'rope': 0.513, 'alibi': 0.999}
gini_ci = {'learned': 0.315, 'sinusoidal': 0.641, 'rope': 0.513, 'alibi': 0.999}

fig, ax = plt.subplots(figsize=(10, 7))

# Background regions
ax.axvspan(0.25, 0.55, alpha=0.08, color='red', label='Embedding space')
ax.axvspan(0.55, 1.05, alpha=0.08, color='blue', label='Attention space')

for ds_name, data, gini_d, marker in [('ImageNet-100', imagenet, gini_in, 's'), ('CIFAR-100', cifar, gini_ci, 'D')]:
    for pe, label in zip(PE_TYPES, PE_LABELS):
        clean = get_clean(data, pe)
        pgd_1 = np.mean([data[pe][s]['pgd_pe']['1.0'] for s in SEEDS])
        norm_loss = ((clean - pgd_1) / clean) * 100
        ax.scatter(gini_d[pe], norm_loss, s=250, color=PE_COLORS[pe], marker=marker,
                   edgecolors='black', linewidth=0.5, zorder=5)

ax.text(0.35, 97, 'Embedding\nspace', fontsize=12, color='red', alpha=0.6, ha='center')
ax.text(0.82, 20, 'Attention\nspace', fontsize=12, color='blue', alpha=0.6, ha='center')

ax.scatter([], [], s=150, marker='s', color='gray', label='ImageNet-100')
ax.scatter([], [], s=150, marker='D', color='gray', label='CIFAR-100')
for pe, label in zip(PE_TYPES, PE_LABELS):
    ax.scatter([], [], s=150, color=PE_COLORS[pe], label=label)

ax.set_xlabel('Gini Coefficient of PE Variance Distribution')
ax.set_ylabel('Normalized accuracy loss at ε=1.0 (%)')
ax.set_title('PE Variance Structure vs. Adversarial Vulnerability')
ax.legend(fontsize=10, loc='center right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig14_variance_vulnerability.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 14 saved")

# ============================================================
# FIG 15: Resolution Effect
# ============================================================
crit_in = {'learned': 0.20, 'sinusoidal': 0.50, 'rope': 1.50, 'alibi': 1.17}
crit_ci = {'learned': 0.05, 'sinusoidal': 0.20, 'rope': 1.50, 'alibi': 0.67}

fig, ax = plt.subplots(figsize=(9, 7))
x_pos = [0, 1]  # CIFAR, ImageNet

for pe, label in zip(PE_TYPES, PE_LABELS):
    y = [crit_ci[pe], crit_in[pe]]
    ax.plot(x_pos, y, 'o-', color=PE_COLORS[pe], linewidth=3, markersize=12, label=label)
    ax.annotate(f'ε*={crit_ci[pe]:.2f}', (0, crit_ci[pe]), textcoords="offset points",
                xytext=(-50, 5), fontsize=9, color=PE_COLORS[pe])
    ax.annotate(f'ε*={crit_in[pe]:.2f}', (1, crit_in[pe]), textcoords="offset points",
                xytext=(10, 5), fontsize=9, color=PE_COLORS[pe])

ax.set_xticks([0, 1])
ax.set_xticklabels(['CIFAR-100\n(64 patches)', 'ImageNet-100\n(196 patches)'])
ax.set_xlabel('Number of patches (sequence length)')
ax.set_ylabel('Critical ε_crit (PGD-PE)')
ax.set_title('Resolution Effect: Patch Count vs. Adversarial Vulnerability')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig15_resolution_effect.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 15 saved")

# ============================================================
# FIG 16: Failure Cases (ALiBi variance + RoPE immunity)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

# ALiBi per-seed variance
colors_seed = ['#4169E1', '#FF8C00', '#2E8B57']
for i, s in enumerate(SEEDS):
    vals = [imagenet['alibi'][s]['pgd_pe'][eps] for eps in EPS_STR]
    ax1.plot(EPSILONS, vals, 's-', label=f'Seed {s}', color=colors_seed[i], linewidth=2, markersize=7)
mean_alibi, _ = get_mean_std(imagenet, 'alibi', 'pgd_pe')
ax1.plot(EPSILONS, mean_alibi, 'k-', label='Mean', linewidth=3, alpha=0.7)
ax1.set_xscale('log')
ax1.set_xlabel('ε')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('ALiBi: High Inter-Seed Variance (ImageNet-100)')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# RoPE consistency
mean_rope_in, _ = get_mean_std(imagenet, 'rope', 'pgd_pe')
mean_rope_ci, _ = get_mean_std(cifar, 'rope', 'pgd_pe')
ax2.plot(EPSILONS, mean_rope_in, 'o-', color=PE_COLORS['rope'], linewidth=2, markersize=7, label='RoPE mean (ImageNet-100)')
ax2.plot(EPSILONS, mean_rope_ci, 'o--', color=PE_COLORS['rope'], linewidth=2, markersize=7, alpha=0.6, label='RoPE mean (CIFAR-100)')
# Per seed thin lines
for s in SEEDS:
    vals_in = [imagenet['rope'][s]['pgd_pe'][eps] for eps in EPS_STR]
    vals_ci = [cifar['rope'][s]['pgd_pe'][eps] for eps in EPS_STR]
    ax2.plot(EPSILONS, vals_in, '-', color=PE_COLORS['rope'], linewidth=0.8, alpha=0.3)
    ax2.plot(EPSILONS, vals_ci, '--', color=PE_COLORS['rope'], linewidth=0.8, alpha=0.3)
ax2.set_xscale('log')
ax2.set_xlabel('ε')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('RoPE: Consistent Immunity Across Seeds & Datasets')
ax2.set_ylim(65, 90)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.suptitle('Failure Case Analysis: When Do Attacks Fail or Show High Variance?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig16_failure_cases.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 16 saved")

# ============================================================
# FIG 17: Attack Heatmap
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

cmap = LinearSegmentedColormap.from_list('custom', ['#8B0000', '#CD5C5C', '#F0E68C', '#90EE90', '#228B22'])

for ds_name, data, ax in [('ImageNet-100', imagenet, ax1), ('CIFAR-100', cifar, ax2)]:
    matrix = []
    for pe in PE_TYPES:
        clean = get_clean(data, pe)
        row = []
        for eps in EPS_STR:
            vals = [data[pe][s]['pgd_pe'][eps] for s in SEEDS]
            retained = (np.mean(vals) / clean) * 100
            row.append(round(retained))
        matrix.append(row)

    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    for i in range(4):
        for j in range(8):
            color = 'white' if matrix[i, j] < 30 else 'black'
            ax.text(j, i, f'{matrix[i, j]}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)

    ax.set_xticks(range(8))
    ax.set_xticklabels(EPS_STR, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(PE_LABELS, fontsize=11)
    ax.set_xlabel('ε')
    ax.set_title(f'{ds_name}: Accuracy Retained (%)')
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('PGD-PE Attack Success Heatmap: Accuracy Retained Across PE Types and Budgets', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'fig17_attack_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Fig 17 saved")

print("\n✅ All 17 figures generated successfully!")
