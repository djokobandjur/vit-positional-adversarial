#!/usr/bin/env python3
"""
generate_figures.py
===================

Generate all figures for the Adversarial PE Attacks paper.
Reads imagenet_results.json and cifar_results.json. Produces 6 figures
covering attack curves, robustness inversion, three-attack comparison
across both datasets, Gini + critical-epsilon summaries, VTA gain across
budgets, and degradation-rate inflection analysis.

USAGE
-----
    python generate_figures.py \
        --imagenet imagenet_results.json \
        --cifar cifar_results.json \
        --outdir figures/ \
        --format pdf            # or png, or both

REQUIREMENTS
------------
    matplotlib >= 3.5
    numpy

OUTPUTS (6 figures)
-------------------
    fig1_attack_curves_imagenet.{pdf,png}   -- FGSM + PGD on ImageNet (column)
    fig2_robustness_inversion.{pdf,png}     -- noise | adversarial (full width)
    fig3_three_attacks_both_datasets.{pdf,png} -- 2x4 grid (full width)
    fig4_vta_gain.{pdf,png}                 -- VTA gain across eps (column)
    fig5_gini_eps_crit.{pdf,png}            -- 2-panel: Gini scatter + bar (column)
    fig6_degradation_rate.{pdf,png}         -- d acc / d log10(eps) (full width)
    fig7_cross_dataset.{pdf,png}            -- PGD curves IN vs CF (full width)

"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# =========================================================================
# CONFIGURATION
# =========================================================================

# v15 color convention preserved
PE_COLORS = {
    'learned':    '#7e57c2',  # purple
    'sinusoidal': '#1976d2',  # blue
    'rope':       '#d32f2f',  # red
    'alibi':      '#f57c00',  # orange
}

PE_LABELS = {
    'learned':    'Learned',
    'sinusoidal': 'Sinusoidal',
    'rope':       'RoPE',
    'alibi':      'ALiBi',
}

# Order in which PE families are drawn (and appear in legends).
# Chosen so legend reflects the corrected vulnerability hierarchy on ImageNet:
# most vulnerable first.
PE_DRAW_ORDER = ['alibi', 'learned', 'sinusoidal', 'rope']

ATTACK_LABELS = {
    'fgsm_pe': 'FGSM-PE',
    'pgd_pe':  'PGD-PE',
    'vta':     'VTA',
}

ATTACK_LINESTYLES = {
    'fgsm_pe': '--',
    'pgd_pe':  '-',
    'vta':     ':',
}

# Width of IEEE single-column / double-column figures (inches).
COL_WIDTH = 3.5
DOUBLE_WIDTH = 7.16

# Real random-noise ablation data on ImageNet-100 (mean accuracy across 3 seeds).
# Source: analysis_data.json from full_scale_experiment.py noise_ablation function.
# Each transformer block receives an independent Gaussian noise sample scaled by
# per-PE sigma_PE (NOT shared across blocks, in contrast to the shared-delta
# adversarial attack). These produce the LEFT panel of fig:inversion.
HARDCODED_NOISE_DATA = {
    'learned':    {0.0: 79.44, 0.1: 79.45, 0.2: 79.41, 0.5: 79.32, 1.0: 79.21, 2.0: 77.91, 3.0: 75.24, 5.0: 51.89},
    'sinusoidal': {0.0: 81.46, 0.1: 81.29, 0.2: 80.91, 0.5: 78.38, 1.0: 47.71, 2.0:  2.06, 3.0:  2.01, 5.0:  1.21},
    'rope':       {0.0: 84.50, 0.1: 84.40, 0.2: 83.91, 0.5: 79.39, 1.0: 33.01, 2.0:  3.57, 3.0:  2.29, 5.0:  1.82},
    'alibi':      {0.0: 81.05, 0.1: 76.39, 0.2: 62.02, 0.5: 32.20, 1.0: 14.23, 2.0:  5.05, 3.0:  3.99, 5.0:  2.22},
}

# Hardcoded Gini coefficients from tab:gini (weight-distribution property).
GINI_BY_PE = {
    'learned':    0.457,
    'sinusoidal': 0.523,
    'rope':       0.507,
    'alibi':      0.998,
}


# =========================================================================
# MATPLOTLIB STYLE
# =========================================================================

def set_publication_style():
    """Configure matplotlib for IEEE-style publication output."""
    rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
        'font.size':         9,
        'axes.labelsize':    9,
        'axes.titlesize':    9,
        'legend.fontsize':   7.5,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'figure.dpi':        300,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype':      42,   # TrueType embedded in PDF
        'ps.fonttype':       42,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.7,
        'lines.linewidth':   1.4,
        'lines.markersize':  4.0,
        'grid.linewidth':    0.4,
        'grid.alpha':        0.35,
    })


# =========================================================================
# DATA LOADING
# =========================================================================

def load_results(path):
    """Load a results JSON and return {pe: {seed: {attack: {eps: acc}}}}."""
    with open(path) as f:
        d = json.load(f)
    return d


def aggregate_curve(results, pe, attack):
    """
    Return (eps_array, mean_acc, std_acc) for a given PE and attack,
    aggregated across seeds. Clean accuracy is prepended at eps=0 (its
    mean and std across the three seeds at the clean operating point).
    """
    pe_data = results['results'][pe]
    seeds = sorted(pe_data.keys(), key=lambda s: int(s))

    # Clean accuracies
    clean = np.array([pe_data[s]['clean_acc'] for s in seeds])

    # Identify eps points (string keys in JSON)
    first_seed = seeds[0]
    eps_strs = list(pe_data[first_seed]['attacks'][attack].keys())
    eps_floats = sorted(float(e) for e in eps_strs)

    # Collect (n_eps, n_seeds) accuracy matrix
    acc_matrix = np.zeros((len(eps_floats), len(seeds)))
    for i, eps_f in enumerate(eps_floats):
        # Match the float back to its string key (JSON keys are strings)
        eps_key = next(k for k in eps_strs if float(k) == eps_f)
        for j, s in enumerate(seeds):
            acc_matrix[i, j] = pe_data[s]['attacks'][attack][eps_key]['accuracy']

    mean_acc = acc_matrix.mean(axis=1)
    std_acc = acc_matrix.std(axis=1, ddof=1) if len(seeds) > 1 else np.zeros_like(mean_acc)

    # Prepend the clean operating point at eps=0
    eps_with_clean = np.concatenate([[0.0], eps_floats])
    mean_with_clean = np.concatenate([[clean.mean()], mean_acc])
    std_with_clean = np.concatenate([[clean.std(ddof=1) if len(seeds) > 1 else 0.0], std_acc])

    return eps_with_clean, mean_with_clean, std_with_clean


def compute_eps_crit(eps, acc, clean_acc):
    """
    Return the smallest eps at which accuracy drops below 50% of clean.
    Uses linear interpolation between adjacent eps points. Returns np.nan
    if the curve never crosses 50% of clean within the tested range.
    """
    target = 0.5 * clean_acc
    for i in range(len(eps) - 1):
        if acc[i] >= target and acc[i + 1] < target:
            # Linear interpolation in log10(eps) space (more honest for log-x plots)
            # but eps[0] = 0 breaks log -- fall back to linear interp if it's the first step.
            if eps[i] == 0:
                # Linear in eps
                frac = (acc[i] - target) / (acc[i] - acc[i + 1])
                return eps[i] + frac * (eps[i + 1] - eps[i])
            log_e0, log_e1 = np.log10(eps[i]), np.log10(eps[i + 1])
            frac = (acc[i] - target) / (acc[i] - acc[i + 1])
            return 10 ** (log_e0 + frac * (log_e1 - log_e0))
    return np.nan


def compute_degradation_rate(eps, acc):
    """
    Return (eps_midpoints, rate) where rate[i] = -d(acc)/d(log10(eps)) between
    adjacent eps points. Skips the eps=0 prepended clean point.
    """
    # Drop eps=0 for log-derivative
    mask = eps > 0
    eps_pos = eps[mask]
    acc_pos = acc[mask]
    if len(eps_pos) < 2:
        return np.array([]), np.array([])

    log_eps = np.log10(eps_pos)
    delta_acc = -np.diff(acc_pos)              # positive when accuracy drops
    delta_log_eps = np.diff(log_eps)
    rate = delta_acc / delta_log_eps           # accuracy loss per decade of eps

    eps_midpoints = 10 ** ((log_eps[:-1] + log_eps[1:]) / 2)
    return eps_midpoints, rate


# =========================================================================
# PLOTTING HELPERS
# =========================================================================

def _setup_eps_axis(ax, x_min=1e-3, x_max=1.0):
    """Configure shared log-eps axis style."""
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_xlim(0, x_max)
    ax.set_xlabel(r'Perturbation budget $\varepsilon$')
    ax.set_ylabel('Accuracy (\\%)')
    ax.set_ylim(0, 100)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.35)


def _draw_pe_curves(ax, results, attack, with_errorbars=True):
    """Plot all four PE attack curves on a single axis."""
    for pe in PE_DRAW_ORDER:
        eps, mean, std = aggregate_curve(results, pe, attack)
        color = PE_COLORS[pe]
        label = PE_LABELS[pe]
        if with_errorbars:
            ax.errorbar(
                eps, mean, yerr=std,
                color=color, label=label,
                marker='o', markersize=3.2,
                capsize=2, elinewidth=0.6, capthick=0.6,
            )
        else:
            ax.plot(eps, mean, color=color, label=label, marker='o', markersize=3.2)


# =========================================================================
# FIGURE 1: Attack curves on ImageNet (FGSM | PGD)
# =========================================================================

def fig1_attack_curves_imagenet(imagenet, outdir, fmt):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.6), sharey=True)

    # Left: FGSM-PE
    _draw_pe_curves(axes[0], imagenet, 'fgsm_pe')
    _setup_eps_axis(axes[0])
    axes[0].set_title('(a) FGSM-PE')

    # Right: PGD-PE
    _draw_pe_curves(axes[1], imagenet, 'pgd_pe')
    _setup_eps_axis(axes[1])
    axes[1].set_title('(b) PGD-PE')
    axes[1].set_ylabel('')

    # Shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.06),
        ncol=4, frameon=False,
    )

    _save(fig, outdir, 'fig1_attack_curves_imagenet', fmt)


# =========================================================================
# FIGURE 2: Robustness inversion (noise | adversarial)
# =========================================================================

def fig2_robustness_inversion(imagenet, outdir, fmt):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.8))

    # Left: random Gaussian noise (hardcoded data)
    ax = axes[0]
    for pe in PE_DRAW_ORDER:
        data = HARDCODED_NOISE_DATA[pe]
        sigma = sorted(data.keys())
        acc = [data[s] for s in sigma]
        ax.plot(
            sigma, acc,
            color=PE_COLORS[pe], label=PE_LABELS[pe],
            marker='o', markersize=3.2,
        )
    ax.set_xlabel(r'Noise scale ($\sigma / \sigma_\mathrm{PE}$)')
    ax.set_ylabel('Accuracy (\\%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 5.0)
    ax.set_title('(a) Random Gaussian noise')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.35)

    # Right: PGD-PE adversarial
    _draw_pe_curves(axes[1], imagenet, 'pgd_pe')
    _setup_eps_axis(axes[1])
    axes[1].set_title('(b) PGD-PE adversarial')

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.04),
        ncol=4, frameon=False,
    )

    _save(fig, outdir, 'fig2_robustness_inversion', fmt)


# =========================================================================
# FIGURE 3: Three attacks per PE, both datasets (2x4 grid)
# =========================================================================

def fig3_three_attacks_both_datasets(imagenet, cifar, outdir, fmt):
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_WIDTH, 4.4), sharex=True, sharey=True)

    datasets = [('ImageNet-100', imagenet, 0), ('CIFAR-100', cifar, 1)]
    attacks = ['fgsm_pe', 'pgd_pe', 'vta']

    for ds_name, results, row_idx in datasets:
        for col_idx, pe in enumerate(PE_DRAW_ORDER):
            ax = axes[row_idx, col_idx]
            for attack in attacks:
                eps, mean, std = aggregate_curve(results, pe, attack)
                ax.plot(
                    eps, mean,
                    color=PE_COLORS[pe],
                    linestyle=ATTACK_LINESTYLES[attack],
                    marker='o', markersize=2.5,
                    label=ATTACK_LABELS[attack] if (row_idx == 0 and col_idx == 0) else None,
                )
            ax.set_xscale('symlog', linthresh=1e-3)
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 100)
            ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.35)
            if row_idx == 0:
                ax.set_title(PE_LABELS[pe])
            if row_idx == 1:
                ax.set_xlabel(r'$\varepsilon$')
            if col_idx == 0:
                ax.set_ylabel(f'{ds_name}\nAccuracy (\\%)')

    # Legend for attacks
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.03),
        ncol=3, frameon=False,
    )

    _save(fig, outdir, 'fig3_three_attacks_both_datasets', fmt)


# =========================================================================
# FIGURE 4: VTA gain over FGSM-PE across eps
# =========================================================================

def fig4_vta_gain(imagenet, outdir, fmt):
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))

    for pe in PE_DRAW_ORDER:
        eps_f, fgsm_acc, _ = aggregate_curve(imagenet, pe, 'fgsm_pe')
        eps_v, vta_acc, _ = aggregate_curve(imagenet, pe, 'vta')
        # Clean is at index 0
        clean = fgsm_acc[0]
        fgsm_drop = clean - fgsm_acc[1:]   # drop from clean
        vta_drop = clean - vta_acc[1:]
        # Avoid division by zero where FGSM drop is negligible
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = np.where(fgsm_drop > 0.5, vta_drop / fgsm_drop, np.nan)
        ax.plot(
            eps_f[1:], gain,
            color=PE_COLORS[pe], label=PE_LABELS[pe],
            marker='o', markersize=3.2,
        )

    ax.axhline(1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
    ax.text(0.5, 1.03, 'VTA = FGSM', color='red', fontsize=7, ha='center')
    ax.set_xscale('log')
    ax.set_xlabel(r'Perturbation budget $\varepsilon$')
    ax.set_ylabel(r'VTA gain $= \Delta_\mathrm{VTA} / \Delta_\mathrm{FGSM}$')
    ax.set_ylim(0, 1.6)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, loc='upper right')

    _save(fig, outdir, 'fig4_vta_gain', fmt)


# =========================================================================
# FIGURE 5: Gini scatter + eps_crit bar plot
# =========================================================================

def fig5_gini_eps_crit(imagenet, cifar, outdir, fmt):
    fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH, 4.0))

    # Top panel: Gini vs eps_crit scatter
    ax = axes[0]
    for pe in PE_DRAW_ORDER:
        gini = GINI_BY_PE[pe]
        # ImageNet eps_crit
        eps_in, acc_in, _ = aggregate_curve(imagenet, pe, 'pgd_pe')
        clean_in = acc_in[0]
        eps_crit_in = compute_eps_crit(eps_in, acc_in, clean_in)
        # CIFAR eps_crit
        eps_cf, acc_cf, _ = aggregate_curve(cifar, pe, 'pgd_pe')
        clean_cf = acc_cf[0]
        eps_crit_cf = compute_eps_crit(eps_cf, acc_cf, clean_cf)

        ax.scatter(gini, eps_crit_in, color=PE_COLORS[pe], marker='o', s=40,
                   label=f'{PE_LABELS[pe]} (IN)')
        ax.scatter(gini, eps_crit_cf, color=PE_COLORS[pe], marker='s', s=40,
                   facecolors='none', edgecolors=PE_COLORS[pe], linewidths=1.2,
                   label=f'{PE_LABELS[pe]} (CF)')

    ax.set_xlabel('Variance Gini coefficient')
    ax.set_ylabel(r'$\varepsilon_\mathrm{crit}$ (50\% clean)')
    ax.set_yscale('log')
    ax.set_title('(a) Gini vs.\\ critical $\\varepsilon$')
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=6, loc='upper right', ncol=2, columnspacing=0.6)

    # Bottom panel: eps_crit bar chart (IN vs CF)
    ax = axes[1]
    width = 0.35
    positions = np.arange(len(PE_DRAW_ORDER))
    eps_in_list, eps_cf_list = [], []
    for pe in PE_DRAW_ORDER:
        e_in, a_in, _ = aggregate_curve(imagenet, pe, 'pgd_pe')
        e_cf, a_cf, _ = aggregate_curve(cifar, pe, 'pgd_pe')
        eps_in_list.append(compute_eps_crit(e_in, a_in, a_in[0]))
        eps_cf_list.append(compute_eps_crit(e_cf, a_cf, a_cf[0]))

    colors = [PE_COLORS[pe] for pe in PE_DRAW_ORDER]
    ax.bar(positions - width/2, eps_in_list, width, color=colors, alpha=0.95, label='ImageNet-100')
    ax.bar(positions + width/2, eps_cf_list, width, color=colors, alpha=0.55, hatch='///',
           edgecolor='black', linewidth=0.4, label='CIFAR-100')

    ax.set_yscale('log')
    ax.set_xticks(positions)
    ax.set_xticklabels([PE_LABELS[pe] for pe in PE_DRAW_ORDER])
    ax.set_ylabel(r'$\varepsilon_\mathrm{crit}$')
    ax.set_title('(b) $\\varepsilon_\\mathrm{crit}$ by dataset')
    ax.grid(True, axis='y', which='both', linestyle='--', linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=7)

    fig.tight_layout()
    _save(fig, outdir, 'fig5_gini_eps_crit', fmt)


# =========================================================================
# FIGURE 6: Degradation rate (inflection points)
# =========================================================================

def fig6_degradation_rate(imagenet, cifar, outdir, fmt):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.8), sharey=True)

    for ax_idx, (name, results) in enumerate([('ImageNet-100', imagenet),
                                                ('CIFAR-100', cifar)]):
        ax = axes[ax_idx]
        for pe in PE_DRAW_ORDER:
            eps, acc, _ = aggregate_curve(results, pe, 'pgd_pe')
            mid, rate = compute_degradation_rate(eps, acc)
            if len(mid) == 0:
                continue
            ax.plot(mid, rate, color=PE_COLORS[pe], label=PE_LABELS[pe],
                    marker='o', markersize=3.2)

        ax.set_xscale('log')
        ax.set_xlabel(r'Perturbation budget $\varepsilon$')
        if ax_idx == 0:
            ax.set_ylabel(r'$-\Delta\mathrm{acc} / \Delta \log_{10}\varepsilon$ (pp/decade)')
        ax.set_title(name)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.35)
        ax.axhline(0, color='black', linewidth=0.4, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.04),
        ncol=4, frameon=False,
    )

    _save(fig, outdir, 'fig6_degradation_rate', fmt)


# =========================================================================
# FIGURE 7: Cross-dataset PGD curves
# =========================================================================

def fig7_cross_dataset(imagenet, cifar, outdir, fmt):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.6), sharey=True)

    for ax_idx, (name, results) in enumerate([('(a) ImageNet-100', imagenet),
                                                ('(b) CIFAR-100', cifar)]):
        ax = axes[ax_idx]
        _draw_pe_curves(ax, results, 'pgd_pe')
        _setup_eps_axis(ax)
        ax.set_title(name)
        if ax_idx == 1:
            ax.set_ylabel('')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.06),
        ncol=4, frameon=False,
    )

    _save(fig, outdir, 'fig7_cross_dataset', fmt)


# =========================================================================
# FIGURE 8 (Supplementary): ALiBi structural ablation
# =========================================================================

def _aggregate_ablation_curve(ablation_data, regime):
    """
    Return (eps_array, mean_acc, std_acc) for an ablation regime
    across the three seeds.
    """
    alibi_data = ablation_data['results']['alibi']
    seeds = sorted(alibi_data.keys(), key=lambda s: int(s))
    clean = np.array([alibi_data[s]['clean_acc'] for s in seeds])

    first_seed = seeds[0]
    regime_key = f'regime_{regime}'
    eps_strs = list(alibi_data[first_seed][regime_key]['pgd_pe'].keys())
    eps_floats = sorted(float(e) for e in eps_strs)

    acc_matrix = np.zeros((len(eps_floats), len(seeds)))
    for i, eps_f in enumerate(eps_floats):
        eps_key = next(k for k in eps_strs if float(k) == eps_f)
        for j, s in enumerate(seeds):
            acc_matrix[i, j] = alibi_data[s][regime_key]['pgd_pe'][eps_key]['accuracy']

    mean_acc = acc_matrix.mean(axis=1)
    std_acc = acc_matrix.std(axis=1, ddof=1) if len(seeds) > 1 else np.zeros_like(mean_acc)

    eps_with_clean = np.concatenate([[0.0], eps_floats])
    mean_with_clean = np.concatenate([[clean.mean()], mean_acc])
    std_with_clean = np.concatenate([[clean.std(ddof=1) if len(seeds) > 1 else 0.0], std_acc])

    return eps_with_clean, mean_with_clean, std_with_clean


def fig8_alibi_ablation(imagenet_abl, cifar_abl, outdir, fmt):
    """ALiBi structural ablation across both datasets (supplementary)."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.8), sharey=True)

    regime_styles = {
        'slopes_only':  ('-',  'slopes only (12 scalars)'),
        'reldist_only': (':',  r'rel\_dist only ($N{\times}N$)'),
        'both':         ('--', 'both (slopes + rel\\_dist)'),
    }
    color = PE_COLORS['alibi']

    for ax_idx, (name, data) in enumerate([('(a) ImageNet-100', imagenet_abl),
                                             ('(b) CIFAR-100', cifar_abl)]):
        ax = axes[ax_idx]
        for regime, (style, label) in regime_styles.items():
            eps, mean, std = _aggregate_ablation_curve(data, regime)
            ax.errorbar(
                eps, mean, yerr=std,
                color=color, linestyle=style,
                marker='o', markersize=3.0,
                capsize=2, elinewidth=0.6, capthick=0.6,
                label=label if ax_idx == 0 else None,
            )
        _setup_eps_axis(ax)
        ax.set_title(name)
        if ax_idx == 1:
            ax.set_ylabel('')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=3, frameon=False,
    )

    _save(fig, outdir, 'fig8_alibi_ablation_supp', fmt)


# =========================================================================
# EXPERIMENT 3 HELPERS: spatial metrics & perturbation norms
# =========================================================================

def _iso_accuracy_metric(pe_data, metric_key, target_acc=40.0):
    """
    For each seed, linearly interpolate the metric value at the eps where
    attacked_accuracy crosses target_acc. Returns (mean, std) over seeds.
    Returns (nan, nan) if interpolation failed (no crossing).
    """
    seed_values = []
    for seed in pe_data:
        attacks = pe_data[seed]['attacks']
        eps_list = sorted(float(e) for e in attacks.keys())
        accs, metrics = [], []
        for e in eps_list:
            key = str(e) if e != 0 else '0'
            if key not in attacks:
                key = str(int(e)) if e == 0 else key
            accs.append(attacks[key]['attacked_accuracy'])
            metrics.append(attacks[key][metric_key])
        accs = np.array(accs)
        metrics = np.array(metrics)
        # Find crossing point (descending acc curve)
        for i in range(len(eps_list) - 1):
            if accs[i] >= target_acc and accs[i + 1] < target_acc:
                frac = (accs[i] - target_acc) / (accs[i] - accs[i + 1])
                m = metrics[i] + frac * (metrics[i + 1] - metrics[i])
                seed_values.append(m)
                break
    if not seed_values:
        return float('nan'), float('nan')
    mean = float(np.mean(seed_values))
    std = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
    return mean, std


def _concentration_ratio(mass, R, grid_side):
    """
    Convert raw R-mass to concentration ratio (mass / mean-coverage), where
    mean-coverage is the average fraction of grid cells within Chebyshev
    distance R of a random grid center (accounts for boundary truncation).
    """
    # Compute boundary-corrected mean coverage by enumeration
    total = grid_side * grid_side
    n_within = 0
    for cx in range(grid_side):
        for cy in range(grid_side):
            for x in range(grid_side):
                for y in range(grid_side):
                    if max(abs(x - cx), abs(y - cy)) <= R:
                        n_within += 1
    mean_coverage = n_within / (total * total)
    return mass / mean_coverage


def _grid_side_for_dataset(dataset_name):
    return 14 if dataset_name == 'imagenet' else 8


def _aggregate_norm_curve(perturbation_data, pe, buffer_name):
    """
    Return (eps_array, mean_frac_ceil, std_frac_ceil) for a given PE and
    specific buffer (e.g., 'cos_cached', 'inv_freq', 'slopes', 'pos_embed').
    """
    pe_data = perturbation_data['results'][pe]
    seeds = sorted(pe_data.keys(), key=lambda s: int(s))

    first_seed = seeds[0]
    eps_strs = list(pe_data[first_seed]['attacks'].keys())
    eps_floats = sorted(float(e) for e in eps_strs if float(e) > 0)

    matrix = np.zeros((len(eps_floats), len(seeds)))
    for i, eps_f in enumerate(eps_floats):
        eps_key = next(k for k in eps_strs if float(k) == eps_f)
        for j, s in enumerate(seeds):
            entry = pe_data[s]['attacks'][eps_key]['norms_per_buffer'].get(buffer_name)
            if entry is None:
                matrix[i, j] = float('nan')
            else:
                matrix[i, j] = entry['fraction_at_ceiling']

    mean = np.nanmean(matrix, axis=1)
    std = np.nanstd(matrix, axis=1, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
    return np.array(eps_floats), mean, std


# =========================================================================
# FIGURE 9: Concentration ratio at iso-accuracy (40%) — main paper
# =========================================================================

def fig9_concentration_ratio(imagenet_spatial, cifar_spatial, outdir, fmt):
    """
    Bar plot of concentration_ratio = R_mass / uniform_coverage at iso-accuracy
    operating point (attacked_accuracy = 40% per PE, interpolated). Two panels:
    R1 (very local) and R3 (broader local). Each panel: 4 PE x 2 datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.6))

    # Use concentration ordering (Learned < Sinusoidal < RoPE < ALiBi)
    # instead of adversarial ordering for the x-axis
    pe_order = ['learned', 'sinusoidal', 'rope', 'alibi']

    datasets = [('ImageNet-100', imagenet_spatial, 'imagenet'),
                ('CIFAR-100',   cifar_spatial,   'cifar')]

    for ax_idx, R in enumerate([1, 3]):
        ax = axes[ax_idx]
        width = 0.35
        positions = np.arange(len(pe_order))

        for ds_idx, (ds_name, ds_data, ds_key) in enumerate(datasets):
            grid_side = _grid_side_for_dataset(ds_key)
            # Compute mean coverage (used for std rescaling below)
            n_within = 0
            for cx in range(grid_side):
                for cy in range(grid_side):
                    for x in range(grid_side):
                        for y in range(grid_side):
                            if max(abs(x - cx), abs(y - cy)) <= R:
                                n_within += 1
            mean_coverage = n_within / (grid_side ** 4)

            ratios, stds = [], []
            for pe in pe_order:
                pe_data = ds_data['results'][pe]
                mean, std = _iso_accuracy_metric(
                    pe_data, f'mean_mass_R{R}_attacked', target_acc=40.0
                )
                ratios.append(_concentration_ratio(mean, R, grid_side))
                stds.append(std / mean_coverage)

            offset = (ds_idx - 0.5) * width
            bar_colors = [PE_COLORS[pe] for pe in pe_order]
            hatch = '///' if ds_idx == 1 else None
            alpha = 0.55 if ds_idx == 1 else 0.95
            ax.bar(positions + offset, ratios, width,
                   color=bar_colors, alpha=alpha, hatch=hatch,
                   edgecolor='black' if ds_idx == 1 else None, linewidth=0.4,
                   yerr=stds, capsize=2, error_kw={'elinewidth': 0.5})

        # Manual legend with neutral grey bars to distinguish datasets only
        if ax_idx == 0:
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor='grey', alpha=0.95, label='ImageNet-100'),
                Patch(facecolor='grey', alpha=0.55, hatch='///', edgecolor='black',
                      linewidth=0.4, label='CIFAR-100'),
            ]
            ax._legend_handles = legend_handles

        ax.axhline(1.0, color='black', linewidth=0.5, linestyle='--', alpha=0.6)
        # Position "uniform" label more carefully
        ax.text(0.02, 1.05, 'uniform attention', fontsize=6.5, ha='left',
                color='black', transform=ax.get_yaxis_transform())
        ax.set_xticks(positions)
        ax.set_xticklabels([PE_LABELS[pe] for pe in pe_order])
        ax.set_ylabel(f'Concentration ratio (mass within $R{{=}}{R}$)' if ax_idx == 0 else '')
        ax.set_title(f'({chr(97 + ax_idx)}) $R={R}$')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.35)

    fig.legend(
        handles=axes[0]._legend_handles,
        loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=2, frameon=False,
    )

    _save(fig, outdir, 'fig9_concentration_ratio', fmt)


# =========================================================================
# FIGURE 10: Adversarial saturation profiles (frac@ceil vs eps) — main paper
# =========================================================================

def fig10_saturation_profiles(imagenet_norms, cifar_norms, outdir, fmt):
    """
    Plot fraction_at_ceiling vs epsilon for each PE's principal buffer,
    on both datasets. RoPE has multiple buffers; we plot cos_cached as the
    main RoPE curve and inv_freq as a secondary line in the same color.
    """
    # Each PE has a primary buffer (one we plot as the main line)
    primary_buffer = {
        'learned':    'pos_embed',
        'sinusoidal': 'pe',
        'rope':       'cos_cached',
        'alibi':      'slopes',
    }
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.8), sharey=True)

    datasets = [('(a) ImageNet-100', imagenet_norms),
                ('(b) CIFAR-100', cifar_norms)]

    for ax_idx, (name, data) in enumerate(datasets):
        ax = axes[ax_idx]
        for pe in PE_DRAW_ORDER:
            buf = primary_buffer[pe]
            eps, mean, std = _aggregate_norm_curve(data, pe, buf)
            ax.errorbar(
                eps, mean, yerr=std,
                color=PE_COLORS[pe], label=PE_LABELS[pe],
                marker='o', markersize=3.2,
                capsize=2, elinewidth=0.6, capthick=0.6,
            )
            # RoPE: also show inv_freq as a faded dotted line
            if pe == 'rope':
                eps_inv, mean_inv, std_inv = _aggregate_norm_curve(data, pe, 'inv_freq')
                ax.plot(eps_inv, mean_inv, color=PE_COLORS[pe], linestyle=':',
                        marker='x', markersize=3.0, alpha=0.65,
                        label='RoPE (inv\\_freq)' if ax_idx == 0 else None)

        ax.set_xscale('log')
        ax.set_xlabel(r'Perturbation budget $\varepsilon$')
        if ax_idx == 0:
            ax.set_ylabel(r'$\mathrm{frac}_{\mathrm{ceil}}$')
        ax.set_ylim(0, 1.05)
        ax.set_title(name)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=5, frameon=False,
    )

    _save(fig, outdir, 'fig10_saturation_profiles', fmt)


# =========================================================================
# FIGURE 11 (Supplementary): MAD_miss vs accuracy drop
# =========================================================================

def fig11_mad_vs_acc_drop_supp(imagenet_spatial, cifar_spatial, outdir, fmt):
    """
    Two panels: left = raw scatter MAD_miss vs accuracy_drop. Right = bar plot
    of mean MAD_miss/drop ratio per (PE, dataset), highlighting CIFAR Learned
    as a structural outlier (factor-of-~15 vs ImageNet Learned for the same
    PE family).
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 3.0),
                             gridspec_kw={'width_ratios': [1.4, 1]})

    datasets = [('ImageNet-100', imagenet_spatial, 'imagenet', 'o', 0.95),
                ('CIFAR-100',   cifar_spatial,   'cifar',   's', 0.55)]

    # LEFT panel: scatter of all (PE, dataset, eps, seed) points
    ax = axes[0]
    pe_order = ['learned', 'sinusoidal', 'rope', 'alibi']
    for pe in pe_order:
        for ds_name, ds_data, ds_key, marker, alpha in datasets:
            pe_data = ds_data['results'][pe]
            xs, ys = [], []
            for seed in pe_data:
                attacks = pe_data[seed]['attacks']
                clean = attacks['0']['clean_accuracy']
                for eps_key, entry in attacks.items():
                    if float(eps_key) == 0:
                        continue
                    drop = clean - entry['attacked_accuracy']
                    mad = entry['mean_mad_miss']
                    if drop > 0 and not np.isnan(mad):
                        xs.append(drop)
                        ys.append(mad)
            ax.scatter(xs, ys, color=PE_COLORS[pe], marker=marker, s=24,
                       alpha=alpha, edgecolors='black', linewidths=0.3,
                       label=f'{PE_LABELS[pe]} ({ds_name[:3]})')

    ax.set_xlabel('Accuracy drop under attack (pp)')
    ax.set_ylabel(r'$\mathrm{MAD}_{\mathrm{miss}}$ (patches)')
    ax.set_title('(a) Raw scatter')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc='upper right')

    # RIGHT panel: ratio MAD_miss/drop per (PE, dataset), bar plot
    ax = axes[1]
    width = 0.35
    positions = np.arange(len(pe_order))
    for ds_idx, (ds_name, ds_data, ds_key, marker, alpha) in enumerate(datasets):
        ratios, errs = [], []
        for pe in pe_order:
            pe_data = ds_data['results'][pe]
            seed_ratios = []
            for seed in pe_data:
                attacks = pe_data[seed]['attacks']
                clean = attacks['0']['clean_accuracy']
                # Find the eps that gets us closest to acc=40% (iso-accuracy point)
                best_dist = float('inf')
                best_pair = None
                for eps_key, entry in attacks.items():
                    if float(eps_key) == 0:
                        continue
                    if abs(entry['attacked_accuracy'] - 40.0) < best_dist:
                        best_dist = abs(entry['attacked_accuracy'] - 40.0)
                        best_pair = (clean - entry['attacked_accuracy'], entry['mean_mad_miss'])
                if best_pair:
                    drop, mad = best_pair
                    if drop > 0:
                        seed_ratios.append(mad / drop)
            ratios.append(np.mean(seed_ratios) if seed_ratios else 0)
            errs.append(np.std(seed_ratios, ddof=1) if len(seed_ratios) > 1 else 0)

        offset = (ds_idx - 0.5) * width
        bar_colors = [PE_COLORS[pe] for pe in pe_order]
        hatch = '///' if ds_idx == 1 else None
        ax.bar(positions + offset, ratios, width,
               color=bar_colors, alpha=alpha, hatch=hatch,
               edgecolor='black' if ds_idx == 1 else None, linewidth=0.4,
               yerr=errs, capsize=2, error_kw={'elinewidth': 0.5})

    # Highlight CIFAR Learned anomaly with red box and annotation
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (positions[0] + width/2 - 0.2, -0.005),
        0.4, 0.075,
        fill=False, edgecolor='red', linewidth=1.4, linestyle='--',
    ))
    # Annotate with axes-relative offset so the arrow stays inside the panel
    ax.annotate('CIFAR Learned\n(post-collapse)',
                xy=(positions[0] + width/2, 0.062),
                xytext=(positions[1] - 0.1, 0.14),
                fontsize=6.5, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.6))

    ax.set_xticks(positions)
    ax.set_xticklabels([PE_LABELS[pe] for pe in pe_order])
    ax.set_ylabel(r'$\mathrm{MAD}_{\mathrm{miss}}$ / accuracy drop')
    ax.set_title('(b) MAD$_{\\mathrm{miss}}$/drop at iso-acc 40\\%')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.35)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='grey', alpha=0.95, label='ImageNet-100'),
        Patch(facecolor='grey', alpha=0.55, hatch='///', edgecolor='black',
              linewidth=0.4, label='CIFAR-100'),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=7,
              loc='upper right', ncol=1)

    _save(fig, outdir, 'fig11_mad_vs_acc_drop_supp', fmt)


# =========================================================================
# FIGURE 12 (Supplementary): ALiBi small-eps R-mass bump
# =========================================================================

def fig12_alibi_smalleps_bump_supp(imagenet_spatial, cifar_spatial, outdir, fmt):
    """
    For ALiBi: plot R1 and R3 attacked-minus-clean mass deltas as a function
    of small eps. Shows the brief concentration *increase* before collapse.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, 2.8), sharey=True)

    datasets = [('(a) ImageNet-100', imagenet_spatial),
                ('(b) CIFAR-100',   cifar_spatial)]
    color = PE_COLORS['alibi']

    for ax_idx, (name, data) in enumerate(datasets):
        ax = axes[ax_idx]
        pe_data = data['results']['alibi']
        seeds = sorted(pe_data.keys(), key=lambda s: int(s))
        first_seed = seeds[0]
        eps_list = sorted(float(e) for e in pe_data[first_seed]['attacks'].keys() if float(e) > 0)

        delta_R1_matrix = np.zeros((len(eps_list), len(seeds)))
        delta_R3_matrix = np.zeros((len(eps_list), len(seeds)))

        for i, eps_f in enumerate(eps_list):
            for j, s in enumerate(seeds):
                attacks = pe_data[s]['attacks']
                eps_key = next(k for k in attacks.keys() if float(k) == eps_f)
                entry = attacks[eps_key]
                delta_R1_matrix[i, j] = entry['mean_mass_R1_attacked'] - entry['mean_mass_R1_clean']
                delta_R3_matrix[i, j] = entry['mean_mass_R3_attacked'] - entry['mean_mass_R3_clean']

        d1_mean = delta_R1_matrix.mean(axis=1)
        d1_std = delta_R1_matrix.std(axis=1, ddof=1)
        d3_mean = delta_R3_matrix.mean(axis=1)
        d3_std = delta_R3_matrix.std(axis=1, ddof=1)

        ax.errorbar(eps_list, d1_mean, yerr=d1_std, color=color, marker='o',
                    markersize=3.2, capsize=2, elinewidth=0.6, capthick=0.6,
                    linestyle='-', label=r'$\Delta R_1 = R_1^{\mathrm{atk}} - R_1^{\mathrm{clean}}$' if ax_idx == 0 else None)
        ax.errorbar(eps_list, d3_mean, yerr=d3_std, color=color, marker='s',
                    markersize=3.2, capsize=2, elinewidth=0.6, capthick=0.6,
                    linestyle='--', alpha=0.7,
                    label=r'$\Delta R_3 = R_3^{\mathrm{atk}} - R_3^{\mathrm{clean}}$' if ax_idx == 0 else None)
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

        ax.set_xscale('log')
        ax.set_xlabel(r'Perturbation budget $\varepsilon$')
        if ax_idx == 0:
            ax.set_ylabel('Attacked $-$ clean R-mass')
        ax.set_title(name)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=2, frameon=False,
    )

    _save(fig, outdir, 'fig12_alibi_smalleps_supp', fmt)


# =========================================================================
# SAVE HELPER
# =========================================================================

def _save(fig, outdir, basename, fmt):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    formats = ['pdf', 'png'] if fmt == 'both' else [fmt]
    for f in formats:
        path = outdir / f'{basename}.{f}'
        fig.savefig(path, format=f)
        print(f'  saved {path}')
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description='Generate paper figures.')
    ap.add_argument('--imagenet', required=True, help='Path to imagenet_results.json')
    ap.add_argument('--cifar',    required=True, help='Path to cifar_results.json')
    ap.add_argument('--imagenet-ablation', default=None,
                    help='Path to imagenet_alibi_ablation.json (for Fig 8 supplementary)')
    ap.add_argument('--cifar-ablation',    default=None,
                    help='Path to cifar_alibi_ablation.json (for Fig 8 supplementary)')
    ap.add_argument('--imagenet-spatial', default=None,
                    help='Path to imagenet_spatial_metrics.json (for Figs 9, 11, 12)')
    ap.add_argument('--cifar-spatial',    default=None,
                    help='Path to cifar_spatial_metrics.json (for Figs 9, 11, 12)')
    ap.add_argument('--imagenet-norms', default=None,
                    help='Path to imagenet_perturbation_norms.json (for Fig 10)')
    ap.add_argument('--cifar-norms',    default=None,
                    help='Path to cifar_perturbation_norms.json (for Fig 10)')
    ap.add_argument('--outdir',   default='figures', help='Output directory')
    ap.add_argument('--format',   default='pdf', choices=['pdf', 'png', 'both'],
                    help='Output format (default: pdf)')
    args = ap.parse_args()

    set_publication_style()

    print('Loading data...')
    imagenet = load_results(args.imagenet)
    cifar = load_results(args.cifar)

    print('\nGenerating figures...')

    print('[1] Attack curves (ImageNet, FGSM + PGD)')
    fig1_attack_curves_imagenet(imagenet, args.outdir, args.format)

    print('[2] Robustness inversion (noise | adversarial)')
    fig2_robustness_inversion(imagenet, args.outdir, args.format)

    print('[3] Three attacks across both datasets (2x4 grid)')
    fig3_three_attacks_both_datasets(imagenet, cifar, args.outdir, args.format)

    print('[4] VTA gain across budgets')
    fig4_vta_gain(imagenet, args.outdir, args.format)

    print('[5] Gini scatter + eps_crit bar')
    fig5_gini_eps_crit(imagenet, cifar, args.outdir, args.format)

    print('[6] Degradation rate (inflection points)')
    fig6_degradation_rate(imagenet, cifar, args.outdir, args.format)

    print('[7] Cross-dataset PGD curves')
    fig7_cross_dataset(imagenet, cifar, args.outdir, args.format)

    n_total = 7

    if args.imagenet_ablation and args.cifar_ablation:
        print('[8] ALiBi structural ablation (supplementary)')
        imagenet_abl = load_results(args.imagenet_ablation)
        cifar_abl = load_results(args.cifar_ablation)
        fig8_alibi_ablation(imagenet_abl, cifar_abl, args.outdir, args.format)
        n_total += 1
    else:
        print('[8] Skipping ALiBi ablation (no --imagenet-ablation/--cifar-ablation)')

    if args.imagenet_spatial and args.cifar_spatial:
        print('[9] Concentration ratio at iso-accuracy 40%')
        imagenet_sp = load_results(args.imagenet_spatial)
        cifar_sp = load_results(args.cifar_spatial)
        fig9_concentration_ratio(imagenet_sp, cifar_sp, args.outdir, args.format)
        n_total += 1
        print('[11] MAD_miss vs accuracy drop (supplementary)')
        fig11_mad_vs_acc_drop_supp(imagenet_sp, cifar_sp, args.outdir, args.format)
        n_total += 1
        print('[12] ALiBi small-eps bump (supplementary)')
        fig12_alibi_smalleps_bump_supp(imagenet_sp, cifar_sp, args.outdir, args.format)
        n_total += 1
    else:
        print('[9, 11, 12] Skipping spatial-metric figures (no --imagenet-spatial/--cifar-spatial)')

    if args.imagenet_norms and args.cifar_norms:
        print('[10] Adversarial saturation profiles (frac@ceil)')
        imagenet_nm = load_results(args.imagenet_norms)
        cifar_nm = load_results(args.cifar_norms)
        fig10_saturation_profiles(imagenet_nm, cifar_nm, args.outdir, args.format)
        n_total += 1
    else:
        print('[10] Skipping saturation profile figure (no --imagenet-norms/--cifar-norms)')

    print(f'\nDone. {n_total} figures saved to {args.outdir}/')


if __name__ == '__main__':
    main()
