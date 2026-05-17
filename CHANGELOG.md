# Changelog

All notable changes to this repository are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-05-16

Corrected multi-block attack and revised manuscript. The original manuscript
was withdrawn from IEEE TIFS on May 6, 2026 after we identified a bug in
the attack implementation that selectively understated RoPE and ALiBi
vulnerability. This release fixes the bug, regenerates all affected
results, adds two new experiments (E2: ALiBi structural ablation; E3:
attention reorganization and saturation), and reframes the central thesis
from "robustness inversion" to "robustness decoupling".

Trained model weights are unchanged — the bug was in attack code only.

### Fixed

- **Multi-block attack bug** (the core fix). In previous version, the perturbation
  δ for RoPE and ALiBi attacks was applied only to
  `model.blocks[0].attn.rope.cos_cached` (and the analogous ALiBi slope
  buffer in block 0), hitting only 1/12 of the RoPE/ALiBi mechanism. The
  corrected attack uses a single shared δ applied to all 12 transformer
  blocks (`shared_delta_all_12_blocks` in result metadata). Learned and
  Sinusoidal PE were unaffected by the bug because their parameters are
  top-level tensors shared across blocks; their original numbers carry over
  unchanged.

### Added

- **`full_reanalysis.py`** — unified new attack script covering both
  ImageNet-100 and CIFAR-100 via `--dataset {imagenet,cifar}`. Replaces
  `adversarial_pe_attacks.py` (ImageNet-only, contained the
  multi-block bug) and supersedes the attack section of
  `cifar100_experiment.py`. Implements FGSM-PE, PGD-PE, and VTA across
  8 ε values and 3 seeds, with checkpointing after each (PE, seed)
  combination.

- **`experiment2_alibi_ablation.py`** — Experiment 2: ALiBi structural
  ablation. Decomposes the ALiBi mechanism into its two components
  (per-head slopes, relative-distance matrix) and attacks each in
  isolation. Establishes that essentially all ALiBi vulnerability is
  carried by the slopes (`slopes_only ≈ both ≈ 14.7%` accuracy at ε=0.1
  on ImageNet-100; `reldist_only ≈ 80.9%`, nearly unchanged from clean).

- **`experiment3_attention_metrics_v3.py`** — Experiment 3a: spatial
  attention metrics under attack (concentration ratio, MAD). Measures
  whether the attack induces sharper focus, broader diffusion, or no
  spatial restructuring.

- **`experiment3_perturbation_norm_v4.py`** — Experiment 3b: per-step
  PGD perturbation norm tracking with `frac@ceil` saturation profile.
  Quantifies how often the attacker's update saturates the L∞ budget
  versus being bounded by gradient signal.

- **`colab_quickstart.ipynb`** — end-to-end Colab notebook covering the
  full pipeline (dataset prep → attacks → E2 → E3 → noise ablation
  → figures) in seven sections. Replaces the earlier notebooks (see
  *Removed*).

- **CLI interface for `extract_tables_data.py`** — the noise ablation
  script previously required editing hardcoded path constants. It now
  accepts `--models_dir`, `--val_dir`, `--output_path`, plus optional
  `--batch_size` and `--seeds`, matching the convention of all other
  experiment scripts.

- **Repository layout** — JSON result files consolidated into `data/`;
  generated figures into `figures/`. Scripts remain in the repository
  root for direct CLI invocation.

- **New result files** under `data/`:
  - `imagenet_alibi_ablation.json`, `cifar_alibi_ablation.json` (E2)
  - `imagenet_spatial_metrics.json`, `cifar_spatial_metrics.json` (E3a)
  - `imagenet_perturbation_norms.json`, `cifar_perturbation_norms.json` (E3b)

### Changed

- **Central thesis reframed** from "robustness inversion" to "robustness
  decoupling". The original "inversion" framing implied a direct
  reordering between random-noise and adversarial regimes. The corrected
  numbers show a more nuanced picture: only ALiBi keeps its rank (bottom)
  across both regimes, while the other three PE families reshuffle in a
  way that does not reduce to a simple inversion. The thesis is now that
  the two robustness rankings are **decoupled** — neither implies the
  other — which is a stronger and more defensible claim than inversion.

- **VTA gain values** — the previous manuscript reported a "4.9× gain"
  headline figure for VTA over PGD-PE. The corrected multi-block VTA
  attack produces sub-1.0 gain across all four PE types (max 0.74× on
  RoPE; Learned and ALiBi ≈ 0.02×). The resubmitted manuscript discusses VTA
  honestly in this regime; the relevant figure has been moved to the
  supplement (Fig. S2).

- **Result JSONs regenerated** for ImageNet-100 and CIFAR-100 with the
  corrected attack. `imagenet_results.json` and `cifar_results.json`
  shipped in this release are the corrected multi-block outputs; the previous outputs
  are not retained in `data/` but remain in repository history.

- **Figures regenerated.** 12 figures total (7 main paper, 5 supplement),
  produced by `generate_figures.py` from the new JSON files. Two figures
  were moved from the main paper to the supplement during revision
  (Fig. S2: VTA gain; Fig. S5: degradation rate). Two figures are new
  (Fig. 6: concentration ratio; Fig. 7: saturation profiles).

- **Clean accuracy values updated** with three-seed mean ± std
  (full numbers in README and paper).

- **Inflection thresholds ε\*** recomputed under the corrected attack.

### Deprecated

- **Attack section of `cifar100_experiment.py`**. The training section
  (Phase 1 of `__main__`) remains the active CIFAR-100 training entry
  point and is unchanged. The attack section (Phase 2 of `__main__`, and
  the supporting functions `get_pe_variance_weights`, `fgsm_pe_attack`,
  `pgd_pe_attack`, `vta_attack`) contains the multi-block bug and
  is no longer executed: `__main__` now exits with `sys.exit(0)` after
  Phase 1 completes, with a console message pointing users to
  `full_reanalysis.py --dataset cifar`. The deprecated attack code is
  retained in the file for transparency and historical reference, behind
  the early exit.

### Removed

- **`adversarial_pe_attacks.py`** — the original ImageNet-100 attack script,
  superseded by `full_reanalysis.py`. Contained the multi-block bug.
  Reachable in repository history via the v1.2.3 tag.

- **`ImageNet100_START.ipynb`** and **`CIFAR100_START.ipynb`** — the earlier
  Colab orchestrator notebooks, superseded by a single
  `colab_quickstart.ipynb`. The new notebook covers both datasets and
  the full new manuscript pipeline in one artifact.

### Repository / process

- **Manuscript LaTeX revisions: v15 → v24** (internal versioning of the
  source file `adversarial_pe_tifs_vNN.tex`; unrelated to repository release
  tags). Two new experiments added (E2, E3); thesis reframed; 7 main figures
  + 5 supplement figures; final manuscript fits within the IEEE TIFS 15-page
  limit.

- **Math notation fix.** Five instances of double-superscript notation
  (`\delta^c_i^m`, `\delta^s_i^m`) corrected to subscript-tuple form
  (`\delta^c_{i,m}`, `\delta^s_{i,m}`). The double-superscript was
  silently fatal in some LaTeX engine configurations and prevented `.aux`
  generation, breaking cross-references on platforms that auto-compile
  per file.

---

## [1.2.3] — 2026

Initial submission to IEEE Transactions on Information Forensics and
Security (T-IFS-26532-2026). **Withdrawn 2026-05-06** after identification
of the multi-block attack bug described above. That release is preserved
for historical reference; superseded by the new submission for all scientific purposes.

### Included

- Four PE variants implemented in a single ViT codebase: Learned,
  Sinusoidal, RoPE, ALiBi
- Two datasets: ImageNet-100, CIFAR-100
- Three attack families: FGSM-PE, PGD-PE, VTA
- Three seeds per (PE, dataset) configuration
- Trained model checkpoints on Google Drive (24 models total)
- Training scripts (`full_scale_experiment.py`, `cifar100_experiment.py`)
- Attack scripts (`adversarial_pe_attacks.py`, attack section of
  `cifar100_experiment.py`) — **contained the multi-block bug**

### Known issues (motivating resubmission)

- Attack δ targeted only `model.blocks[0]` for RoPE and ALiBi, not all
  12 blocks
- Reported VTA gain figures overstated the attack's effectiveness due to
  the same multi-block scoping
- "Robustness inversion" framing was not fully supported by the corrected
  numbers
