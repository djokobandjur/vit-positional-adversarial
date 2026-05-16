# Adversarial Vulnerability of Positional Encoding in Vision Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djokobandjur/vit-positional-adversarial/blob/main/colab_quickstart.ipynb)
[![Paper Status](https://img.shields.io/badge/paper-resubmission-orange)](#paper-and-citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, trained models, and reproducibility scripts for the paper
*"Adversarial Vulnerability of Positional Encoding in Vision Transformers"*
(T-IFS-26532-2026, resubmission v24).

---

## What is this repository?

Vision Transformers depend on a *positional encoding* (PE) to inject spatial
structure into a permutation-invariant attention mechanism. This work measures
how four widely-used PE families — **Learned**, **Sinusoidal**, **RoPE**, and
**ALiBi** — behave under targeted, gradient-based perturbations applied
*directly to the PE buffers* (rather than to input pixels), at training-comparable
budgets and with a corrected multi-block attack that perturbs all 12 transformer
blocks via a single shared δ.

The central empirical finding is what the paper calls the
**robustness decoupling**: the random-noise robustness ranking and the
adversarial robustness ranking are not the same and do not reduce to one another.
Across two datasets (ImageNet-100, CIFAR-100), three seeds, and three attack
families (FGSM-PE, PGD-PE, VTA):

| Regime          | Ranking (most → least robust)            |
| --------------- | ---------------------------------------- |
| Random noise    | Learned ≫ Sinusoidal > RoPE > ALiBi      |
| Adversarial PGD-PE | RoPE ≫ Learned ≈ Sinusoidal ≫ ALiBi   |

Only ALiBi keeps its rank (bottom) across both regimes. The paper develops
this through three experiments: the main attack sweep (E1), an ALiBi
structural ablation (E2), and an attention-reorganization + saturation
analysis (E3).

---

## What changed in this release

The v1.0 manuscript was withdrawn from IEEE TIFS on May 6, 2026 after we
identified a **multi-block bug** in the original attack implementation: the
perturbation δ was applied only to `model.blocks[0].attn.rope.cos_cached`
(and the analogous ALiBi slope buffer), hitting only 1/12 of the RoPE/ALiBi
mechanism. Learned and Sinusoidal results were unaffected because their PE
parameters are top-level tensors shared across blocks. The bug therefore
selectively understated RoPE and ALiBi vulnerability.

v2.0 corrects this. The new attack pipeline (`full_reanalysis.py`) uses a
single shared δ applied to **all 12 transformer blocks** for RoPE and ALiBi,
matching the intent of the original protocol. Trained model weights are
unchanged (the bug was in attack code, not training code), so v1.0 checkpoints
remain valid; only the attack JSON outputs and the figures derived from them
have been regenerated.

See [`CHANGELOG.md`](CHANGELOG.md) for the full list of changes between v1.0
and v2.0, including the addition of Experiment 2 (ALiBi structural ablation)
and Experiment 3 (attention reorganization and saturation), and the
reframing of the thesis from "robustness inversion" to "robustness decoupling".

---

## Repository layout

```
vit-positional-adversarial/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
│
├── full_scale_experiment.py            # ImageNet-100 training + shared helpers
├── cifar100_experiment.py              # CIFAR-100 training (attack section deprecated)
├── 00_setup_imagenet.py                # Colab utility: build ImageNet-100 val/
│
├── full_reanalysis.py                  # v2.0 attacks (IN + CF, multi-block corrected)
├── experiment2_alibi_ablation.py       # E2: ALiBi structural ablation
├── experiment3_attention_metrics_v3.py # E3: spatial attention metrics
├── experiment3_perturbation_norm_v4.py # E3: perturbation norms / saturation
├── extract_tables_data.py              # Noise ablation (IN-only)
├── generate_figures.py                 # Figure generation from JSON results
│
├── colab_quickstart.ipynb              # End-to-end Colab pipeline notebook
│
├── data/                               # JSON result files (9 total)
│   ├── imagenet_results.json
│   ├── cifar_results.json
│   ├── imagenet_alibi_ablation.json
│   ├── cifar_alibi_ablation.json
│   ├── imagenet_spatial_metrics.json
│   ├── cifar_spatial_metrics.json
│   ├── imagenet_perturbation_norms.json
│   ├── cifar_perturbation_norms.json
│   └── analysis_data.json
│
└── figures/                            # Generated figures (12 total: 7 main + 5 supplement)
    └── (PDF/PNG output of generate_figures.py)
```

### Script-to-output mapping

| Script | Purpose | Output |
| --- | --- | --- |
| `full_scale_experiment.py` | ViT model + 4 PE variants; ImageNet-100 training; helpers (`noise_ablation`, `probe_analysis`, `extract_positional_embedding`) imported by other scripts | Trained model checkpoints |
| `cifar100_experiment.py` | CIFAR-100 training (12 models = 4 PE × 3 seeds); attack section is **deprecated** and exits early — use `full_reanalysis.py` for v2.0 attacks | Trained model checkpoints |
| `00_setup_imagenet.py` | Colab-specific: extracts the 100 selected classes from ILSVRC2012 val tar into `/content/imagenet100/val/` (50 images per class). Hardcoded for the Colab pipeline; see *Adapting for Local Execution* below | `/content/imagenet100/val/<class>/` |
| `full_reanalysis.py` | **Main v2.0 attack script.** Runs FGSM-PE, PGD-PE, VTA across 8 ε values, 3 seeds, both datasets. Multi-block corrected. Checkpoints after each (PE, seed) combination | `imagenet_results.json`, `cifar_results.json` |
| `experiment2_alibi_ablation.py` | Decomposes ALiBi into slopes vs. relative-distance components; attacks each in isolation | `imagenet_alibi_ablation.json`, `cifar_alibi_ablation.json` |
| `experiment3_attention_metrics_v3.py` | Spatial attention metrics under attack (concentration ratio, MAD) | `imagenet_spatial_metrics.json`, `cifar_spatial_metrics.json` |
| `experiment3_perturbation_norm_v4.py` | Per-step PGD perturbation norm tracking; `frac@ceil` saturation profile | `imagenet_perturbation_norms.json`, `cifar_perturbation_norms.json` |
| `extract_tables_data.py` | Random-noise ablation on PE buffers (ImageNet-100 only); decoupling-thesis data | `analysis_data.json` |
| `generate_figures.py` | Consumes all 9 JSON files → 12 figures (7 main + 5 supplement) | `figures/*.pdf` (or `.png` with `--format png`) |

---

## Trained models

Trained model checkpoints (4 PE types × 3 seeds × 2 datasets = 24 models)
are hosted on Google Drive:

- **ImageNet-100 models:** [drive.google.com/.../1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9](https://drive.google.com/drive/folders/1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9)
- **CIFAR-100 models:** [drive.google.com/.../1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL](https://drive.google.com/drive/folders/1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL)

The folder structure expected by the scripts is:

```
<models_dir>/
├── learned_seed42/best_model.pth
├── learned_seed123/best_model.pth
├── learned_seed456/best_model.pth
├── sinusoidal_seed42/best_model.pth
├── ...
└── alibi_seed456/best_model.pth
```

These are the same weights used in v1.0. The multi-block bug fixed in v2.0
was in attack code (inference time), not in training code, so retraining
was not required.

---

## Reproducing v2.0 results

### Recommended: one-click Colab

The fastest path is the [Colab Quickstart notebook](colab_quickstart.ipynb)
([open in Colab](https://colab.research.google.com/github/djokobandjur/vit-positional-adversarial/blob/main/colab_quickstart.ipynb)).
It clones this repo, mounts Drive for model and dataset access, and walks
through the full pipeline (attacks → E2 → E3 → noise ablation → figures)
in seven sections. Each section can be re-run independently provided the
prior JSON outputs are present on Drive.

Expected wall-clock on a single G4: roughly 10–14h for the complete
pipeline. Most of that is the main attack sweep (`full_reanalysis.py`,
~6–8h).

### Manual CLI workflow

The notebook is a thin wrapper over CLI commands. Each script supports a
consistent `--models_dir / --val_dir / --output_path` interface. The
command sequence below assumes Colab paths and reproduces the JSON files
shipped under `data/`.

**1. Prepare datasets**

```bash
# ImageNet-100: extract 100 selected classes from ILSVRC2012 val tar
python 00_setup_imagenet.py
# CIFAR-100 needs no preparation — torchvision auto-downloads on first use.
```

**2. Run the corrected attacks (Experiment 1)**

```bash
# ImageNet-100
python full_reanalysis.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/content/imagenet100/val" \
    --output_path "data/imagenet_results.json" \
    --dataset imagenet

# CIFAR-100
python full_reanalysis.py \
    --models_dir "/path/to/Trained models_CIFAR100" \
    --val_dir    "/path/to/cifar100_cache" \
    --output_path "data/cifar_results.json" \
    --dataset cifar
```

**3. ALiBi structural ablation (Experiment 2)**

```bash
python experiment2_alibi_ablation.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/content/imagenet100/val" \
    --output_path "data/imagenet_alibi_ablation.json" \
    --dataset imagenet

python experiment2_alibi_ablation.py \
    --models_dir "/path/to/Trained models_CIFAR100" \
    --val_dir    "/path/to/cifar100_cache" \
    --output_path "data/cifar_alibi_ablation.json" \
    --dataset cifar
```

**4. Attention reorganization + saturation (Experiment 3)**

```bash
# Spatial metrics
python experiment3_attention_metrics_v3.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/content/imagenet100/val" \
    --output_path "data/imagenet_spatial_metrics.json" \
    --dataset imagenet --batch_size 128

python experiment3_attention_metrics_v3.py \
    --models_dir "/path/to/Trained models_CIFAR100" \
    --output_path "data/cifar_spatial_metrics.json" \
    --dataset cifar --batch_size 128

# Perturbation norms / saturation
python experiment3_perturbation_norm_v4.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/content/imagenet100/val" \
    --output_path "data/imagenet_perturbation_norms.json" \
    --dataset imagenet --batch_size 128

python experiment3_perturbation_norm_v4.py \
    --models_dir "/path/to/Trained models_CIFAR100" \
    --output_path "data/cifar_perturbation_norms.json" \
    --dataset cifar --batch_size 128
```

Note: CIFAR-100 runs of the E3 scripts do not require `--val_dir` — the
script invokes `torchvision.datasets.CIFAR100(download=True)` internally.

**5. Noise ablation (decoupling thesis data; ImageNet-100 only)**

```bash
python extract_tables_data.py \
    --models_dir "/path/to/Trained models_ImageNet100" \
    --val_dir    "/content/imagenet100/val" \
    --output_path "data/analysis_data.json"
```

**6. Generate figures**

```bash
python generate_figures.py \
    --imagenet           data/imagenet_results.json \
    --cifar              data/cifar_results.json \
    --imagenet-ablation  data/imagenet_alibi_ablation.json \
    --cifar-ablation     data/cifar_alibi_ablation.json \
    --imagenet-spatial   data/imagenet_spatial_metrics.json \
    --cifar-spatial      data/cifar_spatial_metrics.json \
    --imagenet-norms     data/imagenet_perturbation_norms.json \
    --cifar-norms        data/cifar_perturbation_norms.json \
    --outdir             figures/ \
    --format pdf
```

Pass `--format png` for raster output or `--format both` to emit both.

---

## Adapting for local execution

All experiment scripts (attacks, E2, E3, noise ablation, figures) accept
their input and output paths as CLI arguments — adapt them to your local
filesystem by changing the `--models_dir`, `--val_dir`, and `--output_path`
values when invoking the script. No code changes are required for the
experiment scripts.

A few practical notes for non-Colab execution:

- **`00_setup_imagenet.py`** is a Colab-specific utility. It hardcodes the
  target directory (`/content/imagenet100/val`) and reads the ILSVRC2012 val
  tar from a specific Drive path. For local execution, either edit the path
  constants near the top of the script, or skip it entirely and assemble
  an ImageFolder-compatible `val/` directory yourself with the 100 classes
  used in this work (the class label list is embedded in the script).

- **ImageNet-100 dataset structure** required by the attack and analysis
  scripts is the standard ImageFolder layout: one subdirectory per class,
  with image files inside. Pass the path to the parent of the class
  subdirectories as `--val_dir`.

- **CIFAR-100** requires no manual setup. The scripts invoke
  `torchvision.datasets.CIFAR100(download=True)` and cache to a local
  directory; for CIFAR-100 runs the `--val_dir` argument selects the
  torchvision cache location. For the E3 scripts on CIFAR, `--val_dir` is
  not required at all.

- **Training scripts** (`full_scale_experiment.py`, `cifar100_experiment.py`)
  contain some hardcoded Colab paths in their `__main__` blocks. If you
  intend to retrain (rather than use the published checkpoints), inspect
  the `RESULTS_DIR` and dataset path constants near the top of each script.

---

## Key v2.0 results

### Inflection thresholds ε* (PGD-PE, ImageNet-100)

ε* is the smallest ε at which an attack first crosses a fixed clean-accuracy
drop threshold. Lower ε* means a smaller perturbation suffices to break the
model.

| PE         | ImageNet-100 ε* | CIFAR-100 ε* |
| ---------- | --------------: | -----------: |
| Learned    | 0.14            | 0.02         |
| Sinusoidal | 0.14            | 0.14         |
| RoPE       | **0.32**        | **0.32**     |
| ALiBi      | 0.07            | 0.07         |

### Clean accuracy (mean ± std over 3 seeds)

| PE         | ImageNet-100   | CIFAR-100      |
| ---------- | -------------- | -------------- |
| Learned    | 79.44 ± 0.62   | 68.28 ± 0.38   |
| Sinusoidal | 81.46 ± 0.33   | 66.92 ± 0.50   |
| RoPE       | 84.51 ± 0.41   | 73.30 ± 0.18   |
| ALiBi      | 81.05 ± 0.36   | 67.66 ± 0.44   |

### ALiBi structural ablation (E2, ImageNet-100, ε=0.1)

| Variant         | Accuracy |
| --------------- | -------: |
| slopes only     | 14.7%    |
| reldist only    | 80.9%    |
| both (joint)    | 14.7%    |

Attacking only the per-head slopes reproduces the joint attack effect almost
exactly; attacking only the relative-distance term leaves the model nearly
untouched. The ALiBi vulnerability is concentrated in the slopes.

Full result tables, additional ε values, and per-attack breakdowns are in
the paper and supplement.

---

## Paper and citation

The paper is currently under resubmission to IEEE Transactions on Information
Forensics and Security (T-IFS-26532-2026). Citation details will be added
once the resubmission status is finalized. In the meantime, please reference
this repository directly if you build on the code or use the v2.0 results.

---

## License

Code and result files in this repository are released under the MIT License
(see [LICENSE](LICENSE)). Trained model weights on Google Drive are
released under the same terms.

The ImageNet-1k validation images required to reproduce the ImageNet-100
attacks are governed by the [ImageNet terms of access](https://www.image-net.org/download.php)
and are not redistributed here.
