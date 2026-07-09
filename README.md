# Adversarial Vulnerability of Positional Encoding in Vision Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/djokobandjur/vit-positional-adversarial/blob/main/scripts/colab_quickstart_n6.ipynb)
[![Paper Status](https://img.shields.io/badge/paper-resubmission-orange)](#paper-and-citation)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

Code, aggregate results, and figure-generation scripts for the n=6 experimental revision of:

**“Adversarial Vulnerability of Positional Encoding in Vision Transformers”**  
IEEE TIFS resubmission, T-IFS-26532-2026.

This repository is a reproducibility package for experiments, JSON outputs, and generated figures. It does **not** include the manuscript source, supplementary source, bibliography file, or submission PDFs.

---

## What is this repository?

Vision Transformers depend on positional encoding (PE) to inject spatial structure into an otherwise permutation-invariant attention mechanism. This repository evaluates how four PE families behave under targeted perturbations applied directly to PE buffers rather than to input pixels:

- Learned absolute PE
- Sinusoidal absolute PE
- RoPE
- ALiBi

The experiments compare three PE-space attack families:

- **FGSM-PE:** one-step gradient-sign PE perturbation
- **PGD-PE:** iterative projected gradient attack on PE buffers
- **VTA:** coordinate-weighted diagnostic attack that weights the gradient sign by normalized PE-buffer magnitude

VTA is used as a saliency / coordinate-weighted diagnostic. It should not be interpreted as a true variance-optimal attack or as a claim of universal attack dominance over PGD-PE.

The current release uses **6 random seeds**:

```text
42, 123, 456, 789, 1011, 1213
```

and evaluates:

```text
4 PE families × 6 seeds × 2 datasets = 48 trained ViT models
```

The central empirical finding is **robustness decoupling**: random-noise robustness and adversarial PE-space robustness follow different orderings. In the tested PGD-PE regime, RoPE has the highest adversarial threshold, while ALiBi is highly sensitive because a very small slope parameter space controls a global distance bias.

---

## What changed in the n=6 release?

This n=6 release supersedes the earlier n=3 results.

Main changes:

1. The experimental seed count increased from **3 to 6**.
2. The model count increased from **24 to 48**.
3. All main JSON result files were regenerated or extended to include six seeds.
4. All figures were regenerated from the n=6 result files.
5. The attack scripts in `scripts/` are the n=6 versions.
6. The figure-generation script now uses `--imagenet-analysis`, so Fig. 2 is generated from the n=6 `analysis_data.json` rather than from an old fallback.
7. VTA is documented as a coordinate-weighted diagnostic attack rather than a true variance attack.

Earlier versions of this project also corrected a multi-block attack bug in which RoPE/ALiBi perturbations were applied only to the first transformer block. The corrected attack applies a shared perturbation to all 12 transformer blocks for RoPE and ALiBi.

---

## Repository layout

```text
vit-positional-adversarial/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
│
├── full_scale_experiment.py
├── cifar100_experiment.py
├── 00_setup_imagenet.py
├── 00_setup_cifar100.py
│
├── data/
│   ├── imagenet100_classes.txt
│   └── val_labels.txt
│
├── scripts/
│   ├── full_reanalysis_n6.py
│   ├── extract_tables_data_n6.py
│   ├── experiment2_alibi_ablation_n6.py
│   ├── experiment3_attention_metrics_v3_n6.py
│   ├── experiment3_perturbation_norm_v4_n6.py
│   ├── generate_figures_n6.py
│   ├── extract_paper_numbers_n6.py
│   └── colab_quickstart_n6.ipynb
│
├── results/
│   ├── imagenet_results.json
│   ├── cifar_results.json
│   ├── analysis_data.json
│   ├── imagenet_alibi_ablation.json
│   ├── cifar_alibi_ablation.json
│   ├── imagenet_spatial_metrics.json
│   ├── cifar_spatial_metrics.json
│   ├── imagenet_perturbation_norms.json
│   └── cifar_perturbation_norms.json
│
└── figures/
    ├── fig1_attack_curves_imagenet.pdf
    ├── fig2_robustness_inversion.pdf
    ├── fig3_three_attacks_both_datasets.pdf
    ├── fig4_vta_gain.pdf
    ├── fig5_gini_eps_crit.pdf
    ├── fig6_degradation_rate.pdf
    ├── fig7_cross_dataset.pdf
    ├── fig8_alibi_ablation_supp.pdf
    ├── fig9_concentration_ratio.pdf
    ├── fig10_saturation_profiles.pdf
    ├── fig11_mad_vs_acc_drop_supp.pdf
    └── fig12_alibi_smalleps_supp.pdf
```

The manuscript `.tex` files, supplementary `.tex` file, bibliography file, and submission PDFs are intentionally not included in this repository.

---

## Script map: inputs and outputs

| Script | Main input | Main output | What it computes / prepares |
| --- | --- | --- | --- |
| `00_setup_imagenet.py` | ILSVRC2012 validation tar, `data/imagenet100_classes.txt`, `data/val_labels.txt` | ImageFolder-compatible ImageNet-100 `val/` directory | Extracts the selected ImageNet-100 validation classes into the folder layout expected by the attack and analysis scripts. |
| `00_setup_cifar100.py` | Cached `cifar-100-python` directory on Drive or local disk | Local torchvision-compatible CIFAR-100 root, e.g. `/tmp/cifar100/cifar-100-python/` | Optional offline/cache helper for CIFAR-100 when the public torchvision download endpoint is unavailable. |
| `full_scale_experiment.py` | ImageNet-100 training data and training configuration | ImageNet-100 model checkpoints | Defines the ViT model and PE variants, trains ImageNet-100 models, and provides shared helper functions imported by the n=6 analysis scripts. |
| `cifar100_experiment.py` | CIFAR-100 through torchvision and training configuration | CIFAR-100 model checkpoints | Trains CIFAR-100 models for the four PE variants. The old attack path is not the active reproducibility path; use `scripts/full_reanalysis_n6.py` for attacks. |
| `scripts/full_reanalysis_n6.py` | Model checkpoints, ImageNet-100 or CIFAR-100 validation data, seed list | `results/imagenet_results.json`, `results/cifar_results.json` | Runs the main n=6 PE-space attack sweep: FGSM-PE, PGD-PE, and VTA across epsilon values and seeds. |
| `scripts/extract_tables_data_n6.py` | ImageNet-100 checkpoints and ImageNet-100 validation data | `results/analysis_data.json` | Computes ImageNet-100 random Gaussian PE-noise ablation and auxiliary probe summaries used for the robustness-decoupling comparison. |
| `scripts/experiment2_alibi_ablation_n6.py` | ALiBi checkpoints and validation data | `results/imagenet_alibi_ablation.json`, `results/cifar_alibi_ablation.json` | Computes the ALiBi structural ablation: slopes-only attack, relative-distance-only attack, and joint attack. |
| `scripts/experiment3_attention_metrics_v3_n6.py` | Checkpoints and validation data | `results/imagenet_spatial_metrics.json`, `results/cifar_spatial_metrics.json` | Computes attention reorganization metrics under attack, including concentration ratio, mean attention distance, and top-1 overlap. |
| `scripts/experiment3_perturbation_norm_v4_n6.py` | Checkpoints and validation data | `results/imagenet_perturbation_norms.json`, `results/cifar_perturbation_norms.json` | Computes PGD perturbation-norm trajectories and saturation summaries such as `frac@ceil`. |
| `scripts/generate_figures_n6.py` | All required n=6 JSON files in `results/` | `figures/*.pdf`, `figures/*.png`, or both | Regenerates all main-paper and supplementary figures from the included n=6 result files. |
| `scripts/extract_paper_numbers_n6.py` | All required n=6 JSON files in `results/` | `results/paper_numbers_n6.json`, `results/paper_numbers_n6.md` | Recomputes paper-level numeric summaries from the JSON files: clean accuracies, `epsilon_crit`, `epsilon_star`, ALiBi ablation values, random-noise summaries, spatial metrics, and perturbation/saturation summaries. |
| `scripts/colab_quickstart_n6.ipynb` | Repository files, model checkpoints, dataset paths, Drive paths | Full n=6 pipeline outputs | Colab orchestration notebook for rerunning the n=6 workflow and regenerating JSON results and figures. |

---

## Included result files

The `results/` folder contains all aggregate n=6 JSON files required to reproduce the figures.

| File | Description |
| --- | --- |
| `imagenet_results.json` | ImageNet-100 FGSM-PE, PGD-PE, and VTA results |
| `cifar_results.json` | CIFAR-100 FGSM-PE, PGD-PE, and VTA results |
| `analysis_data.json` | ImageNet-100 random-noise PE ablation and auxiliary probes |
| `imagenet_alibi_ablation.json` | ImageNet-100 ALiBi slopes / relative-distance ablation |
| `cifar_alibi_ablation.json` | CIFAR-100 ALiBi slopes / relative-distance ablation |
| `imagenet_spatial_metrics.json` | ImageNet-100 attention reorganization and spatial metrics |
| `cifar_spatial_metrics.json` | CIFAR-100 attention reorganization and spatial metrics |
| `imagenet_perturbation_norms.json` | ImageNet-100 perturbation norm and saturation data |
| `cifar_perturbation_norms.json` | CIFAR-100 perturbation norm and saturation data |

These JSON files are small enough to be versioned directly in Git.

---

## Trained models

Model checkpoints are not stored directly in this repository.

To rerun the attacks from scratch, provide checkpoint folders with this structure:

```text
<models_dir>/
├── learned_seed42/best_model.pth
├── learned_seed123/best_model.pth
├── learned_seed456/best_model.pth
├── learned_seed789/best_model.pth
├── learned_seed1011/best_model.pth
├── learned_seed1213/best_model.pth
├── sinusoidal_seed42/best_model.pth
├── ...
└── alibi_seed1213/best_model.pth
```

The scripts expect the same structure for both ImageNet-100 and CIFAR-100 model directories.

Do not commit model checkpoints to this repository. Use external storage such as Google Drive, GitHub Releases, or Zenodo for large checkpoint files.

---

## Reproducing the figures from included JSON files

The fastest reproducibility path is to regenerate the figures from the included n=6 JSON files:

```bash
python scripts/generate_figures_n6.py \
    --imagenet           results/imagenet_results.json \
    --cifar              results/cifar_results.json \
    --imagenet-analysis  results/analysis_data.json \
    --imagenet-ablation  results/imagenet_alibi_ablation.json \
    --cifar-ablation     results/cifar_alibi_ablation.json \
    --imagenet-spatial   results/imagenet_spatial_metrics.json \
    --cifar-spatial      results/cifar_spatial_metrics.json \
    --imagenet-norms     results/imagenet_perturbation_norms.json \
    --cifar-norms        results/cifar_perturbation_norms.json \
    --outdir             figures \
    --format             pdf
```

Use `--format png` for raster output or `--format both` to emit both PDF and PNG.

The `--imagenet-analysis` argument should be provided so that the random-noise panel in Fig. 2 is generated from the n=6 `analysis_data.json`.

---

## Recomputing paper-level numbers from JSON files

To audit the numeric values reported in the paper and README, run:

```bash
python scripts/extract_paper_numbers_n6.py \
    --results_dir results \
    --out_json results/paper_numbers_n6.json \
    --out_md results/paper_numbers_n6.md
```

The script reads the included n=6 JSON files and recomputes clean-accuracy summaries, PGD-PE `epsilon_crit`, peak-degradation `epsilon_star`, ALiBi ablation values, random-noise summaries, spatial attention metrics, and perturbation/saturation summaries.

The generated `paper_numbers_n6.json` and `paper_numbers_n6.md` files are derivative audit outputs; they can be regenerated from the committed `results/*.json` files.

---

## Full manual workflow

The included JSON files already contain the n=6 aggregate results. The commands below are only needed if you want to rerun the GPU-heavy experiments from model checkpoints.

### 1. Prepare ImageNet-100

```bash
python 00_setup_imagenet.py \
    --tar_path     "/path/to/ILSVRC2012_img_val.tar" \
    --labels_path  "data/val_labels.txt" \
    --classes_path "data/imagenet100_classes.txt" \
    --output_dir   "/content/imagenet100"
```

The attack scripts expect an ImageFolder-compatible directory:

```text
/content/imagenet100/val/<class_name>/*.JPEG
```

CIFAR-100 usually requires no manual dataset preparation because the scripts use
`torchvision.datasets.CIFAR100(download=True)` where applicable. If the public
torchvision download endpoint is unavailable, use the optional helper:

```bash
python 00_setup_cifar100.py \
    --source_dir  "/content/drive/MyDrive/cifar100_data/cifar-100-python" \
    --output_root "/tmp/cifar100"
```

Then pass `--val_dir "/tmp/cifar100"` to CIFAR-100 scripts that accept `--val_dir`.

### 2. Run the main n=6 attack sweep

```bash
python scripts/full_reanalysis_n6.py \
    --models_dir  "/path/to/Trained models_ImageNet100" \
    --val_dir     "/content/imagenet100/val" \
    --output_path "results/imagenet_results.json" \
    --dataset     imagenet \
    --seeds       42 123 456 789 1011 1213

python scripts/full_reanalysis_n6.py \
    --models_dir  "/path/to/Trained models_CIFAR100" \
    --output_path "results/cifar_results.json" \
    --dataset     cifar \
    --seeds       42 123 456 789 1011 1213
```

### 3. Run ImageNet-100 random-noise ablation

```bash
python scripts/extract_tables_data_n6.py \
    --models_dir  "/path/to/Trained models_ImageNet100" \
    --val_dir     "/content/imagenet100/val" \
    --output_path "results/analysis_data.json" \
    --seeds       42 123 456 789 1011 1213
```

### 4. Run ALiBi structural ablation

```bash
python scripts/experiment2_alibi_ablation_n6.py \
    --models_dir  "/path/to/Trained models_ImageNet100" \
    --val_dir     "/content/imagenet100/val" \
    --output_path "results/imagenet_alibi_ablation.json" \
    --dataset     imagenet \
    --seeds       42 123 456 789 1011 1213

python scripts/experiment2_alibi_ablation_n6.py \
    --models_dir  "/path/to/Trained models_CIFAR100" \
    --output_path "results/cifar_alibi_ablation.json" \
    --dataset     cifar \
    --seeds       42 123 456 789 1011 1213
```

### 5. Run spatial attention metrics

```bash
python scripts/experiment3_attention_metrics_v3_n6.py \
    --models_dir  "/path/to/Trained models_ImageNet100" \
    --val_dir     "/content/imagenet100/val" \
    --output_path "results/imagenet_spatial_metrics.json" \
    --dataset     imagenet \
    --batch_size  128 \
    --seeds       42 123 456 789 1011 1213

python scripts/experiment3_attention_metrics_v3_n6.py \
    --models_dir  "/path/to/Trained models_CIFAR100" \
    --output_path "results/cifar_spatial_metrics.json" \
    --dataset     cifar \
    --batch_size  128 \
    --seeds       42 123 456 789 1011 1213
```

### 6. Run perturbation-norm / saturation logging

```bash
python scripts/experiment3_perturbation_norm_v4_n6.py \
    --models_dir  "/path/to/Trained models_ImageNet100" \
    --val_dir     "/content/imagenet100/val" \
    --output_path "results/imagenet_perturbation_norms.json" \
    --dataset     imagenet \
    --batch_size  128 \
    --seeds       42 123 456 789 1011 1213

python scripts/experiment3_perturbation_norm_v4_n6.py \
    --models_dir  "/path/to/Trained models_CIFAR100" \
    --output_path "results/cifar_perturbation_norms.json" \
    --dataset     cifar \
    --batch_size  128 \
    --seeds       42 123 456 789 1011 1213
```

---

## Key n=6 results

### Clean accuracy

Mean ± standard deviation over six seeds.

| PE | ImageNet-100 | CIFAR-100 |
| --- | ---: | ---: |
| Learned | 79.46 ± 0.51 | 67.92 ± 0.46 |
| Sinusoidal | 81.62 ± 0.41 | 67.22 ± 0.47 |
| RoPE | 84.56 ± 0.30 | 73.18 ± 0.19 |
| ALiBi | 81.01 ± 0.36 | 67.76 ± 0.30 |

### Critical epsilon under PGD-PE

`εcrit` is the interpolated PGD-PE perturbation budget at which accuracy falls below 50% of the clean baseline.

| PE | ImageNet-100 εcrit | CIFAR-100 εcrit |
| --- | ---: | ---: |
| Learned | 0.13 | 0.03 |
| Sinusoidal | 0.14 | 0.10 |
| RoPE | 0.34 | 0.30 |
| ALiBi | 0.04 | 0.09 |

### Peak degradation location

`ε*` is the geometric midpoint of the epsilon interval with the peak degradation rate in the PGD-PE curve.

| PE | ImageNet-100 ε* | CIFAR-100 ε* |
| --- | ---: | ---: |
| Learned | 0.14 | 0.02 |
| Sinusoidal | 0.14 | 0.14 |
| RoPE | 0.32 | 0.32 |
| ALiBi | 0.07 | 0.07 |

### ALiBi structural ablation

At ImageNet-100, ε = 0.1:

| ALiBi attack regime | Accuracy |
| --- | ---: |
| slopes only | 16.7% |
| relative distance only | 80.9% |
| slopes + relative distance | 16.5% |

The ALiBi vulnerability is concentrated in the per-head slope parameters; adding relative-distance perturbations does not substantially increase damage beyond the slopes-only attack.

---

## Notes on interpretation

- The n=6 results support a robustness-decoupling view: random Gaussian PE noise and adversarial PE-space perturbations expose different vulnerability mechanisms.
- RoPE has the highest adversarial threshold in the tested PGD-PE setting, but it is not immune; it collapses at sufficiently large epsilon.
- ALiBi is highly sensitive because a very small slope parameter space controls a global distance bias.
- VTA is included as a diagnostic coordinate-weighted attack. It should not be interpreted as an optimized or strictly stronger alternative to PGD-PE.
- PGD-PE is generally the strongest iterative attack family, but ALiBi can show regimes where one-step FGSM-PE is comparable or stronger because the slopes-only attack space is extremely low-dimensional.

---

## Paper and citation

The manuscript is under resubmission to IEEE Transactions on Information Forensics and Security. Citation details will be added when the resubmission status is finalized.

For now, please cite this repository if you use the code, results, or figures.

---

## License

This repository uses a dual-licensing scheme:

- **Source code** is released under the MIT License. See [`LICENSE`](LICENSE).
- **Aggregate result files, generated figures, and documentation** are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
- **Model checkpoints**, if obtained from external storage, are separate derivative artifacts and may be subject to upstream dataset restrictions.
- **ImageNet validation images** are not redistributed here and remain governed by the ImageNet terms of access.

If you use the code, cite the repository under the MIT-licensed code terms. If you use the results or figures, provide attribution under CC BY 4.0.
