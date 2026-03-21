# Adversarial Vulnerability of Positional Encoding in Vision Transformers

This repository contains the code and experimental data for the paper:

> **Adversarial Vulnerability of Positional Encoding in Vision Transformers: A Targeted Attack Analysis**
>
> Djoko Bandjur, Milos Bandjur, Branimir Jaksic
>
> *Submitted to IEEE Transactions on Information Forensics and Security (TIFS), 2026*

## Key Findings

- **Robustness Inversion**: Learned PE (most robust to random noise [1]) is catastrophically vulnerable to adversarial attack — PGD-PE at ε=0.2 reduces accuracy from 79.4% to 2.3%
- **RoPE Immunity**: RoPE retains 81.4% accuracy even at ε=1.0 (only 3.2pp loss) due to its rotational operation in attention space
- **Cross-dataset validation**: Identical vulnerability hierarchy confirmed on both ImageNet-100 and CIFAR-100
- **Novel VTA attack**: Variance-Targeted Attack achieves 3.1×–4.9× gain on ALiBi but 0.05×–0.37× on Learned PE
- **Resolution effect**: Learned PE is 4× more vulnerable on lower-resolution inputs (64 patches vs 196)

## Vulnerability Hierarchy (identical on both datasets)

```
Most vulnerable ← Learned ≫ Sinusoidal ≫ ALiBi ≫ RoPE → Most robust
```

This is the **exact inverse** of the random noise robustness hierarchy from [1].

## Repository Structure

```
├── full_scale_experiment.py          # ViT model definition + PE implementations
├── cifar100_experiment.py            # CIFAR-100 training + adversarial attacks
├── adversarial_pe_attacks.py         # Adversarial attacks on ImageNet-100 models
├── results/
│   ├── adversarial_pe_results.json   # ImageNet-100 adversarial attack results
│   └── {pe_type}_seed{s}/           # Per-model weights + training history
├── results_cifar100/
│   ├── adversarial_pe_results_cifar100.json  # CIFAR-100 adversarial results
│   └── {pe_type}_seed{s}/           # Per-model weights + training history
└── README.md
```

## Trained Models

24 ViT-Base models (4 PE types × 3 seeds × 2 datasets):

| Dataset | PE Type | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|---------|---------|---------|----------|----------|------------|
| **ImageNet-100** | Learned | 79.68% | 79.90% | 78.74% | 79.44 ± 0.49% |
| | Sinusoidal | 81.84% | 81.30% | 81.24% | 81.46 ± 0.26% |
| | RoPE | 84.96% | 84.18% | 84.38% | 84.51 ± 0.32% |
| | ALiBi | 81.16% | 81.34% | 80.66% | 81.05 ± 0.28% |
| **CIFAR-100** | Learned | 68.72% | 68.07% | 68.04% | 68.28 ± 0.31% |
| | Sinusoidal | 67.40% | 66.41% | 66.95% | 66.92 ± 0.40% |
| | RoPE | 73.10% | 73.35% | 73.45% | 73.30 ± 0.15% |
| | ALiBi | 67.39% | 68.16% | 67.42% | 67.66 ± 0.36% |

Model weights available on Google Drive:
- [ImageNet-100 models](https://drive.google.com/drive/folders/1gPwVSE0qctWVeaGwCv3eGQdQR4IK6Xds?usp=sharing) (12 models, ~3.3 GB)
- [CIFAR-100 models](https://drive.google.com/drive/folders/16pEAbdH4aRpw-3s2vm4TbMey1GPQn2FQ?usp=sharing) (12 models, ~3.3 GB)

## Architecture

All models use identical ViT-Base architecture:

| Config | ImageNet-100 | CIFAR-100 |
|--------|-------------|-----------|
| Image size | 224×224 | 32×32 |
| Patch size | 16×16 | 4×4 |
| Num patches | 196 | 64 |
| Layers | 12 | 12 |
| Attention heads | 12 | 12 |
| Embedding dim | 768 | 768 |
| Parameters | 85.9M | 85.9M |

Training: AdamW (lr=3×10⁻⁴, weight decay 0.1), cosine annealing, 20 warmup epochs, 300 total epochs, batch size 128, Mixup (α=0.8), label smoothing 0.1.

## Quick Start

### Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy
```

### Train CIFAR-100 Models + Run Adversarial Attacks

```bash
# Single script: trains 12 models then runs all 3 attacks automatically
python cifar100_experiment.py
```

### Run Adversarial Attacks on ImageNet-100 Models

```bash
python adversarial_pe_attacks.py
```

### Loading Pre-trained Models

```python
from full_scale_experiment import VisionTransformer

model = VisionTransformer(
    img_size=224, patch_size=16, num_classes=100, embed_dim=768,
    depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, pe_type='rope'
)
state = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
model.eval()
```

## Attack Methods

Three attack strategies evaluated at ε ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}:

| Attack | Description | Reference |
|--------|-------------|-----------|
| **FGSM-PE** | Single-step gradient attack on PE parameters | Goodfellow et al., 2015 |
| **PGD-PE** | Multi-step (T=20) projected gradient descent on PE | Madry et al., 2018 |
| **VTA** | Variance-Targeted Attack (ours) — allocates perturbation budget proportionally to per-dimension PE variance | This work |

## Adversarial Attack Results

### PGD-PE — ImageNet-100

| ε | Learned | Sinusoidal | RoPE | ALiBi |
|---|---------|------------|------|-------|
| 0 (clean) | 79.4 | 81.5 | 84.5 | 81.1 |
| 0.1 | 67.8 | 77.2 | 84.1 | 78.3 |
| 0.2 | **2.3** | 56.2 | **83.9** | 69.3 |
| 0.5 | 1.3 | 1.2 | 83.2 | 65.2 |
| 1.0 | 1.0 | 1.0 | **81.4** | 28.9 |

### PGD-PE — CIFAR-100

| ε | Learned | Sinusoidal | RoPE | ALiBi |
|---|---------|------------|------|-------|
| 0 (clean) | 68.3 | 66.9 | 73.3 | 67.7 |
| 0.1 | 1.4 | 43.1 | 73.0 | 65.3 |
| 0.2 | **1.0** | 4.3 | **72.6** | 50.6 |
| 0.5 | 1.0 | 1.0 | 71.7 | 35.0 |
| 1.0 | 1.0 | 1.0 | **70.3** | 23.0 |

## Dataset

- **ImageNet-100**: [100-class subset](https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt) from ILSVRC-2012 (Tian et al., ECCV 2020). Requires ILSVRC-2012 access.
- **CIFAR-100**: Downloaded automatically via `torchvision.datasets.CIFAR100`.

## Related Work

This paper builds on our information-theoretic analysis of PE strategies:

> [1] Bandjur, D., Bandjur, M., Jaksic, B. (2026). "Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers." [DOI: 10.5281/zenodo.19063156](https://doi.org/10.5281/zenodo.19063156)

## Citation

```bibtex
@article{bandjur2026adversarial,
  title={Adversarial Vulnerability of Positional Encoding in Vision 
         Transformers: A Targeted Attack Analysis},
  author={Bandjur, Djoko and Bandjur, Milos and Jaksic, Branimir},
  journal={Submitted to IEEE Transactions on Information Forensics 
           and Security},
  year={2026}
}
```

## License

This code is released for academic and research purposes only.
If you use this code in your research, please cite our paper.
