
# Adversarial Vulnerability of Positional Encoding in Vision Transformers

This repository contains the code and experimental data for the paper:

> **Adversarial Vulnerability of Positional Encoding in Vision Transformers: A Targeted Attack Analysis**
> 
> *Submitted to IEEE Transactions on Information Forensics and Security (TIFS), 2026*

## 🚀 Key Findings

- **Robustness Inversion**: Learned PE (most robust to random noise) is catastrophically vulnerable to adversarial attacks — PGD-PE at $\epsilon=0.2$ reduces accuracy from **79.4% to 2.3%**.
- **RoPE Immunity**: RoPE retains **81.4%** accuracy even at $\epsilon=1.0$ (only 3.2pp loss) due to its rotational operation.
- **Novel VTA Attack**: Variance-Targeted Attack achieves up to **4.9× gain** in attack efficiency on specific PE types.
- **Resolution effect**: Learned PE is 4× more vulnerable on lower-resolution inputs (64 patches vs 196)
- **Forensic Implications**: PE tampering requires modifying **<0.2%** of model parameters, enabling stealthy supply chain attacks.

### Vulnerability Hierarchy (Identical on both datasets)
`Most vulnerable ← Learned ≫ Sinusoidal ≫ ALiBi ≫ RoPE → Most robust`
*This is the **exact inverse** of the random noise robustness hierarchy.*

---

## 🛠️ Reproduction Steps (Google Colab)

To reproduce the results presented in the paper, we recommend using **Google Colab**.
> [!IMPORTANT]
> **Note on Local Execution**
> This repository is optimized for Google Colab. The scripts contain hardcoded absolute paths. To run this project locally, you must perform a search for these paths and update all directory-related variables to match your local environment.

### **Step 1 --- Google Drive Preparation**
- Create a directory named **`pe_experiment`** in your root Google Drive directory (**`/My Drive/pe_experiment/`**).
   * **Note:** In Colab, the full path will be: **`/content/drive/MyDrive/pe_experiment/`**.
- **From GitHub:** Download the repository and copy its entire content into the **`pe_experiment`** directory.

### **Step 2 --- Data Setup & Structure**
⚠️ **IMPORTANT:** The **`pe_experiment`** directory structure **must** be identical to the diagram below, as all script paths are hardcoded.

```text

📁 pe_experiment/
├── full_scale_experiment.py                    # ViT model definition + PE implementations
├── cifar100_experiment.py                      # CIFAR-100 models training + adversarial attacks
├── adversarial_pe_attacks.py                   # Adversarial attacks on ImageNet-100 models
├── generate_figures.py                         # Generates all figures
├── ImageNet100_START.ipynb                     # Colab script for reproducing ImageNet-100 results (outcome: adversarial_pe_results.json)
├── CIFAR100_START.ipynb                        # Colab script for reproducing CIFAR-100 results (outcome: adversarial_pe_results_cifar100.json)
├── imagenet100_classes.txt                     # 100 ImageNet class IDs (WordNet synsets)
├── val_labels.txt                              # Validation set labels
├── analysis_data.json                          # Noise ablation study data
│
│                                                                                  
├──📁 imagenet/                                 # Keep archived! 
│   ├── ILSVRC2012_img_train.tar                                                     
│   └── ILSVRC2012_img_val.tar
│ 
├──📁 results/                                  # ImageNet100 results
│   ├── adversarial_pe_results.json             <-- Generated automatically after execution
│   └──📁{pe_type}_seed{s}/                     # Per-model weights + training history
│        ├── best_model.pth
│        └── training history.json  
│                                                  
├──📁 results_cifar100/                         # CIFAR100 results
│   ├── adversarial_pe_results_cifar100.json    <-- Generated automatically after execution
│   └──📁{pe_type}_seed{s}/                     # Per-model weights + training history
│       ├── best_model.pth
│       └── training history.json
│                                
└── README.md
```

### 🗂️ Step 3: Dataset Acquisition


| Dataset | Preparation Process |
| :--- | :--- |
| **ImageNet-100** | Register at [image-net.org](https://image-net.org). Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` and place them in `/pe_experiment/imagenet/`. **Do not extract.** |
| **CIFAR-100** | **Fully Automated.** The script uses `torchvision.datasets` to fetch and prepare data automatically upon execution. |

> [!IMPORTANT]
> The `ImageNet100_START.ipynb` notebook filters exactly 100 classes on-the-fly using the `imagenet100_synsets.txt` file.

---

### 📥 Download Pre-trained Weights
1. **[ImageNet-100 Models](https://drive.google.com/drive/folders/1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9?usp=sharing)**
2. **[CIFAR-100 Models](https://drive.google.com/drive/folders/1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL?usp=sharing)**

> [!IMPORTANT]
> **Instruction:** Select all -> **"Make a copy"** -> Move copies to `/pe_experiment/results/` (or `results_cifar100/`). Ensure that each model's individual **subdirectory** is preserved and contains both `best_model.pth` and `training_history.json`.

---

## 🚀 Step 4: Execution & Configuration

1. **Open the Notebook:** Locate and open the `ImageNet100_START.ipynb` (or `CIFAR100_START.ipynb`) directly in **Google Colab**.
   > [!CAUTION]
   > **Google Colab Pro+ is REQUIRED**. Only this subscription level guarantees sufficient Colab SSD storage to handle the extraction and processing of the ImageNet-100 dataset archives.
2. **Hardware**: Navigate to **Runtime > Change runtime type** and select **GPU (H100 or A100)**. 
3. **Verify GPU (Cell 1):** Execute the first cell to confirm the runtime is configured with a selected high-performance **GPU (H100 or A100)**.
4. **Mount Google Drive (Cell 2):** Run the second cell and follow the authorization prompt to **"Connect to Google Drive"**.
5. **Sequential Execution (From Cell 3 Onwards):** Run all remaining cells **one by one** in the provided order.
   * This ensures that the environment is properly initialized (script copying) and that the specific experiment workflow for each dataset proceeds correctly.
   * **Note:** Ensure each cell finishes completely before starting the next one to maintain the correct data flow and variable states.

> [!TIP]
> **CIFAR-100 Training Bypass:**
> For the **`CIFAR100_START.ipynb`** notebook, the script features an **automatic detection logic**. If you have correctly placed the downloaded weights and logs into the **`/results_cifar100/`** subdirectories, the script will:
> * **Verify** the integrity of existing models.
> * **Skip** the time-consuming training phase.
> * **Proceed** directly to the **adversarial attack analysis** and evaluation.

## 📊 Step 5: Figure Generation

To generate all 17 figures used in the paper, copy the **`generate_figures.py`** script to the local Colab storage directory (`/content/`) and run it.
> [!IMPORTANT]
> Make sure the `analysis_data.json` file is also copied to the local Colab storage directory (`/content/`) before running the  **`generate_figures.py`** script.

---

## 📦 Architecture Summary

All models use identical ViT-Base architecture:

| Config | ImageNet-100 | CIFAR-100 |
| :--- | :---: | :---: |
| **Image size** | 224×224 | 32×32 |
| **Patch size** | 16×16 | 4×4 |
| **Num patches** | 196 | 64 |
| **Layers** | 12 | 12 |
| **Attention heads** | 12 | 12 |
| **Embedding dim** | 768 | 768 |
| **Parameters** | 85.9M | 85.9M |

Training: AdamW (lr=3×10⁻⁴, weight decay 0.1), cosine annealing, 20 warmup epochs, 300 total epochs, batch size 128, Mixup (α=0.8), label smoothing 0.1.

---

## 📊 Model Results & Weights

We provide **24 ViT-Base models** (7.6 GB total) trained from scratch (4 PE types × 3 seeds × 2 datasets).

| Dataset | PE Type | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ImageNet-100** | Learned | 79.68% | 79.90% | 78.74% | 79.44 ± 0.62% |
| | Sinusoidal | 81.84% | 81.30% | 81.24% | 81.46 ± 0.33% |
| | **RoPE** | **84.96%** | **84.18%** | **84.38%** | **84.51 ± 0.41%** |
| | ALiBi | 81.16% | 81.34% | 80.66% | 81.05 ± 0.36% |
| **CIFAR-100** | Learned | 68.72% | 68.07% | 68.04% | 68.28 ± 0.38% |
| | Sinusoidal | 67.40% | 66.41% | 66.95% | 66.92 ± 0.50% |
| | **RoPE** | **73.10%** | **73.35%** | **73.45%** | **73.30 ± 0.18%** |
| | ALiBi | 67.39% | 68.16% | 67.42% | 67.66 ± 0.44% |

---

## 🛡️ Attack Methods & Robustness Analysis

Three attack strategies evaluated at $\epsilon \in \{0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$:

| Attack | Description | Reference |
| :--- | :--- | :--- |
| **FGSM-PE** | Single-step gradient attack on PE parameters | Goodfellow et al., 2015 |
| **PGD-PE** | Multi-step (T=20) projected gradient descent on PE | Madry et al., 2018 |
| **VTA** | **Variance-Targeted Attack (ours)** — allocates perturbation budget proportionally to per-dimension PE variance | This work |

### 📊 Adversarial Attack Results (PGD-PE)

#### ImageNet-100 Accuracy (%)

| ε | Learned | Sinusoidal | RoPE | ALiBi |
| :--- | :---: | :---: | :---: | :---: |
| **0 (clean)** | 79.4% | 81.5% | **84.5%** | 81.0% |
| **0.1** | 67.7% | 77.2% | **84.1%** | 78.3% |
| **0.2** | 2.3% | 56.2% | **83.9%** | 69.3% |
| **0.5** | 1.3% | 1.2% | **83.2%** | 65.2% |
| **1.0** | 1.1% | 1.0% | **81.4%** | 28.9% |

#### CIFAR-100 Accuracy (%)

| ε | Learned | Sinusoidal | RoPE | ALiBi |
| :--- | :---: | :---: | :---: | :---: |
| **0 (clean)** | 68.3% | 66.9% | **73.3%** | 67.7% |
| **0.1** | 1.4% | 43.1% | **73.0%** | 65.3% |
| **0.2** | 1.0% | 4.3% | **72.6%** | 50.6% |
| **0.5** | 1.0% | 1.0% | **71.7%** | 35.0% |
| **1.0** | 1.0% | 1.0% | **70.3%** | 23.0% |

---
> [!IMPORTANT]
Results are reported as an average of multiple runs; minor numerical variations may occur due to hardware non-determinism during reproduction.
