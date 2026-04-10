

# Adversarial Vulnerability of Positional Encoding in Vision Transformers


This repository contains the code and experimental data for the paper:

> **Adversarial Vulnerability of Positional Encoding in Vision Transformers: A Targeted Attack Analysis**
>>
> *Submitted to IEEE Transactions on Information Forensics and Security (TIFS), 2026*

## Key Findings

- **Robustness Inversion**: Learned PE (most robust to random noise [1]) is catastrophically vulnerable to adversarial attack — PGD-PE at ε=0.2 reduces accuracy from 79.4% to 2.3%
- **RoPE Immunity**: RoPE retains 81.4% accuracy even at ε=1.0 (only 3.2pp loss) due to its rotational operation in attention space
- **Cross-dataset validation**: Identical vulnerability hierarchy confirmed on both ImageNet-100 and CIFAR-100
- **Novel VTA attack**: Variance-Targeted Attack achieves 3.1×–4.9× gain on ALiBi but 0.05×–0.37× on Learned PE
- **Resolution effect**: Learned PE is 4× more vulnerable on lower-resolution inputs (64 patches vs 196)
- **Forensic implications**: PE tampering requires modifying <0.2% of model parameters, enabling stealthy supply chain attacks detectable through PE checksumming protocols
<br>
 
## Vulnerability Hierarchy (identical on both datasets)
```
Most vulnerable ← Learned ≫ Sinusoidal ≫ ALiBi ≫ RoPE → Most robust
```
This is the **exact inverse** of the random noise robustness hierarchy from [1].

<br>

### To reproduce the results presented in the paper, follow these steps using Google Colab (recommended)::

### **Step 1 --- Google Drive Preparation**
*   Create a folder named `pe_experiment` in your root Google Drive directory.
*   **Final Path on Drive:** **`/My Drive/pe_experiment/`**
*   **Note:** In Colab, the full path will be: **`/content/drive/MyDrive/pe_experiment/`**

### **Step 2 --- Data Setup & Structure**
⚠️ **IMPORTANT:** The folder structure must be identical to the diagram below. All script paths are hardcoded.
**From GitHub:** Download the repository and copy the following files into the root folder **`/pe_experiment/`**:

*   **Python scripts: `full_scale_experiment.py`, `cifar100_experiment.py`, `adversarial_pe_attacks.py` and `generate_figures.py`** 
*   **The Colab notebooks: `ImageNet100_START.ipynb` and `CIFAR100_START.ipynb`**

## 📥 Dataset Acquisition
This project utilizes two primary datasets, each requiring a different preparation approach.

### 🖼️ ImageNet-100
    Due to licensing restrictions and large file sizes, ImageNet requires a manual setup process:

  1. **Folder Structure:** Create a folder named `imagenet` within the **`/pe_experiment/`** directory.
  2. **Registration:** Visit [image-net.org](https://image-net.org) and register using an **academic email address**. 
  3. **Download:** Once your account is approved, use the unique download link provided in your email to acquire  **`ILSVRC2012_img_train.tar`** and **`ILSVRC2012_img_val.tar`** archives.
  4. **Placement:** Place the downloaded archives directly inside the **`/imagenet/`** folder.

> [!IMPORTANT]
> **Do NOT extract the archives.** The **`ImageNet100_START.ipynb`** notebook handles the `.tar` files automatically. It performs on-the-fly filtering to select exactly **100 classes** based on the WordNet synsets defined in the **`imagenet100_synsets.txt`** file provided in this repository.

### 🍱 CIFAR-100
    Unlike ImageNet, the CIFAR-100 setup is fully automated:

  1. **Automatic Download:** The **`cifar100_experiment.py`** script utilizes **`torchvision.datasets`** to programmatically fetch the data.
  2. **Data Storage:** The dataset will be downloaded and prepared within the directory specified by the **`DATA_DIR`** variable in the script.
  3. **Plug-and-Play:** No manual download or prior intervention is required. The script automatically handles the data integrity check, augmentation, and normalization upon execution.

<br>

## 📊 Model Results (7.6 GB Total)

24 ViT-Base models (4 PE types × 3 seeds × 2 datasets):

| Dataset | PE Type | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ImageNet-100** | Learned | 79.68% | 79.90% | 78.74% | 79.44 ± 0.49% |
| | Sinusoidal | 81.84% | 81.30% | 81.24% | 81.46 ± 0.26% |
| | RoPE | 84.96% | 84.18% | 84.38% | 84.51 ± 0.32% |
| | ALiBi | 81.16% | 81.34% | 80.66% | 81.05 ± 0.28% |
| **CIFAR-100** | Learned | 68.72% | 68.07% | 68.04% | 68.28 ± 0.31% |
| | Sinusoidal | 67.40% | 66.41% | 66.95% | 66.92 ± 0.40% |
| | RoPE | 73.10% | 73.35% | 73.45% | 73.30 ± 0.15% |
| | ALiBi | 67.39% | 68.16% | 67.42% | 67.66 ± 0.36% |


The weights and training logs for all 24 models trained from scratch are available at the links below:

* **[ImageNet Trained Models](https://drive.google.com/drive/folders/1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9?usp=sharing)**
* **[CIFAR-100 Trained Models](https://drive.google.com/drive/folders/1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL?usp=sharing)**


### Architecture

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


### 🛠️ Access & Setup Instructions
  Since the **directories** are shared with **Viewer** access, follow these steps to integrate the trained models:

  1. **Copy the Folders:** Open the links, **Select all folders**, right-click, and choose **"Make a copy"**.
  2. **Locate Copies:** The copies will appear in your Google Drive storage (typically within the main My Drive section).
  3. **Organize Subdirectories:** Move these copied **directories** into their respective project paths:\
      ImageNet results $\rightarrow$ `/pe_experiment/results/`  \
      CIFAR-100 results $\rightarrow$ `/pe_experiment/results_cifar100/`

> [!IMPORTANT]
> **Rename Directories:** Each individual model subdirectory must contain both `best_model.pth` and `training_history.json`.

### 🚀 Runtime Configuration
  To execute the experiments, navigate to **Runtime > Change runtime type** and select a high-performance **GPU (H100 or A100)**.

> [!NOTE]
> A **Colab Pro+** profile is required to ensure the virtual machine is provisioned with sufficient **local SSD storage** to handle the ImageNet-100 datasets.

### 🚀 Notebook Execution & Setup

1. **Open the Notebook:** Locate and open the `ImageNet100_START.ipynb` (or `CIFAR100_START.ipynb`) directly in **Google Colab**.
2. **Verify GPU (Cell 1):** Execute the first cell to confirm the runtime is configured with a high-performance **GPU (H100 or A100)**.
3. **Mount Google Drive (Cell 2):** Run the second cell and follow the authorization prompt to **"Connect to Google Drive"**.
4. **Sequential Execution (From Cell 3 Onwards):** Run all remaining cells **one by one** in the provided order.
   * This ensures that the environment is properly initialized (script copying) and that the specific experiment workflow for each dataset proceeds correctly.
   * **Note:** Ensure each cell finishes completely before starting the next one to maintain the correct data flow and variable states.

> [!TIP]
> **CIFAR-100 Training Bypass:**
> For the **`CIFAR100_START.ipynb`** notebook, the script features an **automatic detection logic**. If you have correctly placed the downloaded weights and logs into the **`/results_cifar100/`** subdirectories, the script will:
> * **Verify** the integrity of existing models.
> * **Skip** the time-consuming training phase.
> * **Proceed** directly to the **adversarial attack analysis** and evaluation.

## 📊 Figure Generation

To generate all 17 figures used in the paper, copy the **`generate_figures.py`** script to the local Colab storage directory (`/content/`) and run it.



## If running locally, install the dependencies:

```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy
```

### Train CIFAR-100 Models + Run Adversarial Attacks
Single script: trains 12 models then runs all 3 attacks automatically
python cifar100_experiment.py

### Run Adversarial Attacks on ImageNet-100 Models
python adversarial_pe_attacks.py

### Loading Pre-trained Models

To load a trained model with a specific Positional Encoding (e.g., **RoPE**), use the following snippet:

```python
import torch
from full_scale_experiment import VisionTransformer

# Initialize model architecture
model = VisionTransformer(
    img_size=224, 
    patch_size=16, 
    num_classes=100, 
    embed_dim=768,
    depth=12, 
    num_heads=12, 
    mlp_ratio=4.0, 
    dropout=0.1, 
    pe_type='rope' # Options: 'learned', 'sinusoidal', 'rope', 'alibi'
)

# Load weights and handle potential 'compile' prefix issues
state = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})

model.eval()
```


### Attack Methods
Three attack strategies evaluated at ε ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}:

| Attack | Description | Reference |
| :--- | :--- | :--- |
| **FGSM-PE** | Single-step gradient attack on PE parameters | Goodfellow et al., 2015 |
| **PGD-PE** | Multi-step (T=20) projected gradient descent on PE | Madry et al., 2018 |
| **VTA** | Variance-Targeted Attack (ours) — allocates perturbation budget proportionally to per-dimension PE variance | This work |


### Adversarial Attack Results

### PGD-PE — ImageNet-100

| ε | Learned | Sinusoidal | RoPE | ALiBi |
| :--- | :---: | :---: | :---: | :---: |
| **0 (clean)** | 79.4% | 81.5% | 84.5% | 81.1% |
| **0.1** | 67.8% | 77.2% | 84.1% | 78.3% |
| **0.2** | 2.3% | 56.2% | 83.9% | 69.3% |
| **0.5** | 1.3% | 1.2% | 83.2% | 65.2% |
| **1.0** | 1.0% | 1.0% | 81.4% | 28.9% |


### PGD-PE — CIFAR-100

| ε | Learned | Sinusoidal | RoPE | ALiBi |
| :--- | :---: | :---: | :---: | :---: |
| **0 (clean)** | 68.3% | 66.9% | 73.3% | 67.7% |
| **0.1** | 1.4% | 43.1% | 73.0% | 65.3% |
| **0.2** | 1.0% | 4.3% | 72.6% | 50.6% |
| **0.5** | 1.0% | 1.0% | 71.7% | 35.0% |
| **1.0** | 1.0% | 1.0% | 70.3% | 23.0% |


### Related Work
This paper builds on our information-theoretic analysis of PE strategies:

[1] Anonymous. (2026). "Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers." DOI: 10.5281/zenodo.19063156



### ✅ Verification
    Once the execution is complete, you can validate your findings by comparing the generated outputs.


## Repository Structure
```
├── full_scale_experiment.py                    # ViT model definition + PE implementations
├── cifar100_experiment.py                      # CIFAR-100 models training + adversarial attacks
├── adversarial_pe_attacks.py                   # Adversarial attacks on ImageNet-100 models
├── ImageNet100_START.ipynb                     # Colab script for reproducing ImageNet-100 results (outcome: adversarial_pe_results.json)
├── CIFAR100_START.ipynb                        # Colab script for reproducing CIFAR-100 results (outcome: adversarial_pe_results_cifar100.json)
├── imagenet100_classes.txt                     # 100 ImageNet class IDs (WordNet synsets)
├── val_labels.txt                              # Validation set labels
│      
│                                                                                   
├── imagenet/                                   # Keep archived! 
│   └── ILSVRC2012_img_train.tar                                                     
│   └── ILSVRC2012_img_val.tar
│ 
├── results/                                    # ImageNet100 results
│   ├── adversarial_pe_results.json             # ImageNet-100 adversarial attack results
│   └── {pe_type}_seed{s}/                      # Per-model weights + training history
│        └── best_model.pth
│        └── training history.json  
│                                                  
├── results_cifar100/                           # CIFAR100 results
│   ├── adversarial_pe_results_cifar100.json    # CIFAR-100 adversarial attack results 
│   └── {pe_type}_seed{s}/                      # Per-model weights + training history
│       └── best_model.pth
│       └── training history.json
│                                
└── README.md
```







# Adversarial Vulnerability of Positional Encoding in Vision Transformers

This repository contains the code and experimental data for the paper:

> **Adversarial Vulnerability of Positional Encoding in Vision Transformers: A Targeted Attack Analysis**
> 
> *Submitted to IEEE Transactions on Information Forensics and Security (TIFS), 2026*

## 🚀 Key Findings

- **Robustness Inversion**: Learned PE (most robust to random noise) is catastrophically vulnerable to adversarial attacks — PGD-PE at $\epsilon=0.2$ reduces accuracy from **79.4% to 2.3%**.
- **RoPE Immunity**: RoPE retains **81.4%** accuracy even at $\epsilon=1.0$ (only 3.2pp loss) due to its rotational operation.
- **Novel VTA Attack**: Variance-Targeted Attack achieves up to **4.9× gain** in attack efficiency on specific PE types.
- **Forensic Implications**: PE tampering requires modifying **<0.2%** of model parameters, enabling stealthy supply chain attacks.

### Vulnerability Hierarchy (Identical on both datasets)
`Most vulnerable ← Learned ≫ Sinusoidal ≫ ALiBi ≫ RoPE → Most robust`
*This is the **exact inverse** of the random noise robustness hierarchy.*

---

## 🛠️ Reproduction Steps (Google Colab)

To reproduce the results presented in the paper, we recommend using **Google Colab**.

### Step 1: Google Drive Preparation
Create a folder named `pe_experiment` in your root Drive directory. The structure **must** be identical to the diagram below as paths are hardcoded:

```text

├── full_scale_experiment.py                    # ViT model definition + PE implementations
├── cifar100_experiment.py                      # CIFAR-100 models training + adversarial attacks
├── adversarial_pe_attacks.py                   # Adversarial attacks on ImageNet-100 models
├── ImageNet100_START.ipynb                     # Colab script for reproducing ImageNet-100 results (outcome: adversarial_pe_results.json)
├── CIFAR100_START.ipynb                        # Colab script for reproducing CIFAR-100 results (outcome: adversarial_pe_results_cifar100.json)
├── imagenet100_classes.txt                     # 100 ImageNet class IDs (WordNet synsets)
├── val_labels.txt                              # Validation set labels
│      
│                                                                                   
├── imagenet/                                   # Keep archived! 
│   └── ILSVRC2012_img_train.tar                                                     
│   └── ILSVRC2012_img_val.tar
│ 
├── results/                                    # ImageNet100 results
│   ├── adversarial_pe_results.json             # ImageNet-100 adversarial attack results
│   └── {pe_type}_seed{s}/                      # Per-model weights + training history
│        └── best_model.pth
│        └── training history.json  
│                                                  
├── results_cifar100/                           # CIFAR100 results
│   ├── adversarial_pe_results_cifar100.json    # CIFAR-100 adversarial attack results 
│   └── {pe_type}_seed{s}/                      # Per-model weights + training history
│       └── best_model.pth
│       └── training history.json
│                                
└── README.md
```

### Step 2: Dataset Acquisition


| Dataset | Preparation Process |
| :--- | :--- |
| **ImageNet-100** | Register at [image-net.org](https://image-net.org). Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` and place them in `/pe_experiment/imagenet/`. **Do not extract.** |
| **CIFAR-100** | **Fully Automated.** The script uses `torchvision.datasets` to fetch and prepare data automatically upon execution. |

> [!IMPORTANT]
> The `ImageNet100_START.ipynb` notebook filters exactly 100 classes on-the-fly using the `imagenet100_synsets.txt` file.

---

## 📊 Model Results & Weights

We provide **24 ViT-Base models** (7.6 GB total) trained from scratch (4 PE types × 3 seeds × 2 datasets).


| Dataset | PE Type | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ImageNet-100** | Learned | 79.68% | 79.90% | 78.74% | 79.44 ± 0.49% |
| | Sinusoidal | 81.84% | 81.30% | 81.24% | 81.46 ± 0.26% |
| | **RoPE** | **84.96%** | **84.18%** | **84.38%** | **84.51 ± 0.32%** |
| | ALiBi | 81.16% | 81.34% | 80.66% | 81.05 ± 0.28% |
| **CIFAR-100** | Learned | 68.72% | 68.07% | 68.04% | 68.28 ± 0.31% |
| | Sinusoidal | 67.40% | 66.41% | 66.95% | 66.92 ± 0.40% |
| | **RoPE** | **73.10%** | **73.35%** | **73.45%** | **73.30 ± 0.15%** |
| | ALiBi | 67.39% | 68.16% | 67.42% | 67.66 ± 0.36% |

### 📥 Download Pre-trained Weights
1. **[ImageNet Models](https://google.com)**
2. **[CIFAR-100 Models](https://google.com)**

> [!IMPORTANT]
> **Instruction:** Select all -> **"Make a copy"** -> Move copies to `/pe_experiment/results/` (or `results_cifar100/`). Ensure that each model's individual **subdirectory** is preserved and contains both `best_model.pth` and `training_history.json`.
`.*

---

## 🚀 Execution & Configuration

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

## 📊 Figure Generation

To generate all 17 figures used in the paper, copy the **`generate_figures.py`** script to the local Colab storage directory (`/content/`) and run it.

---

### Architecture Summary

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
| **0 (clean)** | 79.4% | 81.5% | **84.5%** | 81.1% |
| **0.1** | 67.8% | 77.2% | **84.1%** | 78.3% |
| **0.2** | 2.3% | 56.2% | **83.9%** | 69.3% |
| **0.5** | 1.3% | 1.2% | **83.2%** | 65.2% |
| **1.0** | 1.0% | 1.0% | **81.4%** | 28.9% |

#### CIFAR-100 Accuracy (%)

| ε | Learned | Sinusoidal | RoPE | ALiBi |
| :--- | :---: | :---: | :---: | :---: |
| **0 (clean)** | 68.3% | 66.9% | **73.3%** | 67.7% |
| **0.1** | 1.4% | 43.1% | **73.0%** | 65.3% |
| **0.2** | 1.0% | 4.3% | **72.6%** | 50.6% |
| **0.5** | 1.0% | 1.0% | **71.7%** | 35.0% |
| **1.0** | 1.0% | 1.0% | **70.3%** | 23.0% |

---

## 📚 Related Work
This paper builds on our information-theoretic analysis of PE strategies:

[1] Anonymous. (2026). "Information-Theoretic Analysis of Positional Encoding Strategies in Vision Transformers." DOI: XXXXXXXXXX)










