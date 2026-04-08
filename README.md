

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

 
## Vulnerability Hierarchy (identical on both datasets)
```
Most vulnerable ← Learned ≫ Sinusoidal ≫ ALiBi ≫ RoPE → Most robust
```
This is the **exact inverse** of the random noise robustness hierarchy from [1].


### In order to reproduce results undertake the following steps:

### **Step 1 --- Google Drive Preparation**
*   Create a folder named `pe_experiment` in your root Google Drive directory.
*   **Final Path on Drive:** **`/My Drive/pe_experiment/`**
*   **Note:** In Colab, the full path will be: **`/content/drive/MyDrive/pe_experiment/`**

### **Step 2 --- Data Setup & Structure**
⚠️ **IMPORTANT:** The folder structure must be identical to the diagram below. All script paths are hardcoded.
**From GitHub:** Download the repository and copy the following files into the root folder **`/pe_experiment/`**.

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

## 📊 Model Results (7.6 GB Total)

The weights and training logs for all 24 models trained from scratch are available at the links below:

* **[ImageNet Trained Models](https://drive.google.com/drive/folders/1WRhjaR3WZHIi2fTi9xcrIBJkBXZddMM9?usp=sharing)**
* **[CIFAR-100 Trained Models](https://drive.google.com/drive/folders/1HBiOjNfuRsh2H0ZGRP4rIdBeydedCBJL?usp=sharing)**


### 🛠️ Access & Setup Instructions
  Because these folders are shared with **Viewer** access, follow these steps to integrate them into your environment:

### 🛠️ Access & Setup Instructions
  Since the **directories** are shared with **Viewer** access, follow these steps to integrate the trained models:

  1. **Copy the Folders:** Open the links, **Select all folders**, right-click, and choose **"Make a copy"**.
  2. **Locate Copies:** The copies will appear in your Google Drive storage (typically within the main My Drive section).
  3. **Organize Subdirectories:** Move these copied **directories** into their respective project paths:
    * ImageNet results $\rightarrow$ `/results/`
    * CIFAR-100 results $\rightarrow$ `/results_cifar100/`

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

To generate all 17 figures used in the paper, follow these steps:

1. **Mount Google Drive:** Ensure your Google Drive is mounted within the Colab environment so the script can access the dataset results (json files).
2. **Setup Script:** Copy the **`generate_figures.py`** script to the local Colab storage directory (`/content/`).
3. **Execution:** Run the script using the following command:

```markdown
```text
!python /content/generate_figures.py
```

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
│   └── ILSVRC2012_img_val.tar                                                     
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
