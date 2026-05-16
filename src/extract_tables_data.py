"""
Run this in Colab after mounting Drive and copying the script.
Extracts noise ablation and probe analysis data for all 12 models.
Saves results to JSON on Drive for paper tables.
Requires GPU and imagenet100_resized dataset on SSD.
"""

import os, json, sys, torch, numpy as np
sys.path.insert(0, '/content')

# Import from our script
from full_scale_experiment import (
    VisionTransformer, extract_positional_embedding,
    probe_analysis, noise_ablation
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

RESULTS = '/content/drive/My Drive/pe_experiment/results'
DATA_DIR = '/content/imagenet100_resized'

# Validation loader
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
print(f"Val: {len(val_dataset)} images")

pe_types = ['learned', 'sinusoidal', 'rope', 'alibi']
seeds = [42, 123, 456]
all_results = {}

for pe_type in pe_types:
    all_results[pe_type] = {}
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Processing: {pe_type} seed={seed}")
        print(f"{'='*50}")
        
        # Load model
        model_path = os.path.join(RESULTS, f'{pe_type}_seed{seed}', 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"  SKIP: {model_path} not found")
            continue
        
        torch.manual_seed(seed)
        model = VisionTransformer(
            img_size=224, patch_size=16, num_classes=100, embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, pe_type=pe_type
        ).to(device)
        
        state = torch.load(model_path, map_location=device)
        model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
        model.eval()
        
        result = {}
        
        # 1. Noise ablation
        print("  Running noise ablation...")
        noise_results = noise_ablation(model, val_loader, device, pe_type)
        result['noise_ablation'] = noise_results
        for level, acc in noise_results.items():
            print(f"    {level}: {acc:.2f}%")
        
        # 2. Probe analysis
        print("  Running probe analysis...")
        pe_matrix = extract_positional_embedding(model, pe_type)
        if pe_matrix is not None:
            probe_results = probe_analysis(pe_matrix, num_patches_per_side=14)
            result['probe'] = probe_results
            for task, vals in probe_results.items():
                print(f"    Probe {task}: {vals['mean']:.1f}% +/- {vals['std']:.1f}%")
        
        all_results[pe_type][seed] = result
        
        # Save incrementally
        save_path = os.path.join(RESULTS, 'analysis_data.json')
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved to {save_path}")

print("\n" + "="*50)
print("DONE! All results saved to analysis_data.json")
print("="*50)

# Print summary tables
print("\n\nNOISE ABLATION SUMMARY (for Table 2):")
print("-" * 80)
noise_levels = ['0.0x', '0.1x', '0.2x', '0.5x', '1.0x', '2.0x', '3.0x', '5.0x', 'no_pe']
header = f"{'Level':<10}" + "".join(f"{pe:<15}" for pe in pe_types)
print(header)
for level in noise_levels:
    row = f"{level:<10}"
    for pe in pe_types:
        vals = []
        for s in seeds:
            if s in all_results.get(pe, {}) and 'noise_ablation' in all_results[pe][s]:
                na = all_results[pe][s]['noise_ablation']
                # Find matching key
                for k, v in na.items():
                    if level.replace('x', '') in k or (level == 'no_pe' and 'without' in k.lower()):
                        vals.append(v)
                        break
        if vals:
            row += f"{np.mean(vals):.2f}±{np.std(vals):.2f}  "
        else:
            row += f"{'N/A':<15}"
    print(row)

print("\n\nPROBE ANALYSIS SUMMARY (for Table 3):")
print("-" * 60)
for pe in pe_types:
    for task in ['row', 'column', 'position']:
        vals = []
        for s in seeds:
            if s in all_results.get(pe, {}) and 'probe' in all_results[pe][s]:
                vals.append(all_results[pe][s]['probe'][task]['mean'])
        if vals:
            print(f"  {pe:12s} {task:10s}: {np.mean(vals):.1f} ± {np.std(vals):.1f}%")
