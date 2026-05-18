"""
00_setup_imagenet.py
====================
Prepares the ImageNet-100 validation set on the Colab local SSD.

What this script does:
  1. Reads the 100-class split from imagenet100_classes.txt
  2. Reads val image-to-synset labels from val_labels.txt
  3. Extracts only the 5,000 relevant images from ILSVRC2012_img_val.tar
  4. Places them in /content/imagenet100/val/{synset_id}/*.JPEG

Prerequisites:
  - ILSVRC2012_img_val.tar must be on your Google Drive
    (obtain from https://image-net.org, requires academic registration)
  - val_labels.txt must be on your Google Drive
    (maps each val image to its synset ID, 50,000 lines)
  - imagenet100_classes.txt must be on your Google Drive
    (100 synset IDs from Tian et al., ECCV 2020)

Dataset split:
  Tian, Y., Krishnan, D., Isola, P.
  "Contrastive Multiview Coding", ECCV 2020.
  https://github.com/HobbitLong/CMC

Output:
  /content/imagenet100/val/{synset_id}/*.JPEG
  (100 folders, 50 images each = 5,000 images total)

Usage in Colab:
  %run /content/00_setup_imagenet.py

  # Or with custom paths:
  %run /content/00_setup_imagenet.py \
      --tar_path "/content/drive/MyDrive/pe_experiment/imagenet/ILSVRC2012_img_val.tar" \
      --labels_path "/content/drive/MyDrive/pe_experiment/val_labels.txt" \
      --classes_path "/content/drive/MyDrive/pe_experiment/imagenet100_classes.txt" \
      --output_dir "/content/imagenet100"
"""

import os
import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm


#  Default paths (adjust if your Drive structure differs) 
DRIVE_ROOT     = '/content/drive/My Drive/pe_experiment'
DEFAULT_TAR    = '/content/drive/My Drive/pe_experiment/imagenet/ILSVRC2012_img_val.tar'
DEFAULT_LABELS = os.path.join(DRIVE_ROOT, 'val_labels.txt')
DEFAULT_CLASSES= os.path.join(DRIVE_ROOT, 'imagenet100_classes.txt')
DEFAULT_OUTPUT = '/content/imagenet100'

# Argument parser 
parser = argparse.ArgumentParser(
    description='Prepare ImageNet-100 validation set on Colab SSD'
)
parser.add_argument('--tar_path',    default=DEFAULT_TAR,
                    help='Path to ILSVRC2012_img_val.tar on Google Drive')
parser.add_argument('--labels_path', default=DEFAULT_LABELS,
                    help='Path to val_labels.txt (50,000 lines, one synset per image)')
parser.add_argument('--classes_path',default=DEFAULT_CLASSES,
                    help='Path to imagenet100_classes.txt (100 synset IDs)')
parser.add_argument('--output_dir',  default=DEFAULT_OUTPUT,
                    help='Output directory on local SSD')
args, _ = parser.parse_known_args()  # parse_known_args avoids Jupyter kernel argument conflicts

# Auto-download val_labels.txt if not present 
VAL_LABELS_URL = (
    'https://raw.githubusercontent.com/tensorflow/models/master/'
    'research/slim/datasets/imagenet_2012_validation_synset_labels.txt'
)

if not os.path.exists(args.labels_path):
    print(f"val_labels.txt not found at {args.labels_path}")
    print(f"Downloading from TensorFlow Models repository...")
    import urllib.request
    os.makedirs(os.path.dirname(args.labels_path), exist_ok=True)
    urllib.request.urlretrieve(VAL_LABELS_URL, args.labels_path)
    print(f"✓ Downloaded: {args.labels_path}")

# Validation 
print("=" * 60)
print("ImageNet-100 Setup")
print("=" * 60)

for path, name in [(args.tar_path,    'ILSVRC2012_img_val.tar'),
                   (args.classes_path,'imagenet100_classes.txt')]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] {name} not found at:\n  {path}\n"
            f"Please update the path and re-run."
        )
    print(f" Found: {name}")

print(f" Found: val_labels.txt at {args.labels_path}")

# Load 100-class split 
with open(args.classes_path) as f:
    classes = set(line.strip() for line in f if line.strip())

print(f"\nImageNet-100 classes: {len(classes)}")
assert len(classes) == 100, f"Expected 100 classes, got {len(classes)}"

# Load val labels (image index → synset) 
with open(args.labels_path) as f:
    val_labels = [line.strip() for line in f.readlines()]

print(f"Val labels loaded: {len(val_labels)} entries")
assert len(val_labels) == 50000, \
    f"Expected 50,000 val labels, got {len(val_labels)}"

# Create output directories 
val_dir = os.path.join(args.output_dir, 'val')
for synset in classes:
    os.makedirs(os.path.join(val_dir, synset), exist_ok=True)

print(f"Output directory: {val_dir}")
print(f"Created {len(classes)} class folders")

# Extract relevant images from tar 
print(f"\nExtracting from: {args.tar_path}")
print("(This may take 5-10 minutes depending on Drive speed...)\n")

extracted = 0
skipped   = 0

with tarfile.open(args.tar_path, 'r') as tar:
    members = tar.getmembers()
    print(f"Total images in tar: {len(members):,}")

    for member in tqdm(members, desc="Filtering", unit="img"):
        # Filename format: ILSVRC2012_val_00000001.JPEG
        stem  = Path(member.name).stem
        try:
            idx = int(stem.split('_')[-1]) - 1   # 0-indexed
        except ValueError:
            skipped += 1
            continue

        if idx >= len(val_labels):
            skipped += 1
            continue

        synset = val_labels[idx]

        if synset not in classes:
            skipped += 1
            continue

        # Extract to correct class folder
        dst_path = os.path.join(val_dir, synset, member.name)

        fobj = tar.extractfile(member)
        if fobj is not None:
            with open(dst_path, 'wb') as out:
                out.write(fobj.read())
            extracted += 1

# Verification 
print(f"\n{'=' * 60}")
print(f"Extraction complete!")
print(f"  Images extracted: {extracted:,}")
print(f"  Images skipped:   {skipped:,}")
print(f"  Expected:         5,000")

if extracted != 5000:
    print(f"\n[WARNING] Expected 5,000 images but got {extracted}.")
    print("  Check that val_labels.txt matches ILSVRC2012 val set.")
else:
    print(f"\n All 5,000 images extracted successfully.")

# Per-class verification
print(f"\nPer-class image count (first 5):")
class_counts = {}
for synset in sorted(classes):
    folder = os.path.join(val_dir, synset)
    count  = len(os.listdir(folder))
    class_counts[synset] = count

for synset, count in list(class_counts.items())[:5]:
    print(f"  {synset}: {count} images")

min_count = min(class_counts.values())
max_count = max(class_counts.values())
print(f"\n  Min images per class: {min_count}")
print(f"  Max images per class: {max_count}")

if min_count == max_count == 50:
    print(f"All classes have exactly 50 images.")
else:
    print(f"  [WARNING] Unequal class sizes — check val_labels.txt")

print(f"\nDataset ready at: {val_dir}")
