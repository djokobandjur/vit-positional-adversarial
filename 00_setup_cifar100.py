#!/usr/bin/env python3
"""
00_setup_cifar100.py
====================
Optional helper for preparing a local CIFAR-100 root for Colab/offline runs.

Most CIFAR-100 experiment scripts can use torchvision.datasets.CIFAR100(
download=True) directly. This helper is intended for cases where torchvision's
public download endpoint is unavailable or where an offline/cached dataset copy
is preferred.

Place a cached `cifar-100-python` directory on Google Drive or local disk and
copy it to a local SSD root. The resulting layout should be:

    /tmp/cifar100/cifar-100-python/{meta, train, test}

Typical Colab usage:

    python 00_setup_cifar100.py \
        --source_dir "/content/drive/MyDrive/cifar100_data/cifar-100-python" \
        --output_root "/tmp/cifar100"

Then pass:

    --val_dir "/tmp/cifar100"

to CIFAR-100 experiment scripts that accept `--val_dir`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a cached CIFAR-100 python-format dataset to a local root."
    )
    parser.add_argument(
        "--source_dir",
        default="/content/drive/MyDrive/cifar100_data/cifar-100-python",
        help="Path to cached cifar-100-python directory, usually on Google Drive.",
    )
    parser.add_argument(
        "--output_root",
        default="/tmp/cifar100",
        help="Local parent directory that will contain cifar-100-python.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing output_root before copying.",
    )
    parser.add_argument(
        "--skip_torchvision_check",
        action="store_true",
        help="Skip the torchvision CIFAR100 integrity check.",
    )
    return parser.parse_args()


def require_cifar_python_dir(path: Path) -> None:
    missing = [name for name in ["meta", "train", "test"] if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{path} does not look like a CIFAR-100 python-format directory; "
            f"missing: {', '.join(missing)}"
        )


def main() -> None:
    args = parse_args()
    source = Path(args.source_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    target = output_root / "cifar-100-python"

    print("=" * 72)
    print("CIFAR-100 local setup")
    print("=" * 72)
    print(f"Source:      {source}")
    print(f"Output root: {output_root}")
    print(f"Target:      {target}")

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    require_cifar_python_dir(source)

    if output_root.exists() and args.force:
        print(f"Removing existing output root: {output_root}")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"Target already exists: {target}")
        require_cifar_python_dir(target)
    else:
        print("Copying cached CIFAR-100 directory...")
        shutil.copytree(source, target)
        require_cifar_python_dir(target)

    print("\nFiles:")
    for name in ["meta", "train", "test"]:
        p = target / name
        print(f"  {name:<5} {p.stat().st_size:,} bytes")

    if not args.skip_torchvision_check:
        print("\nRunning torchvision integrity check...")
        try:
            from torchvision.datasets import CIFAR100
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Could not import torchvision. Install requirements or rerun with "
                "--skip_torchvision_check."
            ) from exc

        ds = CIFAR100(root=str(output_root), train=False, download=False)
        print(f"CIFAR-100 test size: {len(ds)}")
        if len(ds) != 10000:
            raise RuntimeError(f"Expected 10,000 CIFAR-100 test images, got {len(ds)}")

    print("\nDataset ready.")
    print(f"Use this argument for CIFAR scripts: --val_dir \"{output_root}\"")


if __name__ == "__main__":
    main()
