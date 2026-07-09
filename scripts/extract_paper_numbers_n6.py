#!/usr/bin/env python3
"""
extract_paper_numbers_n6.py
===========================
Summarize the n=6 quantitative results used in the TIFS manuscript from the
aggregate JSON files shipped in this repository.

This is a lightweight audit/reproducibility helper: it does not rerun attacks
or redraw figures. It recomputes the main paper-level numbers directly from
`results/*.json` and writes both a machine-readable JSON summary and a Markdown
report.

Typical usage from the repository root:

    python scripts/extract_paper_numbers_n6.py \
        --results_dir results \
        --out_json results/paper_numbers_n6.json \
        --out_md results/paper_numbers_n6.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PE_ORDER = ["learned", "sinusoidal", "rope", "alibi"]
ATTACKS = ["fgsm_pe", "pgd_pe", "vta"]
DATASETS = ["imagenet", "cifar"]
DATASET_LABEL = {"imagenet": "ImageNet-100", "cifar": "CIFAR-100"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract n=6 paper-level numeric summaries from result JSON files."
    )
    p.add_argument("--results_dir", default="results", help="Directory with the 9 n=6 JSON result files.")
    p.add_argument("--out_json", default="results/paper_numbers_n6.json", help="Output JSON summary path.")
    p.add_argument("--out_md", default="results/paper_numbers_n6.md", help="Output Markdown summary path.")
    p.add_argument("--target_fraction", type=float, default=0.5, help="Critical-epsilon threshold as a fraction of clean accuracy.")
    p.add_argument("--iso_accuracy", type=float, default=40.0, help="Iso-accuracy target used for spatial concentration summaries.")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all(results_dir: Path) -> Dict[str, Any]:
    files = {
        "imagenet_results": "imagenet_results.json",
        "cifar_results": "cifar_results.json",
        "analysis_data": "analysis_data.json",
        "imagenet_alibi_ablation": "imagenet_alibi_ablation.json",
        "cifar_alibi_ablation": "cifar_alibi_ablation.json",
        "imagenet_spatial_metrics": "imagenet_spatial_metrics.json",
        "cifar_spatial_metrics": "cifar_spatial_metrics.json",
        "imagenet_perturbation_norms": "imagenet_perturbation_norms.json",
        "cifar_perturbation_norms": "cifar_perturbation_norms.json",
    }
    out = {}
    missing = []
    for key, name in files.items():
        path = results_dir / name
        if not path.exists():
            missing.append(str(path))
        else:
            out[key] = load_json(path)
    if missing:
        raise FileNotFoundError("Missing result JSON files:\n" + "\n".join(missing))
    return out


def sorted_seed_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    def key(item: Tuple[str, Any]) -> int:
        try:
            return int(item[0])
        except Exception:
            return 10**9
    return sorted(d.items(), key=key)


def sample_mean_std(vals: Iterable[float]) -> Dict[str, float]:
    arr = np.array([float(v) for v in vals if v is not None and not math.isnan(float(v))], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def common_float_keys(dicts: List[Dict[str, Any]], positive_only: bool = False) -> List[float]:
    common = None
    for d in dicts:
        vals = {float(k) for k in d.keys() if (not positive_only or float(k) > 0)}
        common = vals if common is None else common & vals
    return sorted(common or [])


def key_for_float(d: Dict[str, Any], value: float) -> str:
    for k in d.keys():
        if float(k) == float(value):
            return k
    raise KeyError(value)


def curve_matrix(results: Dict[str, Any], pe: str, attack: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    pe_data = results["results"][pe]
    seed_items = [
        (seed, data) for seed, data in sorted_seed_items(pe_data)
        if data.get("status") == "ok" and "clean_acc" in data and attack in data.get("attacks", {})
    ]
    if not seed_items:
        raise ValueError(f"No completed seeds for {pe}/{attack}")
    clean = np.array([data["clean_acc"] for _, data in seed_items], dtype=float)
    attack_dicts = [data["attacks"][attack] for _, data in seed_items]
    eps = common_float_keys(attack_dicts)
    mat = np.full((len(eps), len(seed_items)), np.nan, dtype=float)
    for i, e in enumerate(eps):
        for j, (_, data) in enumerate(seed_items):
            ad = data["attacks"][attack]
            mat[i, j] = float(ad[key_for_float(ad, e)]["accuracy"])
    eps_with_clean = np.array([0.0] + eps, dtype=float)
    mat_with_clean = np.vstack([clean[None, :], mat])
    return eps_with_clean, mat_with_clean, clean, [s for s, _ in seed_items]


def aggregate_curve(results: Dict[str, Any], pe: str, attack: str) -> Dict[str, Any]:
    eps, mat, clean, seeds = curve_matrix(results, pe, attack)
    mean = np.nanmean(mat, axis=1)
    std = np.nanstd(mat, axis=1, ddof=1) if mat.shape[1] > 1 else np.zeros(mat.shape[0])
    return {
        "eps": [float(x) for x in eps],
        "mean_accuracy": [float(x) for x in mean],
        "std_accuracy": [float(x) for x in std],
        "n": len(seeds),
        "seeds": seeds,
        "clean_mean": float(np.mean(clean)),
        "clean_std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
    }


def compute_eps_crit(eps: np.ndarray, acc: np.ndarray, clean_acc: float, target_fraction: float = 0.5) -> float:
    target = target_fraction * clean_acc
    for i in range(len(eps) - 1):
        if acc[i] >= target and acc[i + 1] < target:
            frac = (acc[i] - target) / (acc[i] - acc[i + 1])
            if eps[i] == 0:
                return float(eps[i] + frac * (eps[i + 1] - eps[i]))
            log_e0, log_e1 = np.log10(eps[i]), np.log10(eps[i + 1])
            return float(10 ** (log_e0 + frac * (log_e1 - log_e0)))
    return float("nan")


def compute_degradation_rate(eps: np.ndarray, acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = eps > 0
    eps_pos = eps[mask]
    acc_pos = acc[mask]
    if len(eps_pos) < 2:
        return np.array([]), np.array([])
    log_eps = np.log10(eps_pos)
    rate = -np.diff(acc_pos) / np.diff(log_eps)
    mid = 10 ** ((log_eps[:-1] + log_eps[1:]) / 2)
    return mid, rate


def summarize_main_results(result_json: Dict[str, Any], target_fraction: float) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"metadata": result_json.get("metadata", {}), "clean_accuracy": {}, "attacks": {}}
    for pe in PE_ORDER:
        clean_vals = [d["clean_acc"] for _, d in sorted_seed_items(result_json["results"][pe]) if d.get("status") == "ok"]
        summary["clean_accuracy"][pe] = sample_mean_std(clean_vals)
    for attack in ATTACKS:
        summary["attacks"][attack] = {}
        for pe in PE_ORDER:
            c = aggregate_curve(result_json, pe, attack)
            eps = np.array(c["eps"], dtype=float)
            acc = np.array(c["mean_accuracy"], dtype=float)
            epscrit = compute_eps_crit(eps, acc, c["clean_mean"], target_fraction)
            mid, rate = compute_degradation_rate(eps, acc)
            peak_idx = int(np.nanargmax(rate)) if len(rate) else None
            peak = {
                "epsilon_star": float(mid[peak_idx]) if peak_idx is not None else float("nan"),
                "peak_rate_pp_per_decade": float(rate[peak_idx]) if peak_idx is not None else float("nan"),
            }
            c["epsilon_crit"] = epscrit
            c["peak_degradation"] = peak
            summary["attacks"][attack][pe] = c
    return summary


def summarize_noise(analysis: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pe in PE_ORDER:
        seed_items = sorted_seed_items(analysis[pe])
        noise_dicts = [d["noise_ablation"] for _, d in seed_items]
        levels = noise_dicts[0]["noise_levels"]
        acc = np.array([d["accuracies"] for d in noise_dicts], dtype=float)
        no_pe = [d["accuracy_no_pe"] for d in noise_dicts]
        out[pe] = {
            "noise_levels": [float(x) for x in levels],
            "mean_accuracy": [float(x) for x in np.mean(acc, axis=0)],
            "std_accuracy": [float(x) for x in np.std(acc, axis=0, ddof=1)],
            "accuracy_no_pe": sample_mean_std(no_pe),
            "n": len(seed_items),
        }
    return out


def summarize_alibi_ablation(ablation: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    alibi = ablation["results"]["alibi"]
    for regime in ["slopes_only", "reldist_only", "both"]:
        regime_key = f"regime_{regime}"
        seed_items = [(s, d) for s, d in sorted_seed_items(alibi) if regime_key in d]
        attack_dicts = [d[regime_key]["pgd_pe"] for _, d in seed_items]
        eps = common_float_keys(attack_dicts)
        mat = np.full((len(eps), len(seed_items)), np.nan)
        for i, e in enumerate(eps):
            for j, (_, d) in enumerate(seed_items):
                ad = d[regime_key]["pgd_pe"]
                mat[i, j] = ad[key_for_float(ad, e)]["accuracy"]
        out[regime] = {
            "eps": [float(x) for x in eps],
            "mean_accuracy": [float(x) for x in np.nanmean(mat, axis=1)],
            "std_accuracy": [float(x) for x in np.nanstd(mat, axis=1, ddof=1)],
            "n": len(seed_items),
        }
    return out


def iso_accuracy_metric(pe_data: Dict[str, Any], metric_key: str, target_acc: float) -> Dict[str, float]:
    vals = []
    for _, seed_data in sorted_seed_items(pe_data):
        attacks = seed_data.get("attacks")
        if not attacks:
            continue
        eps_list = sorted(float(e) for e in attacks.keys())
        accs, metrics = [], []
        ok = True
        for e in eps_list:
            entry = attacks[key_for_float(attacks, e)]
            if "attacked_accuracy" not in entry or metric_key not in entry:
                ok = False
                break
            accs.append(float(entry["attacked_accuracy"]))
            metrics.append(float(entry[metric_key]))
        if not ok:
            continue
        for i in range(len(eps_list) - 1):
            if accs[i] >= target_acc and accs[i + 1] < target_acc:
                frac = (accs[i] - target_acc) / (accs[i] - accs[i + 1])
                vals.append(metrics[i] + frac * (metrics[i + 1] - metrics[i]))
                break
    return sample_mean_std(vals)


def mean_coverage_chebyshev(R: int, grid_side: int) -> float:
    n_within = 0
    for cx in range(grid_side):
        for cy in range(grid_side):
            for x in range(grid_side):
                for y in range(grid_side):
                    if max(abs(x - cx), abs(y - cy)) <= R:
                        n_within += 1
    return n_within / (grid_side ** 4)


def summarize_spatial(spatial: Dict[str, Any], dataset: str, iso_accuracy: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"iso_accuracy": iso_accuracy, "iso_accuracy_metrics": {}, "curves": {}}
    grid = 14 if dataset == "imagenet" else 8
    for pe in PE_ORDER:
        pe_data = spatial["results"][pe]
        out["iso_accuracy_metrics"][pe] = {}
        for metric in ["mean_top1_overlap", "mean_top5_overlap", "mean_js", "mean_mad_miss", "mean_mass_R1_attacked", "mean_mass_R3_attacked"]:
            out["iso_accuracy_metrics"][pe][metric] = iso_accuracy_metric(pe_data, metric, iso_accuracy)
        for R in [1, 3]:
            mass = out["iso_accuracy_metrics"][pe][f"mean_mass_R{R}_attacked"]
            cov = mean_coverage_chebyshev(R, grid)
            out["iso_accuracy_metrics"][pe][f"concentration_ratio_R{R}"] = {
                "mean": float(mass["mean"] / cov) if mass["n"] else float("nan"),
                "std": float(mass["std"] / cov) if mass["n"] else float("nan"),
                "n": mass["n"],
                "boundary_corrected_uniform_coverage": float(cov),
            }
    return out


def summarize_norms(norms: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pe in PE_ORDER:
        pe_data = norms["results"][pe]
        seed_items = [(s, d) for s, d in sorted_seed_items(pe_data) if "attacks" in d]
        attack_dicts = [d["attacks"] for _, d in seed_items]
        eps = common_float_keys(attack_dicts, positive_only=True)
        out[pe] = {"eps": [float(x) for x in eps], "n": len(seed_items), "mean_fraction_at_ceiling": [], "std_fraction_at_ceiling": [], "mean_delta_inf_to_eps": [], "std_delta_inf_to_eps": []}
        for e in eps:
            frac_vals, ratio_vals = [], []
            for _, d in seed_items:
                entry = d["attacks"][key_for_float(d["attacks"], e)]
                frac_vals.append(float(entry.get("fraction_at_ceiling_mean", np.nan)))
                ratio_vals.append(float(entry.get("delta_inf_to_eps", np.nan)))
            f = sample_mean_std(frac_vals)
            r = sample_mean_std(ratio_vals)
            out[pe]["mean_fraction_at_ceiling"].append(f["mean"])
            out[pe]["std_fraction_at_ceiling"].append(f["std"])
            out[pe]["mean_delta_inf_to_eps"].append(r["mean"])
            out[pe]["std_delta_inf_to_eps"].append(r["std"])
    return out


def fmt_ms(d: Dict[str, float], decimals: int = 2) -> str:
    return f"{d['mean']:.{decimals}f} ± {d['std']:.{decimals}f}"


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def render_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# n=6 paper-number summary")
    lines.append("")
    lines.append("Generated from the repository `results/*.json` aggregate files. Values are means ± sample standard deviations over available seeds unless noted otherwise.")
    lines.append("")
    seeds = summary["metadata"]["imagenet"]["config"].get("seeds", [])
    lines.append(f"Seeds: `{', '.join(str(s) for s in seeds)}`. Model count: `4 PE × {len(seeds)} seeds × 2 datasets = {4*len(seeds)*2}`.")
    lines.append("")

    lines.append("## Clean accuracy")
    rows = []
    for pe in PE_ORDER:
        rows.append([pe, fmt_ms(summary["datasets"]["imagenet"]["clean_accuracy"][pe]), fmt_ms(summary["datasets"]["cifar"]["clean_accuracy"][pe])])
    lines.append(md_table(["PE", "ImageNet-100", "CIFAR-100"], rows))
    lines.append("")

    lines.append("## PGD-PE critical epsilon and peak degradation")
    rows = []
    for pe in PE_ORDER:
        pi = summary["datasets"]["imagenet"]["attacks"]["pgd_pe"][pe]
        pc = summary["datasets"]["cifar"]["attacks"]["pgd_pe"][pe]
        rows.append([
            pe,
            f"{pi['epsilon_crit']:.2f}",
            f"{pc['epsilon_crit']:.2f}",
            f"{pi['peak_degradation']['epsilon_star']:.2f}",
            f"{pc['peak_degradation']['epsilon_star']:.2f}",
            f"{pi['peak_degradation']['peak_rate_pp_per_decade']:.1f}",
            f"{pc['peak_degradation']['peak_rate_pp_per_decade']:.1f}",
        ])
    lines.append(md_table(["PE", "IN eps_crit", "CF eps_crit", "IN eps*", "CF eps*", "IN peak pp/dec", "CF peak pp/dec"], rows))
    lines.append("")

    lines.append("## Random-noise ablation, ImageNet-100")
    levels = summary["noise"]["learned"]["noise_levels"]
    rows = []
    for pe in PE_ORDER:
        n = summary["noise"][pe]
        vals = {str(level): f"{mean:.2f} ± {std:.2f}" for level, mean, std in zip(n["noise_levels"], n["mean_accuracy"], n["std_accuracy"])}
        rows.append([pe, vals.get("0.0"), vals.get("1.0"), vals.get("3.0"), vals.get("5.0"), fmt_ms(n["accuracy_no_pe"])])
    lines.append(md_table(["PE", "sigma=0", "sigma=1", "sigma=3", "sigma=5", "no PE"], rows))
    lines.append("")

    lines.append("## ALiBi structural ablation at epsilon = 0.1")
    rows = []
    for ds in DATASETS:
        abl = summary["alibi_ablation"][ds]
        row = [DATASET_LABEL[ds]]
        for regime in ["slopes_only", "reldist_only", "both"]:
            eps = abl[regime]["eps"]
            idx = min(range(len(eps)), key=lambda i: abs(eps[i] - 0.1))
            row.append(f"{abl[regime]['mean_accuracy'][idx]:.1f} ± {abl[regime]['std_accuracy'][idx]:.1f}")
        rows.append(row)
    lines.append(md_table(["Dataset", "slopes only", "reldist only", "both"], rows))
    lines.append("")

    lines.append("## Spatial concentration at 40% iso-accuracy")
    rows = []
    for ds in DATASETS:
        for pe in PE_ORDER:
            m = summary["spatial"][ds]["iso_accuracy_metrics"][pe]
            rows.append([DATASET_LABEL[ds], pe, fmt_ms(m["concentration_ratio_R1"], 2), fmt_ms(m["concentration_ratio_R3"], 2), fmt_ms(m["mean_mad_miss"], 2), fmt_ms(m["mean_top1_overlap"], 3)])
    lines.append(md_table(["Dataset", "PE", "R1 concentration", "R3 concentration", "MAD-miss", "top1 overlap"], rows))
    lines.append("")

    lines.append("## Included full tables")
    lines.append("")
    lines.append("The companion JSON output contains full attack curves, all epsilon-level mean/std arrays, ALiBi ablation curves, spatial iso-accuracy metrics, and perturbation-norm summaries.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    data = load_all(results_dir)

    summary: Dict[str, Any] = {
        "metadata": {
            "imagenet": data["imagenet_results"].get("metadata", {}),
            "cifar": data["cifar_results"].get("metadata", {}),
            "target_fraction_for_epsilon_crit": args.target_fraction,
            "iso_accuracy_for_spatial_metrics": args.iso_accuracy,
        },
        "datasets": {
            "imagenet": summarize_main_results(data["imagenet_results"], args.target_fraction),
            "cifar": summarize_main_results(data["cifar_results"], args.target_fraction),
        },
        "noise": summarize_noise(data["analysis_data"]),
        "alibi_ablation": {
            "imagenet": summarize_alibi_ablation(data["imagenet_alibi_ablation"]),
            "cifar": summarize_alibi_ablation(data["cifar_alibi_ablation"]),
        },
        "spatial": {
            "imagenet": summarize_spatial(data["imagenet_spatial_metrics"], "imagenet", args.iso_accuracy),
            "cifar": summarize_spatial(data["cifar_spatial_metrics"], "cifar", args.iso_accuracy),
        },
        "perturbation_norms": {
            "imagenet": summarize_norms(data["imagenet_perturbation_norms"]),
            "cifar": summarize_norms(data["cifar_perturbation_norms"]),
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
