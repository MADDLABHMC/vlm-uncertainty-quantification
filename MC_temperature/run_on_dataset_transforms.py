"""
Temperature-scaled MC Dropout on a dataset with image transforms.

Default: dropout 0.7, 150 MC samples, 7 classes (paved-area, grass, roof, car, tree, ar-marker, obstacle).

Transforms (one per invocation via --transform):
  monochrome          — single run (grayscale → RGB)
  gaussian_blur       — sigma 2, 5, 10, 20 (sequential runs)
  vignette            — levels 1, 2, 3, 4
  occlusions          — num_rois 15, 20, 25, 30
  smoke               — num_rois 10, 20, 30, 40

Baseline (no transform) is loaded from --baseline-json if set, else from
<output-dir>/baseline/dataset_results.json if present, else computed once and cached there.
Use --force-baseline to recompute baseline.

Outputs: JSON per condition, summary JSON, bar charts (entropy + std_prob per class).

Usage:
  python run_on_dataset_transforms.py --dataset-path /path/to/data --transform gaussian_blur --output-dir outputs_mcT
  python run_on_dataset_transforms.py --dataset-path /path/to/data --transform monochrome --baseline-json /path/to/dataset_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.calibration import calibrate_temperature
from src.data_utils import (
    get_indices_for_classes,
    iter_dataset_pairs,
    load_dataset_classes,
)
from src.model import load_model
from run_on_dataset import aggregate_from_per_image_results, run_single_image

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

DEFAULT_CLASSES = [
    "paved-area",
    "grass",
    "roof",
    "car",
    "tree",
    "ar-marker",
    "obstacle",
]

TRANSFORM_PARAM_LEVELS: dict[str, list] = {
    "monochrome": [None],
    "gaussian_blur": [2, 5, 10, 20],
    "vignette": [1, 2, 3, 4],
    "occlusions": [15, 20, 25, 30],
    "smoke": [10, 20, 30, 40],
}


def _get_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _kwargs_for_level(transform: str, level) -> dict:
    if transform == "monochrome":
        return {}
    if transform == "gaussian_blur":
        return {"sigma": float(level)}
    if transform == "vignette":
        return {"level": float(level)}
    if transform == "occlusions":
        return {"num_rois": int(level), "opacity": 1.0}
    if transform == "smoke":
        return {"num_rois": int(level), "opacity": 1.0}
    raise ValueError(f"Unknown transform {transform}")


def _label_for_level(transform: str, level) -> str:
    if transform == "monochrome":
        return "monochrome"
    return f"{transform}_{level}"


def resolve_baseline_aggregate(
    baseline_path: Path | None,
    output_dir: Path,
    force_baseline: bool,
    dataset_path: Path,
    class_names: list[str],
    class_indices: list[int],
    n_samples: int,
    dropout_rate: float,
    val_images_min: int,
    max_val_images: int,
    limit: int | None,
    device: str,
) -> tuple[dict, Path]:
    """
    Return (aggregate dict, path to JSON file used).
    Computes baseline if needed and saves to output_dir/baseline/dataset_results.json.
    """
    from run_on_dataset import main as run_dataset_main

    if baseline_path is not None and baseline_path.exists() and not force_baseline:
        with open(baseline_path) as f:
            data = json.load(f)
        return data["aggregate"], baseline_path.resolve()

    default_cache = output_dir / "baseline" / "dataset_results.json"
    if default_cache.exists() and not force_baseline:
        with open(default_cache) as f:
            data = json.load(f)
        return data["aggregate"], default_cache.resolve()

    print("\n[Baseline] Computing baseline (no transform) and caching to", default_cache)
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    run_dataset_main(
        dataset_path=str(dataset_path),
        output_dir=str(baseline_dir),
        n_samples=n_samples,
        dropout_rate=dropout_rate,
        val_images_min=val_images_min,
        max_val_images=max_val_images,
        limit=limit,
        device=device,
        class_names=class_names,
        class_indices=class_indices,
    )
    with open(default_cache) as f:
        data = json.load(f)
    return data["aggregate"], default_cache.resolve()


def run_transform_condition(
    dataset_path: Path,
    pairs: list,
    class_names: list[str],
    class_indices: list[int],
    model,
    processor,
    temperature: float,
    n_samples: int,
    device: str,
    dropout_rate: float,
    transform_name: str,
    transform_kwargs: dict,
    output_subdir: Path,
) -> dict:
    """Run full dataset with one transform setting; return aggregate."""
    per_image_results = []
    t0 = time.time()
    for idx, (img_path, mask_path) in enumerate(
        tqdm(pairs, desc=f"  {output_subdir.name}")
    ):
        result = run_single_image(
            model,
            processor,
            img_path,
            mask_path,
            class_names,
            class_indices,
            temperature,
            n_samples,
            device,
            transform_name=transform_name,
            transform_kwargs=transform_kwargs,
            image_index=idx,
        )
        if result.get("per_class_channel_std_probs"):
            all_std_vals = []
            for stats in result["per_class_channel_std_probs"].values():
                m = stats.get("mean")
                if m is not None:
                    all_std_vals.append(m)
            if all_std_vals:
                result["mean_std_prob"] = float(np.mean(all_std_vals))
                result["std_std_prob"] = float(np.std(all_std_vals))
        per_image_results.append(result)
    total_time = time.time() - t0

    aggregate = aggregate_from_per_image_results(
        per_image_results,
        class_names,
        dataset_path,
        n_samples,
        temperature,
        total_time,
        dropout_rate,
    )
    aggregate["transform"] = transform_name
    aggregate["transform_params"] = transform_kwargs
    aggregate["condition_label"] = output_subdir.name

    output_subdir.mkdir(parents=True, exist_ok=True)
    out_json = output_subdir / "dataset_results.json"
    with open(out_json, "w") as f:
        json.dump({"aggregate": aggregate, "per_image": per_image_results}, f, indent=2)
    return aggregate


def plot_bar_charts(
    class_names: list[str],
    baseline_label: str,
    baseline_agg: dict,
    runs: list[tuple[str, dict]],
    output_path: Path,
) -> None:
    """Two figures: mean normalized entropy ± std, and mean std_prob ± std, per class subplots."""
    if plt is None:
        print("matplotlib not installed; skipping plots.")
        return

    n_cls = len(class_names)
    ncols = min(4, n_cls)
    nrows = int(np.ceil(n_cls / ncols))

    labels = [baseline_label] + [r[0] for r in runs]
    x = np.arange(len(labels))
    width = 0.7

    def _per_class_series(key_outer: str, key_mean: str, key_std: str):
        """key_outer e.g. mean_normalized_entropy or mean_std_prob"""
        series = {c: {"means": [], "stds": []} for c in class_names}
        for c in class_names:
            b = baseline_agg.get("per_class", {}).get(c, {}).get(key_outer, {})
            series[c]["means"].append(b.get(key_mean))
            series[c]["stds"].append(b.get(key_std))
        for _, agg in runs:
            for c in class_names:
                s = agg.get("per_class", {}).get(c, {}).get(key_outer, {})
                series[c]["means"].append(s.get(key_mean))
                series[c]["stds"].append(s.get(key_std))
        return series

    ent_series = _per_class_series(
        "mean_normalized_entropy", "mean", "std"
    )
    sp_series = _per_class_series("mean_std_prob", "mean", "std")

    for title, series in [
        ("Mean normalized entropy (± std across images)", ent_series),
        ("Mean std probability (± std across images)", sp_series),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.atleast_2d(axes)
        for i, c in enumerate(class_names):
            r, col = divmod(i, ncols)
            ax = axes[r, col]
            means = [v if v is not None else 0.0 for v in series[c]["means"]]
            stds = [v if v is not None else 0.0 for v in series[c]["stds"]]
            ax.bar(x, means, width, yerr=stds, capsize=3, color="steelblue", ecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(c, fontsize=9)
            ax.set_ylabel(title.split("(")[0].strip())
        for j in range(len(class_names), nrows * ncols):
            r, col = divmod(j, ncols)
            axes[r, col].set_visible(False)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        suffix = "entropy" if "entropy" in title.lower() else "std_prob"
        fig.savefig(output_path.parent / f"{output_path.stem}_{suffix}.png", dpi=150)
        plt.close(fig)

    print(f"Saved plots to {output_path.parent}/*_entropy.png and *_std_prob.png")


def main():
    parser = argparse.ArgumentParser(
        description="MC_temperature dataset evaluation with transforms"
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs_mcT_transforms")
    parser.add_argument(
        "--transform",
        type=str,
        required=True,
        choices=list(TRANSFORM_PARAM_LEVELS.keys()),
        help="Transform family to run (with predefined parameter sweeps)",
    )
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--dropout-rate", type=float, default=0.7)
    parser.add_argument("--val-images-min", type=int, default=30)
    parser.add_argument("--max-val-images", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=DEFAULT_CLASSES,
    )
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Path to existing baseline dataset_results.json (skips baseline run if file exists)",
    )
    parser.add_argument(
        "--force-baseline",
        action="store_true",
        help="Recompute baseline even if cache exists",
    )
    args = parser.parse_args()

    if args.classes and args.indices and len(args.classes) != len(args.indices):
        parser.error("--classes and --indices must match in length")

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or _get_device()

    full_names, _ = load_dataset_classes(dataset_path)
    class_names = args.classes
    class_indices = args.indices or get_indices_for_classes(full_names, class_names)

    pairs = list(iter_dataset_pairs(dataset_path))
    if args.limit:
        pairs = pairs[: args.limit]

    baseline_path = Path(args.baseline_json) if args.baseline_json else None
    baseline_agg, baseline_json_used = resolve_baseline_aggregate(
        baseline_path,
        output_dir,
        args.force_baseline,
        dataset_path,
        class_names,
        class_indices,
        args.n_samples,
        args.dropout_rate,
        args.val_images_min,
        args.max_val_images,
        args.limit,
        device,
    )
    print(f"\nUsing baseline from: {baseline_json_used}")

    print("\n[Calibrate T once on validation (no transform)]")
    model, processor = load_model(dropout_rate=args.dropout_rate)
    model.to(device)
    temperature = calibrate_temperature(
        model,
        processor,
        dataset_path,
        n_mc_samples=args.n_samples,
        val_images_min=args.val_images_min,
        max_val_images=args.max_val_images,
        device=device,
        class_names=class_names,
        class_indices=class_indices,
        verbose=True,
    )
    print(f"  Calibrated T = {temperature:.4f}")

    transform = args.transform
    levels = TRANSFORM_PARAM_LEVELS[transform]
    runs_for_plot: list[tuple[str, dict]] = []
    all_summaries: dict = {
        "dataset_path": str(dataset_path),
        "transform_family": transform,
        "n_samples": args.n_samples,
        "dropout_rate": args.dropout_rate,
        "temperature": temperature,
        "baseline_json": str(baseline_json_used),
        "conditions": {},
    }

    for level in levels:
        kwargs = _kwargs_for_level(transform, level)
        label = _label_for_level(transform, level)
        subdir = output_dir / label
        print(f"\n{'='*60}\nCondition: {label}  params={kwargs}\n{'='*60}")
        agg = run_transform_condition(
            dataset_path,
            pairs,
            class_names,
            class_indices,
            model,
            processor,
            temperature,
            args.n_samples,
            device,
            args.dropout_rate,
            transform,
            kwargs,
            subdir,
        )
        runs_for_plot.append((label, agg))
        all_summaries["conditions"][label] = {
            "aggregate": agg,
            "path": str(subdir / "dataset_results.json"),
        }

    summary_path = output_dir / "transform_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    plot_bar_charts(
        class_names,
        "original",
        baseline_agg,
        runs_for_plot,
        output_dir / "comparison_bars.png",
    )


if __name__ == "__main__":
    main()
