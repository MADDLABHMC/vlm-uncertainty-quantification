# MC Dropout for CLIPSeg Uncertainty Quantification

Monte Carlo Dropout for epistemic uncertainty estimation in semantic segmentation using CLIPSeg. Dropout is injected into the CLIP vision encoder's MLP layers to enable Bayesian-like inference at test time.

## Project Structure

```
MC_dropout/
├── src/
│   ├── __init__.py
│   ├── model.py          # CLIPSegWithEncoderDropout, load_model
│   ├── data_utils.py     # load_image_and_mask, prepare_ground_truth, iter_dataset_pairs
│   ├── inference.py      # mc_dropout_predict
│   ├── metrics.py        # Mean IoU, uncertainty stats
│   └── visualization.py  # plotting utilities
├── main.py               # Main pipeline (single image)
├── run_on_dataset.py     # Full dataset evaluation
├── run_convergence_study.py  # Sample size convergence analysis
├── example.py            # Minimal example
├── requirements.txt
├── README.md
└── MC_Dropout_CLIPSeg.ipynb  # Original notebook
```

## Installation

```bash
cd MC_dropout
pip install -r requirements.txt
```

## Usage

### Basic: Image with default RUGD classes (no ground truth)

```bash
python main.py --image path/to/image.jpg --output-dir outputs
```

### With ground truth mask (for Mean IoU evaluation)

```bash
python main.py --image path/to/image.jpg --mask path/to/mask.tiff --output-dir outputs
```

### Custom classes

```bash
python main.py --image image.jpg --mask mask.tiff \
  --classes paved-area dirt grass rocks vegetation person dog bicycle tree \
  --indices 1 2 3 6 8 15 16 18 19
```

### Full dataset (Semantic Drone)

Run MC Dropout on an entire dataset. Expects Semantic Drone–style layout: `images/` and `labels/tiff/` with matching filenames (e.g., `001.jpg` / `001.tiff`). Classes are loaded from `classes.csv` in the dataset root.

```bash
python run_on_dataset.py --dataset-path /path/to/semantic_drone --output-dir outputs
```

Test on a subset of images:

```bash
python run_on_dataset.py --dataset-path /path/to/semantic_drone --limit 10 --output-dir outputs
```

Use a subset of classes (indices are auto-derived from `classes.csv`):

```bash
python run_on_dataset.py --dataset-path /path/to/semantic_drone \
  --classes paved-area dirt grass rocks vegetation person dog bicycle tree
```

Or specify indices explicitly:

```bash
python run_on_dataset.py --dataset-path /path/to/semantic_drone \
  --classes paved-area dirt grass --indices 1 2 3
```

### Dataset with image transforms

Run MC Dropout with one of the image transforms from `Image_Transform.ipynb`. Use `--transform` to select which transform to apply (one per run):

```bash
# No transform (baseline)
python run_on_dataset_transforms.py --dataset-path /path/to/dataset --transform none

# Gaussian blur (--transform-sigma, default 10)
python run_on_dataset_transforms.py --dataset-path /path/to/dataset --transform gaussian_blur
python run_on_dataset_transforms.py --dataset-path /path/to/dataset --transform gaussian_blur --transform-sigma 20

# Vignette (--transform-level, default 5; lower = stronger)
python run_on_dataset_transforms.py --dataset-path /path/to/dataset --transform vignette --transform-level 4

# Occlusions (--transform-num-rois, --transform-opacity)
python run_on_dataset_transforms.py --dataset-path /path/to/dataset --transform occlusions --transform-num-rois 3

# Smoke (--transform-num-rois, --transform-opacity)
python run_on_dataset_transforms.py --dataset-path /path/to/dataset --transform smoke
```

Results are saved to `outputs/transform_<name>/` when a transform is applied.

### Convergence study

Test different MC sample sizes (5, 10, 20, 30, 50, 100) to find the optimal trade-off:

```bash
python run_convergence_study.py --image image.jpg --mask mask.tiff --output-dir outputs
```

## Outputs

**Single image** (`main.py`):
- `mc_dropout_results.png` — 2×3 visualization (image, ground truth, predictions, errors, uncertainty map, histogram)
- `mc_dropout_results.json` — Metrics (mean IoU, mean uncertainty, inference time, etc.)

**Full dataset** (`run_on_dataset.py`):
- `dataset_results.json` — Aggregate metrics (mean IoU ± std, mean uncertainty, total time) and per-image results
- `per_image/*.json` — Per-image metrics (unless `--no-per-image`)

## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-samples` | 500 | Number of MC Dropout forward passes |
| `--dropout-rate` | 0.1 | Dropout probability in vision encoder |
| `--model` | CIDAS/clipseg-rd64-refined | HuggingFace model ID |
| `--limit` | — | (run_on_dataset) Limit number of images for testing |
| `--classes` | — | (run_on_dataset) Subset of class names |
| `--indices` | — | (run_on_dataset) Mask channel indices for `--classes` (optional; auto-derived if omitted) |

