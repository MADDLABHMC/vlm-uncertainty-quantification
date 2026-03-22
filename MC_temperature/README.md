# Temperature-Scaled MC Dropout for CLIPSeg

Combines **Monte Carlo Dropout** with **temperature scaling** for calibrated uncertainty quantification in semantic segmentation, following the paper's implementation steps.

## Implementation (per paper)

1. **Train with dropout** — Dropout (p=0.3) is inserted before the final decoder layer.
2. **Held-out validation set** — ~5,000 samples (or equivalent images) reserved for calibration.
3. **MC dropout at test time** — N=25 stochastic forward passes, dropout kept active.
4. **Temperature parameter T** — Applied as a scalar divisor to logits before softmax: σ_SM(T⁻¹ f(x)).
5. **Optimize T on validation** — Frozen weights; minimize negative log-likelihood to find T > 0.
6. **Calibrated inference** — Use optimized T at test time. Optional H̃_max threshold for rejection.

## Project structure

```
MC_temperature/
├── src/
│   ├── __init__.py
│   ├── model.py        # CLIPSegWithDecoderDropout (dropout p=0.3 before final layer)
│   ├── data_utils.py   # Data loading, train/val split
│   ├── calibration.py   # Temperature optimization (NLL)
│   ├── inference.py     # mc_temperature_predict, predict_with_rejection
│   ├── metrics.py
│   └── visualization.py
├── main.py             # Single image pipeline
├── run_on_dataset.py   # Full dataset with calibration
├── requirements.txt
└── README.md
```

## Installation

```bash
cd MC_temperature
pip install -r requirements.txt
```

## Usage

### Single image (default T=1.0)

```bash
python main.py --image path/to/image.jpg --output-dir outputs
```

### With ground truth

```bash
python main.py --image image.jpg --mask mask.tiff --output-dir outputs
```

### Calibrate T on a dataset, then run

```bash
python main.py --image image.jpg --mask mask.tiff \
  --calibrate-on /path/to/semantic_drone_dataset \
  --output-dir outputs
```

### Optional uncertainty rejection

```bash
python main.py --image image.jpg --mask mask.tiff \
  --temperature 1.5 --H-max 0.8
```

Pixels with normalized entropy > 0.8 are rejected (marked as -1).

### Full dataset

```bash
python run_on_dataset.py --dataset-path /path/to/semantic_drone --output-dir outputs
```

Expects Semantic Drone–style layout: `images/` and `labels/tiff/` with matching filenames.

## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--temperature` | 1.0 | Temperature T (use `--calibrate-on` to optimize) |
| `--n-samples` | 25 | MC dropout forward passes |
| `--H-max` | — | Uncertainty threshold for rejection |
| `--dropout-rate` | 0.3 | Dropout before final layer |
| `--model` | CIDAS/clipseg-rd64-refined | HuggingFace model ID |

## Outputs

- `mc_temperature_results.png` — Visualization (image, GT, predictions, errors, uncertainty)
- `mc_temperature_results.json` — Metrics (accuracy, T, inference time, etc.)
