# CLIPSeg Split Conformal Prediction

Pixel-wise conformal prediction for semantic segmentation using CLIPSeg.

## Overview

This project implements split conformal prediction for CLIPSeg semantic segmentation, providing uncertainty-quantified prediction sets with coverage guarantees.

## Project Structure

```
clipseg-conformal/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # CLIPSeg model wrapper
│   ├── conformal.py         # Conformal prediction implementation
│   ├── data_utils.py        # Data loading and preprocessing
│   ├── visualization.py     # Visualization utilities
│   └── main.py              # Main pipeline script
├── data/                    # Data directory (place your data here)
├── outputs/                 # Output directory for results
├── tests/                   # Unit tests (to be implemented)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd clipseg-conformal
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the pipeline with your data:

```bash
python src/main.py \
    --image data/015.jpg \
    --mask data/015.tiff \
    --classes paved-area dirt grass rocks person dog car bicycle tree \
    --indices 1 2 3 6 15 16 17 18 19 \
    --alpha 0.1 \
    --output-dir outputs/
```

### Arguments

- `--image`: Path to input image (JPEG, PNG, etc.)
- `--mask`: Path to ground truth mask (multi-channel TIFF)
- `--classes`: Space-separated list of class names
- `--indices`: Space-separated list of channel indices in mask
- `--alpha`: Miscoverage rate (default: 0.1 for 90% coverage)
- `--cal-ratio`: Calibration set ratio (default: 0.5)
- `--output-dir`: Output directory (default: "outputs")
- `--seed`: Random seed (default: 42)

### As a Library

You can also import and use the components directly:

```python
from src import CLIPSegModel, ConformalPredictor
from src import load_image_and_mask, split_calibration_test
from src import visualize_results

# Load data
image, ground_truth = load_image_and_mask(
    "data/015.jpg", 
    "data/015.tiff",
    class_indices=[1, 2, 3, 6, 15, 16, 17, 18, 19]
)

# Split calibration/test
cal_mask, test_mask = split_calibration_test(352, 352, cal_ratio=0.5)

# Run predictions
model = CLIPSegModel()
probs = model.predict(image, class_names)

# Calibrate and predict
cp = ConformalPredictor(alpha=0.1)
cp.calibrate(probs.numpy(), ground_truth, cal_mask)
prediction_sets, set_sizes = cp.predict(probs.numpy())

# Visualize
visualize_results(image, ground_truth, probs.numpy(), 
                 prediction_sets, set_sizes, save_path="results.png")
```

## How It Works

1. **Data Loading**: Load image and multi-channel ground truth mask
2. **Calibration/Test Split**: Randomly split pixels 50/50 for calibration and testing
3. **Model Prediction**: Run CLIPSeg to get class probabilities for each pixel
4. **Calibration**: Compute conformity scores on calibration set and determine threshold
5. **Prediction Sets**: Generate prediction sets that include all plausible classes
6. **Evaluation**: Measure empirical coverage on held-out test pixels

## Key Features

- **Coverage Guarantees**: Provides valid prediction sets with user-specified coverage (e.g., 90%)
- **Uncertainty Quantification**: Set size reflects prediction uncertainty
- **Pixel-wise Split**: Uses spatial split conformal prediction
- **Extensible**: Clean modular code for easy experimentation

## Output

The pipeline generates:
- Visualization comparing ground truth, point predictions, and conformal sets
- Uncertainty maps showing prediction set sizes
- Coverage statistics and set size distributions
- Example prediction sets for individual pixels

## Next Steps

This is a simple initial structure designed to be scaled up later. Potential extensions:

- Add unit tests in `tests/`
- Implement batch processing for multiple images
- Add support for different conformal methods (full conformal, weighted CP, etc.)
- Integrate with MLflow or Weights & Biases for experiment tracking
- Add CI/CD pipeline
- Create Docker container
- Add more visualization options

## References

- CLIPSeg: [https://huggingface.co/CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined)
- Conformal Prediction: Vovk et al., "Algorithmic Learning in a Random World" (2005)

## License

MIT License