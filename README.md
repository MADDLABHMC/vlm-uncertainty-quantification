# VLM Uncertainty Quantification

A research toolkit for applying **uncertainty quantification (UQ)** methods to Vision-Language Models (VLMs). This repository implements and compares three distinct UQ approaches — Monte Carlo Dropout, Monte Carlo Temperature Sampling, and Conformal Prediction — on top of [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined), a vision-language model for image segmentation.

---

## Overview

Deploying VLMs in real-world applications requires more than high accuracy — it requires knowing *when the model doesn't know*. This project investigates how different uncertainty quantification strategies can be applied to VLMs, helping practitioners make informed decisions about prediction reliability.

The three methods explored here represent a spectrum of UQ approaches:

- **MC Dropout** — Bayesian approximation via stochastic inference
- **MC Temperature Sampling** — Uncertainty via sampling with controlled randomness
- **Conformal Prediction** — Distribution-free, statistically rigorous prediction sets

---

## Repository Structure

```
vlm-uncertainty-quantification/
├── MC_dropout/                 # Monte Carlo Dropout experiments
├── MC_temperature/             # Monte Carlo Temperature Sampling experiments
├── conformal_prediction/       # Conformal Prediction experiments
└── .ipynb_checkpoints/         # Jupyter notebook checkpoints
```

---

## Methods

### 1. Monte Carlo Dropout (`MC_dropout/`)

MC Dropout treats dropout layers as a Bayesian approximation. Rather than disabling dropout at inference time (as is standard), dropout is kept active and multiple forward passes are run over the same input. The variance across passes estimates **epistemic uncertainty** — uncertainty due to model limitations or out-of-distribution inputs.

Key parameters:
- `K` — number of forward passes
- Uncertainty measured as variance of prediction scores across passes

### 2. Monte Carlo Temperature Sampling (`MC_temperature/`)

Temperature sampling introduces stochasticity at the output level by scaling the logits before applying softmax. Running multiple passes with different temperatures produces a distribution of predictions, from which uncertainty can be estimated.

Key parameters:
- Temperature `τ` — controls sharpness/spread of the output distribution
- Higher `τ` → more uniform distributions, higher apparent uncertainty
- Lower `τ` → more peaked predictions, lower uncertainty

### 3. Conformal Prediction (`conformal_prediction/`)

Conformal prediction is a **distribution-free** framework that produces prediction *sets* (rather than point estimates) with a guaranteed coverage rate. Given a user-specified error rate `α`, the method guarantees that the true label is included in the prediction set at least `1 - α` of the time, without requiring distributional assumptions about the data.

Key parameters:
- `α` — desired error rate (e.g., `α = 0.1` for 90% coverage)
- Calibration set — held-out data used to determine the conformal threshold `q̂`

---

## Installation

**Requirements:** Python 3.8+, PyTorch, and the Hugging Face `transformers` library.

```bash
# Clone the repository
git clone https://github.com/MADDLABHMC/vlm-uncertainty-quantification.git
cd vlm-uncertainty-quantification

# Install dependencies
pip install torch transformers pandas pillow
```

---

## Usage

Each UQ method is contained in its own subdirectory. Navigate into the method of interest and run the corresponding notebook or script.

```bash
# Example: run MC Dropout experiments
cd MC_dropout
jupyter notebook
```

```bash
# Example: run Conformal Prediction experiments
cd conformal_prediction
jupyter notebook
```

---

## Results

### Conformal Prediction — Robustness Under Image Perturbations

Conformal prediction results are located in `conformal_prediction/new_stats/plots/`. To stress-test the method's robustness, conformal prediction was evaluated across **400 images** under 7 distinct visual perturbations applied using [Albumentations](https://albumentations.ai/):

| Perturbation | Plot |
|---|---|
| Atmospheric Fog | `prediction_set_atmospheric_fog.png` |
| Gaussian Blur | `prediction_set_gaussian_blur.png` |
| Glass Blur | `prediction_set_glass_blur.png` |
| Horizontal Blur | `prediction_set_horizontal_blur.png` |
| Monochrome | `prediction_set_monochrome.png` |
| Rain | `prediction_set_rain.png` |
| Vertical Blur | `prediction_set_vertical_blur.png` |

These experiments probe whether the coverage guarantees provided by conformal prediction hold under distribution shift introduced by common real-world image degradations.

---

## Background

### Why Uncertainty Quantification for VLMs?

Vision-Language Models make probabilistic decisions by jointly reasoning over images and text. However, standard VLM outputs are point predictions — a single segmentation mask, a single class label — with no indication of confidence. This is problematic in safety-critical settings such as medical imaging, autonomous systems, or decision support, where it matters deeply whether the model is confident or guessing.

UQ methods address this by augmenting predictions with uncertainty estimates, enabling downstream systems to:
- Defer low-confidence predictions to human review
- Filter unreliable outputs before acting on them
- Evaluate model calibration across input distributions

### Method Comparison

| Method | Type | Distribution-Free | Coverage Guarantee | Computational Cost |
|---|---|---|---|---|
| MC Dropout | Bayesian | No | No | Medium (K forward passes) |
| MC Temperature | Sampling-based | No | No | Medium (K forward passes) |
| Conformal Prediction | Frequentist | **Yes** | **Yes** | Low (calibration only) |

---

## References

- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML.
- Angelopoulos & Bates (2022). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv.
- Bethell et al. (2023). *Robust Uncertainty Quantification using Conformalised Monte Carlo Prediction.* AAAI 2024.

---

## About

This repository is maintained by the **MADDLAB @ HMC**.
