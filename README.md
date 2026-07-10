# SPAS: Signature-Preserving Adaptive Sampling for Intrusion Detection

This repository contains the experimental source code used in the dissertation research on computer attack detection in corporate networks based on the time-frequency representation of network traffic.

The implementation covers three experimental stages:

1. comparison of a conventional vector baseline with a sonification-based 2D-CNN model;
2. evaluation of the proposed Signature-Preserving Adaptive Sampling (SPAS) method against the baseline and SMOTENC;
3. selective reclassification of uncertain predictions using SPAS, ErrorBoost, an auxiliary classifier, and a configurable τ-zone.

## Repository structure

```text
SPAS/
├── src/
│   ├── section_4_2_baseline_vs_sonification.py
│   ├── section_4_3_spas_balancing.py
│   └── section_4_4_uncertainty_reclassification.py
├── data/
│   └── nsl-kdd/
│       └── README.md
├── results/
│   └── README.md
├── README.md
└── requirements.txt
```

## Experimental modules

### `src/section_4_2_baseline_vs_sonification.py`

Implements the experiments from dissertation Section 4.2. The script:

- loads and preprocesses NSL-KDD;
- performs one-hot encoding and feature scaling;
- converts network records into waveform signals;
- builds STFT-based spectrograms;
- trains a vector neural-network baseline;
- trains a sonification-based 2D-CNN;
- calculates Accuracy, Precision, Recall, F1-score, FPR, and FNR;
- exports tables, figures, confusion matrices, and classification reports.

### `src/section_4_3_spas_balancing.py`

Implements the experiments from dissertation Section 4.3. The script:

- compares the baseline, SMOTENC, and SPAS;
- generates minority-class samples within class-specific percentile bounds;
- preserves categorical attack-signature attributes;
- evaluates binary attack detection and per-attack recall;
- exports balancing plans, distributions, metrics, and visual comparisons.

### `src/section_4_4_uncertainty_reclassification.py`

Implements the experiments from dissertation Section 4.4. The script:

- trains the baseline SPAS model;
- identifies errors used by the ErrorBoost procedure;
- trains an auxiliary model;
- selects a τ-zone and confidence-gating thresholds;
- selectively replaces uncertain predictions;
- compares baseline, auxiliary, and mixed-model results;
- exports tables, figures, and experiment summaries.

## Requirements

- Python 3.10 or newer
- TensorFlow/Keras
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

Install the dependencies from the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

## Dataset preparation

The NSL-KDD dataset is not included in the repository. Place the following files in `data/nsl-kdd/`:

```text
data/nsl-kdd/KDDTrain+.txt
data/nsl-kdd/KDDTest+.txt
```

Additional details are provided in [`data/nsl-kdd/README.md`](data/nsl-kdd/README.md).

## Running the experiments

Run each experiment from the repository root:

```bash
python src/section_4_2_baseline_vs_sonification.py
python src/section_4_3_spas_balancing.py
python src/section_4_4_uncertainty_reclassification.py
```

The scripts are computationally intensive because they generate waveform signals and STFT spectrograms for the complete dataset. A CUDA-capable GPU is recommended for model training.

## Results

Generated tables, plots, reports, configuration files, and summaries are saved automatically in:

```text
results/section_4_2/
results/section_4_3/
results/section_4_4/
```

The exact numerical results can vary slightly because of hardware, software versions, and nondeterministic operations in neural-network training.

## Research context

The repository supports the experimental validation of:

- time-frequency representation of network traffic through sonification;
- two-dimensional convolutional neural-network classification;
- Signature-Preserving Adaptive Sampling;
- error-oriented reinforcement of difficult attack subclasses;
- uncertainty-zone detection and selective decision reclassification.

## Author

**Bohdan Semeniuk**  
Khmelnytskyi National University

Repository: `https://github.com/Faludore/SPAS`
