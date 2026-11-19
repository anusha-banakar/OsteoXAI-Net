# OsteoXAI-Net: Explainable Multi-Scale Spectral-Spatial Deep Learning for Osteoporosis Risk Prediction

OsteoXAI-Net is an explainable deep learning framework for osteoporosis risk prediction from knee radiographs. It integrates bone-aware preprocessing, multi-scale wavelet decomposition, DenseNet121 feature extraction, spectral Transformers, and cross-attention fusion. Monte Carlo Dropout estimates predictive uncertainty, while Grad-CAM++ and spatial attention provide interpretable explanations.

**Note:** This is a research implementation for QUS-referenced osteoporosis risk assessment, not a clinical diagnostic tool.

## Key Features

- Bone-adaptive CLAHE and median denoising.
- Three-level Daubechies-4 (DB4) wavelet decomposition.
- Ten directional spectral sub-bands with learnable embeddings.
- Pretrained DenseNet121 spatial feature extraction.
- Channel and spatial bone attention mechanisms.
- Four-layer Spectral Transformer.
- Eight-head cross-attention fusion.
- Weighted Binary Cross-Entropy loss and attention regularization.
- Monte Carlo Dropout for predictive uncertainty estimation.
- Grad-CAM++ and spatial attention visualizations.
- Explanation consistency and trust estimation.
- ROC-AUC, PR-AUC, F1-score, MCC and bootstrap confidence intervals.
- Baseline comparisons, ablation studies and robustness evaluation.
- Three-class classification extension and Optuna hyperparameter optimization.

## Dataset

The implementation uses two publicly available datasets:

1. **Mendeley Knee X-ray Osteoporosis Database:**  
   https://data.mendeley.com/datasets/fxjm8fb6mw/2

2. **Kaggle Multi-Class Knee Osteoporosis X-Ray Dataset:**  
   https://www.kaggle.com/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset

Download the datasets and organize them as follows:

```text
data/
└── raw/
    ├── mendeley/
    │   ├── Normal/
    │   └── Osteoporosis/
    └── kaggle/
        ├── Normal/
        ├── Osteoporosis/
        └── Osteopenia/
```

The implementation also supports CSV manifests with image paths and class labels.

## Installation

Clone the repository:

```bash
git clone https://github.com/anusha-banakar/OsteoXAI-Net
cd OsteoXAI-Net
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest tests -q
```

## Data Preparation

Prepare the dataset manifests and fixed train-validation-test split:

```bash
python scripts/prepare_data.py --mendeley data/raw/mendeley --kaggle data/raw/kaggle --out data/manifests
```

## Model Training

Train the proposed OsteoXAI-Net model:

```bash
python scripts/train.py --manifest data/manifests/fixed_split.csv
```

## Model Evaluation

Evaluate the model predictions:

```bash
python scripts/evaluate.py --predictions results/fixed/predictions.csv
```

## Prediction and Explainability

Generate predictions and explanations for a knee radiograph:

```bash
python scripts/predict.py --checkpoint results/fixed/best.pt --image path/to/knee_xray.png --explain
```

## Research Experiments

**Cross-validation:**

```bash
python scripts/cross_validate.py --manifest data/manifests/mendeley_all.csv --repeats 5 --folds 5
```

**Ablation study:**

```bash
python scripts/run_ablations.py --manifest data/manifests/fixed_split.csv
```

**Baseline comparison:**

```bash
python scripts/run_baselines.py --manifest data/manifests/fixed_split.csv
```

**Hyperparameter optimization:**

```bash
python scripts/tune.py --manifest data/manifests/fixed_split.csv --trials 50
```

**Explainability analysis:**

```bash
python scripts/run_explainability.py --checkpoint results/fixed/best.pt --manifest data/manifests/fixed_split.csv --limit 10
```

**Robustness evaluation:**

```bash
python scripts/run_robustness.py --checkpoint results/fixed/best.pt --manifest data/manifests/fixed_split.csv
```

## Project Structure

```text
OsteoXAI-Net/
├── osteoxai/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── experiments.py
│   └── metrics.py
├── scripts/
├── configs/
│   ├── default.json
│   └── smoke.json
├── tests/
├── data/
├── checkpoints/
├── results/
├── requirements.txt
├── HOW_TO_RUN.md
└── README.md
```

## Methodological Limitations

- Original datasets and pretrained model weights are not included.
- Bone ROI extraction uses morphological and intensity thresholds rather than learned segmentation.
- Some implementation parameters are reasonable initialization choices because exact numerical values are not specified in the manuscript.
- The implemented baseline models include DenseNet121, ResNet50, EfficientNet-B0 and ViT-B/16.
- The three-class extension is a separate research experiment and is not clinically validated.
- Potential overlaps between Mendeley and Kaggle images must be manually reviewed before cross-collection evaluation.
- Actual final dataset training and reproduction of published numerical results have not been performed as part of the packaged implementation.

## Reproducibility

The implementation supports stratified 70:15:15 data splitting, 5×5 repeated stratified cross-validation, training-only preprocessing statistics, checkpointing and validation-based uncertainty calibration.

For reproducible experiments, use the supplied configurations and maintain independent training, validation and test datasets.

## Citation

If you use this implementation in academic research, please cite the associated research manuscript and acknowledge the original dataset providers. Follow the respective dataset licensing and attribution requirements.
