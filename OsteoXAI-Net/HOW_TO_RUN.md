# How to Run OsteoXAI-Net

1. Install Python >=3.10 (3.11 recommended), matching `torch`/`torchvision`, and `pip install -r requirements.txt`.
2. Download and unpack the two public knee X-ray datasets into image folders named `Normal`, `Osteoporosis`, `Osteopenia`.
3. From the project root, run `python scripts/prepare_data.py --mendeley data/raw/mendeley --kaggle data/raw/kaggle`.
4. Inspect class counts and `data/manifests/fixed_split.csv`; never adjust the test labels or split by hand after seeing evaluation results.
5. Run `python scripts/train.py --manifest data/manifests/fixed_split.csv --config configs/default.json`.
6. Run `python scripts/evaluate.py --predictions results/fixed/predictions.csv`.
7. Generate Grad-CAM++ images and MC uncertainty with `python scripts/predict.py --checkpoint results/fixed/best.pt --image 'path/to/image.png' --explain`.
8. Run five-by-five CV with `python scripts/cross_validate.py --manifest data/manifests/mendeley_all.csv`.
9. Add baselines, ablations, robustness, and clinical baseline using scripts named in README.md.
10. Run `python -m pytest -q tests` after changing any code.

Optional: configure a smaller exploratory run using `--config configs/smoke.json` (not paper-equivalent). For GPUs set CUDA-enabled PyTorch installation per pytorch.org. Model weights may download from torchvision on first use; pre-download for offline execution. To disable pretrained downloads use `"pretrained": false` in JSON but report this departure from the manuscript.

Output: `results/fixed/best.pt`, preprocessing statistics, split manifest, validation-selected threshold, CSV predictions, training curve history, metrics and 600dpi ROC/PR graphics.
