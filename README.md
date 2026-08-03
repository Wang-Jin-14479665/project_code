# COMP9517 Deep Learning Experiments

This repository contains the deep-learning experiments for the COMP9517 group project. The experiments compare:

- training from scratch vs. ImageNet transfer learning;
- basic vs. strong data augmentation;
- ResNet-18 vs. EfficientNet-B0;
- accuracy, macro-F1, training time, inference speed, parameter count, and common classification errors.

The dataset used by these notebooks is `shared_dataset_seed9517` and is expected to follow a PyTorch `ImageFolder` layout with separate `train`, `val`, and `test` splits.

## Experiments

| Experiment | Architecture | Initialisation | Augmentation | Main purpose |
|---|---|---|---|---|
| E1 | ResNet-18 | Random / from scratch | Basic | Scratch baseline |
| E2 | ResNet-18 | ImageNet pretrained | Basic | Transfer learning vs. scratch |
| E3 | ResNet-18 | ImageNet pretrained | Strong | Basic vs. strong augmentation |
| E4 | EfficientNet-B0 | ImageNet pretrained | Strong | Architecture comparison |

The experiment configuration used in the notebooks is:

- random seed: `42`
- input size: `224 x 224`
- batch size: `64`
- data loader workers: `2`
- E1 maximum epochs: `25`
- E2-E4 maximum epochs: `12`
- early stopping based on validation macro-F1
- model selection based on the best validation macro-F1

## Repository files

The experiment notebooks are currently separated so that each notebook runs one main experiment:

```text
COMP9517_deep_learning_experiments_E1.ipynb
COMP9517_deep_learning_experiments_E2(1).ipynb
COMP9517_deep_learning_experiments-E3(2).ipynb
COMP9517_deep_learning_experiments_E4(1).ipynb
requirements.txt
deep_learning_results/
```

Each notebook contains the common training/evaluation pipeline and an `EXPERIMENTS` dictionary defining E1-E4. The final execution cell selects the experiment to run.

## Dataset structure

The code recursively searches for `train`, `val`, and `test`, so the dataset can contain an additional top-level directory as long as the following structure exists somewhere below the configured dataset root:

```text
shared_dataset_seed9517/
├── train/
│   ├── class001/
│   ├── class002/
│   └── ...
├── val/
│   ├── class001/
│   ├── class002/
│   └── ...
└── test/
    ├── class001/
    ├── class002/
    └── ...
```

The notebooks check that the class-to-index mapping is identical across all three splits.

---

# Running locally

## 1. Clone the repository

```bash
git clone <repository-url>
cd project_code
```

## 2. Create a Python environment

Example using `venv` on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The supplied `requirements.txt` uses the PyTorch CUDA 13.2 wheel index. If the machine uses a different CUDA/PyTorch configuration, install a compatible PyTorch build first and then install the remaining dependencies.

## 3. Place the dataset next to the code

For local use, the default project layout should be:

```text
project_code/
├── shared_dataset_seed9517/
│   ├── train/
│   ├── val/
│   └── test/
├── deep_learning_results/
├── COMP9517_deep_learning_experiments_E1.ipynb
├── COMP9517_deep_learning_experiments_E2(1).ipynb
├── COMP9517_deep_learning_experiments-E3(2).ipynb
├── COMP9517_deep_learning_experiments_E4(1).ipynb
└── requirements.txt
```

If the dataset is downloaded as a ZIP file, extract it before running locally.

## 4. Change the Colab-only path cell

The current notebooks were written primarily for Google Colab, so the cell containing:

```python
from google.colab import drive
drive.mount('/content/drive')
```

should be skipped when running locally.

Use the following local paths instead:

```python
from pathlib import Path

PROJECT_DIR = Path.cwd()
EXTRACT_DIR = PROJECT_DIR / "shared_dataset_seed9517"
OUTPUT_DIR = PROJECT_DIR / "deep_learning_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

The existing `find_split_dirs(EXTRACT_DIR)` function can then locate the `train`, `val`, and `test` directories automatically.

If the notebook is launched from a different working directory, set `PROJECT_DIR` explicitly to the repository directory.

## 5. Select a device

The notebooks automatically use CUDA when available:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

A CUDA-capable GPU is strongly recommended for the full experiments. CPU execution is supported by the code but will be much slower.

## 6. Run a smoke test first

Before a full experiment, set:

```python
RUN_MODE = "smoke"
```

Smoke mode uses a reduced subset and trains for two epochs. It is intended only to verify that the complete pipeline works; smoke-test results should not be used in the final report.

After the pipeline runs successfully, change it back to:

```python
RUN_MODE = "full"
```

## 7. Run the required experiment

In the experiment execution cell, uncomment only the experiment that should be run. For example:

```python
result_E1 = run_experiment(EXPERIMENTS["E1"])
```

or:

```python
result_E4 = run_experiment(EXPERIMENTS["E4"])
```

Running experiments one at a time is recommended because each full training run is relatively long and creates a separate best-model checkpoint.

---

# Running in Google Colab

## 1. Upload the dataset to Google Drive

The current notebook code expects the compressed dataset at:

```text
/content/drive/MyDrive/COMP9517/shared_dataset_seed9517.zip
```

and saves experiment outputs to:

```text
/content/drive/MyDrive/COMP9517/deep_learning_results
```

> Note: `deep_learning_results` is the output directory in the current notebooks, not the dataset directory. If the dataset has actually been placed inside another Drive folder, update `ZIP_PATH` accordingly.

## 2. Open the notebook in Colab

Upload/open the required `.ipynb` file, then select:

```text
Runtime -> Change runtime type -> GPU
```

## 3. Mount Google Drive

Run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The notebook then uses:

```python
ZIP_PATH = Path("/content/drive/MyDrive/COMP9517/shared_dataset_seed9517.zip")
EXTRACT_DIR = Path("/content/dataset")
```

The ZIP archive is extracted into the Colab runtime rather than training directly from Drive. This avoids repeatedly reading thousands of small image files from Google Drive and generally gives better training performance.

A `.extract_complete` marker is created under `/content/dataset` so repeated execution in the same Colab runtime does not unnecessarily extract the dataset again.

## 4. Verify the dataset

Run the dataset-check cells and confirm that:

- `train`, `val`, and `test` are detected;
- all three splits use the same class mapping;
- the expected number of classes is found;
- the image counts look correct.

## 5. Run smoke mode, then full mode

First use:

```python
RUN_MODE = "smoke"
```

After a successful test, switch to:

```python
RUN_MODE = "full"
```

Then run the experiment-specific execution cell.

Because checkpoints and CSV/JSON results are written to Google Drive, they remain available after the Colab runtime disconnects.

---

# Data augmentation

## Basic augmentation

The basic training pipeline uses:

- `RandomResizedCrop(224, scale=(0.75, 1.0))`
- random horizontal flip
- ImageNet normalisation

## Strong augmentation

The strong pipeline additionally uses:

- a wider random crop scale;
- random rotation;
- random affine translation/scale;
- colour jitter;
- random erasing.

Validation and test images use deterministic resize/centre-crop preprocessing only.

---

# Outputs

Each experiment writes files to `deep_learning_results/`.

For an experiment such as `E2_resnet18_pretrained_basic`, the main outputs are:

```text
E2_resnet18_pretrained_basic_best.pt
E2_resnet18_pretrained_basic_config.json
E2_resnet18_pretrained_basic_history.csv
E2_resnet18_pretrained_basic_metrics.json
E2_resnet18_pretrained_basic_predictions.csv
```

Additional analysis may produce:

```text
experiment_summary.csv
analysis_figures/
analysis_tables/
```

The files contain:

- `*_best.pt` - best training checkpoint selected by validation macro-F1;
- `*_config.json` - experiment configuration and dataset information;
- `*_history.csv` - epoch-by-epoch training/validation history;
- `*_metrics.json` - final test metrics and timing information;
- `*_predictions.csv` - test targets and predicted classes;
- `experiment_summary.csv` - combined comparison of available experiment metrics.

The evaluation pipeline reports top-1 accuracy, top-5 accuracy, macro precision, macro recall, macro-F1, inference time, throughput, parameter count, and training time.

## Recorded full-run results

The uploaded notebooks currently contain the following completed full-run results:

| Experiment | Top-1 | Top-5 | Macro-F1 | Best val macro-F1 |
|---|---:|---:|---:|---:|
| E1 ResNet-18 scratch + basic | 0.3208 | 0.5632 | 0.3162 | 0.3154 |
| E2 ResNet-18 pretrained + basic | 0.5772 | 0.8038 | 0.5717 | 0.5705 |
| E3 ResNet-18 pretrained + strong | 0.5636 | 0.7974 | 0.5542 | 0.5459 |
| E4 EfficientNet-B0 pretrained + strong | 0.7216 | 0.8986 | 0.7177 | 0.7085 |

These values are provided for experiment tracking; they should be interpreted together with the project report rather than treated as standalone conclusions.

---

# Checkpoints and Git

The `.pt` files are PyTorch model checkpoints. They are around tens or hundreds of megabytes each and are binary files that change completely when a model is retrained.

For this project, they normally should **not** be committed to the regular Git repository because:

- they are large;
- Git does not efficiently diff binary model files;
- repeated checkpoints can quickly inflate repository size;
- GitHub has file-size limits for normal Git objects.

The repository therefore ignores:

```gitignore
*.pt
*.pth
*.ckpt
```

The small experiment metadata and analysis files (`.json`, `.csv`, figures, and tables) can still be committed because they are useful for reproducing and documenting the reported results.

If model weights must be shared with the group or submitted separately, use Google Drive, a release/artifact store, or Git LFS rather than normal Git history.

---

# Reproducibility notes

The code sets random seeds for Python, NumPy, and PyTorch and enables deterministic cuDNN behaviour where possible. However, exact training results may still vary slightly across hardware, PyTorch/CUDA versions, and execution environments.

Do not tune hyperparameters repeatedly against the test set. The intended workflow is:

1. train on `train`;
2. select/check models using `val`;
3. use `test` only for final evaluation.

---

# Important notebook check

Before committing the notebooks, verify the final experiment-selection cell in each file. In the currently uploaded E3 notebook, the stored output is from E3, but the visible execution line currently selects E4:

```python
result_E4 = run_experiment(EXPERIMENTS["E4"])
```

For the E3 notebook, this should be changed to:

```python
result_E3 = run_experiment(EXPERIMENTS["E3"])
```

Then clear stale outputs or rerun the notebook so the source code and displayed output are consistent.
