# EEG Foundation Model Evaluation

Compare frozen **CBraMod** and **LaBraM** encoders for eyes-open versus eyes-closed EEG classification. The project trains logistic-regression classifiers on pretrained features, random-encoder features, and a frequency-bandpower baseline.

It uses baseline recordings from twenty participants in the public [PhysioNet EEG Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/). The experiment is an application of released encoders, not a reproduction of the papers' benchmarks.

## Requirements

- **Python 3.11** for the pinned local dependencies in `requirements.txt`.
- Internet access for the initial dataset, source-code and checkpoint downloads.
- A CUDA GPU is optional. The code falls back to CPU if CUDA is unavailable; feature extraction will be slower. Cross-validation uses CPU classifiers.

## Local setup

Open a terminal in the project folder containing `requirements.txt`:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=numpy,scipy -r requirements.txt
```

On Windows, activate the environment with `.venv311\Scripts\activate` instead.

Run the offline checks:

```bash
python smoke_test.py
python -m unittest discover -s tests -v
```

These checks use synthetic signals. They verify evaluation behavior, not real EEG accuracy.

## Option A: Run the notebook locally

```bash
jupyter lab EEG_Foundation_Lab.ipynb
```

Execute the cells in order. The notebook covers downloading assets, preparing EEG, inspecting signals, extracting features, training classifiers, and repeated nested cross-validation.

If using VS Code, select the `.venv311` Python interpreter as the notebook kernel. A terminal's active environment does not automatically change an existing notebook kernel. Confirm the interpreter inside the notebook:

```python
import sys
print(sys.version)
print(sys.executable)
```

## Option B: Run from the terminal

Run these stages from the project folder:

```bash
# Download pinned official model source and pretrained checkpoints.
python eeg_lab.py setup

# Download and preprocess EEG from participants 1–20.
python eeg_lab.py prepare

# Optional: run only the inexpensive frequency-feature baseline and label control.
python eeg_lab.py baseline

# Extract both pretrained and random encoder features; evaluate the initial split.
python eeg_lab.py all
```

The `all` stage also runs the baseline and label control. It saves four encoder-feature files for reuse. Unlike the notebook's resume cell, rerunning this CLI stage recomputes the encoder features.

After it finishes, run repeated nested participant cross-validation:

```bash
python - <<'PY'
import numpy as np
import eeg_lab as lab
from eeg_cross_validation import run_cached_cv

with np.load(lab.ROOT / 'dataset.npz', allow_pickle=False) as saved:
    data = dict(saved)

cv_results = run_cached_cv(
    data,
    root=lab.ROOT,
    repeats=3,
    outer_folds=5,
    inner_folds=4,
    seed=42,
)
PY
```

The multiline command above works in macOS/Linux shells. On Windows, run its Python contents in the notebook or a Python script.

This stage reuses saved encoder features; it does not download assets or run the encoders again. Each outer fold tests four participants. Inner folds select classifier regularization using only the other sixteen. Feature scaling is fitted inside the training folds. Metrics are calculated per participant, then averaged.

## Run in Google Colab

1. Upload `EEG_Foundation_Lab.ipynb` to Colab.
2. Upload `eeg_lab.py`, `eeg_cross_validation.py`, `smoke_test.py`, and `requirements-colab.txt` into its working folder, usually `/content`.
3. Before importing packages, run:

   ```python
   %pip install -r requirements-colab.txt
   ```

4. Restart the runtime if requested, then run the notebook cells in order. If you installed dependencies with the command above, skip the notebook's alternative installation cell.
5. A T4 GPU helps with feature extraction. Cross-validation does not require it.

Use `requirements-colab.txt` rather than the full local requirements: it preserves Colab's existing PyTorch/torchvision and scientific-Python installations. Download outputs before the runtime is discarded.

## Files and outputs

| File or directory | Purpose |
|---|---|
| `EEG_Foundation_Lab.ipynb` | Guided experiment and plots |
| `eeg_lab.py` | Data preparation, official-model adapters and initial evaluation |
| `eeg_cross_validation.py` | Repeated nested participant cross-validation |
| `dataset.npz` | Generated EEG windows, labels and participant IDs |
| `*_features.npy` | Generated cached encoder features |
| `results.json` | Generated initial-split results |
| `cross_validation/cv_summary.csv` | Generated aggregate cross-validation metrics |
| `cross_validation/cv_participants.csv` | Generated participant-level metrics |
| `cross_validation/cv_results.json` | Generated protocol, splits, selections and results |
| `cross_validation/cv_oof_predictions.npz` | Generated held-out predictions and scores |
| `cross_validation/cv_comparison.png` | Generated comparison plot |
| `results/reported_cv_summary.*` | Previously reported Colab summaries; separate from new runs |

`asset_manifest.json`, `environment.json` (created by the notebook), and `cross_validation/cv_input_manifest.json` record provenance information. Raw data, downloaded upstream code, model weights and feature caches are excluded from Git by `.gitignore`.

## Troubleshooting

- **NumPy or SciPy attempts a source build:** verify that both pip and the notebook use Python 3.11. The local dependency pins do not target Python 3.13 or newer.
- **Missing feature file:** finish the `all` stage or the notebook's extraction cell before calling `run_cached_cv`.
- **Slow execution:** encoder extraction is the expensive stage. Cross-validation reuses those features; low GPU utilization during classifier fitting is expected.
- **Notebook imports an older helper:** restart the kernel after replacing `.py` files, then rerun the required cells.
- **Changed data or preprocessing:** remove the old encoder-feature caches and regenerate them. Do not reuse caches after changing recording selection, row order, preprocessing, model weights or pooling.

## Interpretation and references

Balanced accuracy averages recall across the two classes. Cross-validation AUROC is calculated within each participant before averaging. Repetition SD describes sensitivity to participant groupings; it is not a confidence interval.

LaBraM's pretraining included this PhysioNet corpus, so participant separation during classifier fitting does not establish independence from pretraining data. This is an exploratory study using frozen encoders and one random initialization per architecture. The label control is one participant-level label permutation, not a significance test.

- [CBraMod paper](https://openreview.net/forum?id=NPNUHgHF2w) and [official implementation](https://github.com/wjq-learning/CBraMod).
- [LaBraM paper](https://openreview.net/forum?id=QzTpTRVtrP) and [official implementation](https://github.com/935963004/LaBraM).
- [PhysioNet dataset and citation information](https://physionet.org/content/eegmmidb/1.0.0/).

Upstream model code, checkpoints and data retain their respective licenses and attribution requirements.
