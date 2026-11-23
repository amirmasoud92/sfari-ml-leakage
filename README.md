# sfari-ml-leakage

Analysis code for the article **"Separating child identity from diagnostic information in machine learning
classification of autism from EEG"** (A. Ahmadi and F. Ghorbani Lohesara).

EEG recorded from the same child is highly individual. When blocks, segments or epochs of one child appear in both
the training and the test set, a classifier can reach high accuracy by recognising the child instead of the
diagnosis. This repository contains the methods used to measure this identity signal, to test classifiers for
identity leakage, and to evaluate diagnostic classification with children kept whole. It also contains the data
and code for every data-driven figure panel.

The repository is organised by method. It documents the analyses reported in the article and is not a single
end-to-end pipeline.

## Contents

| Folder | Methods |
|---|---|
| [`preprocessing/`](preprocessing) | Bad-channel detection with a robust reference (PyPREP), segment-wise FIR filtering, ICA with ICLabel, interpolation, average reference, artefact subspace reconstruction, quality-control tables |
| [`features/`](features) | Event-defined blocks, 1-Hz log power (2–34 Hz, 64 channels), parameterised spectra, band covariances in tangent space, phase-lag connectivity, evoked and behavioural indices |
| [`identification/`](identification) | Cross-paradigm identification of children with child-bootstrap intervals; controls for spectral shape, surface Laplacian, muscle artefact, amplifier system and recording visit; self-identifiability |
| [`variance_partition/`](variance_partition) | Between-child and between-group shares of block-level variance; variance components |
| [`leakage_tests/`](leakage_tests) | Split contrast, permuted-label control, dose-response, model families, positive controls, cross-paradigm and cross-visit leakage, size-matched control, number of children |
| [`deep_learning/`](deep_learning) | EEGNet and ShallowConvNet, permuted-label control for EEGNet, subject-contrastive encoder |
| [`classification/`](classification) | Child-level folds, repeated nested cross-validation, multi-view factor model (MOFA+), AUC estimators, null calibration, permutation and likelihood-ratio tests, robustness analyses |
| [`markers/`](markers) | Split-half reliability, age- and sex-adjusted effect sizes, error-corrected AUC, normative deviations, published group effects |
| [`stats/`](stats) | Bootstrap and permutation helpers, multiple-comparison correction, seeded generators, sample characteristics |
| [`identity_audit/`](identity_audit) | Stand-alone leakage audit for any scikit-learn classifier, with tests |
| [`figures/`](figures) | Source data and plotting script for each data-driven figure panel |
| [`config/`](config), [`utils/`](utils) | Analysis parameters, configuration loader, dataset access and shared helpers |

## Data

The analyses use the OpenNeuro dataset [ds006780](https://openneuro.org/datasets/ds006780) (version 1.0.0, BIDS,
CC0), with 64-channel EEG from autistic, typically developing and sibling children in seven paradigms. The dataset
is not included in this repository. Group labels and covariates are read from the dataset's `participants.tsv`.

## Installation

```bash
git clone https://github.com/amirmasoud92/sfari-ml-leakage.git
cd sfari-ml-leakage
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The analyses were run with Python 3.11. `requirements.txt` lists the package versions used.

## Usage

### Analyses

Set `paths.data_root` in [`config/analysis.yaml`](config/analysis.yaml) to the local copy of ds006780. All other
parameters are defined in the same file. Analysis modules are run from the repository root, for example:

```bash
python -m leakage_tests.split_contrast
```

Each module's docstring describes its inputs, the procedure and the article section, figure or table it
corresponds to. Random generators are derived from the master seed in `config/analysis.yaml` and a fixed key per
analysis ([`stats/rng.py`](stats/rng.py)), so permutations, bootstraps and fold assignments are deterministic.

### Identity leakage audit

[`identity_audit`](identity_audit/identity_audit.py) works with any data that has several blocks per subject and
does not need the dataset or the configuration:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from identity_audit import identity_audit as ia

# X: (n_blocks, n_features); y: subject label of each block; groups: subject ID of each block
def make_model(seed):
    return make_pipeline(StandardScaler(), PCA(10, random_state=seed), LogisticRegression(max_iter=1000))

contrast = ia.split_contrast(X, y, groups, make_model, repeats=10)   # subjects kept whole vs blocks split
null = ia.permuted_label_control(X, y, groups, make_model, n_perm=40)
dose = ia.dose_response(X, y, groups, make_model)                    # AUC as k blocks per test subject leak
```

The tests in [`identity_audit/tests`](identity_audit/tests) use synthetic data in which the features carry subject
identity and the labels carry no information:

```bash
pytest identity_audit/tests
```

### Figures

Each data-driven panel has a folder such as `figures/figure_3/panel_b/` that contains the plotted values (CSV) and
a `plot.py` that draws the panel:

```bash
python figures/figure_3/panel_b/plot.py
```

Schematic panels are not included. See [`figures/README.md`](figures/README.md) for the list of panels.

## Citation

If you use this code, please cite the article above. Full citation details will be added on publication.

## License

The code and the figure data are released under the [MIT License](LICENSE). The EEG dataset is distributed
separately by OpenNeuro under CC0.

## Contact

Amirmasoud Ahmadi, Max Planck Institute for Biological Intelligence, Seewiesen, Germany  
Amirmasoud.Ahmadi@bi.mpg.de
