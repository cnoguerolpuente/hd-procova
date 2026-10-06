# PROCOVA for Huntington's Disease

This pipeline trains machine learning models on longitudinal data to predict individual disease progression rates, which can then be used as prognostic covariates in clinical trial analysis frameworks (e.g., MMRM) to improve statistical power.

## Method

**Step 1 — Progression Rate Estimation:**
fits a random-slope linear mixed model to extract per-patient progression rates from longitudinal clinical scores.

Progression Rate Estimation. Hierarchical Bayesian linear regression over observed clinical scores to estimate individual annual progression rates for 5 UHDRS endpoints.  

**Step 2 — Progression Rate Prediction:**
trains 9 supervised regressors (Lasso, Ridge, ElasticNet, SVR, Decision Tree, Random Forest, XGBoost, LightGBM, HistGradientBoosting) to predict the progression rates estimated in Step 1 using from baseline covariates.

![Pipeline overview](images/pipeline.png)

## Quick start

```bash
# Install dependencies
pip install -e .

# Generate simulated data
python scripts/generate_data.py

# Run cross-validation 
python scripts/train_cv.py 

# Train final model
python scripts/train_final.py
```

## Project structure

```
hd-procova/
├── src/
│   ├── config.py        # Model registry, hyperparameter spaces, global config
│   ├── simulate.py      # Simulated longitudinal data generation
│   ├── data.py          # Data loading, scaling, preprocessing
│   ├── hbm.py           # Hierarchical Bayesian Model (PyMC/MCMC)
│   └── ml.py            # ML training, Optuna tuning, feature importance
├── scripts/
│   ├── generate_data.py # Generate simulated train/test datasets
│   ├── train_cv.py      # 10x10 repeated cross-validation pipeline
│   └── train_final.py   # Final model training on full data
└── notebooks/
    └── figures.ipynb    # Result visualizations
```


## Pipeline details

### 0. Data simulation

The simulated data mimics the structure of longitudinal clinical trial data. Each individual *i* has a random number of follow-up visits (1–10) and a baseline score *x₀ᵢ* drawn uniformly from {0, …, 20}.

The observed score at time *t* is:

```
xᵢₜ = x₀ᵢ + rᵢ · t + εᵢₜ
```

where:

- **Progression rate:** `rᵢ = β₀ᵢ + Zᵢ · β`, with `β₀ᵢ ~ N(0, 1)` capturing unexplained individual variation.
- **Covariates:** `Zᵢ ~ N(0, I₅₀)`, 50 baseline covariates per individual.
- **Covariate effects:** 30 covariates are irrelevant (`β = 0`); the remaining 20 have effects linearly spaced from 0.02 to 0.25.
- **Observation noise:** `εᵢₜ ~ N(0, 1)`, independent across visits.

```python
from src.simulate import create_simulated_dataset
df = create_simulated_dataset(n_individuals=1000)
```

### 1. Cross-validation (`train_cv.py`)

Implements 10x10 repeated cross-validation (10 repetitions of 10-fold outer CV) on 800 training individuals (200 held out as final test set). Each fold:
1. Splits patients into train/validation (patient-level, no leakage)
2. Fits the HBM on training patients to extract progression rates
3. Trains all ML models with Optuna hyperparameter tuning (5-fold inner CV)
4. Evaluates on held-out validation patients

```bash
# Run fold 1
python scripts/train_cv.py 1

# Run all 100 folds (e.g., via job array)
for i in $(seq 1 100); do python scripts/train_cv.py $i; done
```

### 2. Final model (`train_final.py`)

Trains on the complete training set and evaluates on the held-out test set. Saves fitted models, scalers, and deployment artefacts.

```bash
python scripts/train_final.py
```

## Requirements

- Python >= 3.10
- PyMC >= 5.10 with NumPyro backend
- See `pyproject.toml` for full dependency list

GPU acceleration (optional): Install JAX with CUDA support for 10-20x faster MCMC sampling.
