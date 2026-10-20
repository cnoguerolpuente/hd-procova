# Quantifying statistical power gains from prognostic modeling in Huntington’s Disease

This repository contains the code supporting the paper “Quantifying statistical power gains from prognostic modeling in Huntington’s Disease.” The pipeline trains machine learning models on longitudinal data to predict individual disease progression rates, which can then be used as prognostic covariates in clinical trial analyses (e.g., MMRM) to improve statistical power.

## How it works

**Step 1 - Estimate progression rates.** Fits a hierarchical Bayesian model to extract a per-patient progression rate.

**Step 2 - Predict progression rates from baseline.** Trains nine Machine Learning models (Lasso, Ridge, ElasticNet, SVR, Decision Tree, Random Forest, XGBoost, LightGBM, HistGradientBoosting) to predict those rates using only baseline covariates.

![Pipeline overview](images/pipeline.png)

## Pipeline details

### 0. Data simulation

The simulated data mimics longitudinal clinical trial structure. Each patient has 1–10 follow-up visits and a baseline score drawn uniformly from {0, …, 20}. Scores evolve as:

```
xᵢₜ = x₀ᵢ + rᵢ · t + εᵢₜ
```

The progression rate `rᵢ = β₀ᵢ + Zᵢ · β` depends on 50 baseline covariates, 20 have real effects (linearly spaced from 0.02 to 0.25) and 30 are noise. Observation noise `εᵢₜ ~ N(0, 1)` is independent across visits.

```
python scripts/generate_data.py 
```

### 1. Cross-validation

10×10 repeated cross-validation (10 repetitions of 10-fold CV) on 800 training individuals, with 200 held out as a final test set. Each fold fits the Bayesian model on training patients, tunes all ML models with Optuna (5-fold inner CV), and evaluates on held-out patients.

```bash
python scripts/train_cv.py   
``` 

### 2. Final model

Trains on the full training set and evaluates on the held-out test set.

```bash
python scripts/train_final.py
```

## Project structure

```
hd-procova/
├── src/
│   ├── config.py        # project configuration
│   ├── simulate.py      # simulated data generation
│   ├── data.py          # loading, scaling, preprocessing
│   ├── hbm.py           # hierarchical Bayesian model
│   └── ml.py            # ML training
├── scripts/
│   ├── generate_data.py
│   ├── train_cv.py
│   └── train_final.py
└── notebooks/
    └── figures.ipynb
```

## Requirements

- Python ≥ 3.10
- PyMC ≥ 5.10 with NumPyro backend
