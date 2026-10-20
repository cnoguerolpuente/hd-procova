#!/usr/bin/env python3
"""
Final model training on full train set, evaluated on held-out test set.
Usage: python scripts/train_final.py
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "mode=FAST_COMPILE,linker=py,cxx=")
os.environ["PYTENSOR_CXX"] = ""

import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytensor
from sklearn.metrics import r2_score

pytensor.config.cxx = ""
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG
from src.data import scale_pipeline_data
from src.pipeline import fit_hbm, train_ml_models

random_state = CONFIG.get("random_state", 42)
np.random.seed(random_state)


def train_final_model():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))

    scalers, covariate_cols, scaled_covariate_cols = scale_pipeline_data(train_data)

    output_dir = os.path.join(CONFIG["output_dir"], "final")
    hbm_df = fit_hbm(train_data, scalers, random_state, 1000, 1000, 4, output_dir)
    
    train_data = train_data.merge(hbm_df[["id", "r_pred", "std_r_pred", "sample_weight"]], on="id", how="left")
    train_data["x_pred"] = train_data["x0"] + train_data["r_pred"] * train_data["time"]

    patient_level = train_data.groupby("id").first()

    all_results = train_ml_models(
        patient_level, test_data, scalers, covariate_cols, scaled_covariate_cols,
        output_dir, random_state, save_model=True, results_prefix="predictions",
    )

    return all_results


if __name__ == "__main__":
    train_final_model()