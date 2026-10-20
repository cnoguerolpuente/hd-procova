#!/usr/bin/env python3
"""
Cross-validation pipeline (10x10 repeated CV).
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "mode=FAST_COMPILE,linker=py,cxx=")
os.environ["PYTENSOR_CXX"] = ""

import sys

import numpy as np
import pandas as pd
import pytensor
from sklearn.model_selection import KFold

pytensor.config.cxx = ""
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG
from src.data import scale_pipeline_data
from src.pipeline import fit_hbm, train_ml_models

random_state = CONFIG.get("random_state", 42)
np.random.seed(random_state)


def main(fold_idx):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    score_data = pd.read_csv(os.path.join(data_dir, "train.csv"))

    patient_ids = score_data["id"].unique()

    repetition = fold_idx // 10
    fold_within_rep = fold_idx % 10
    print(f"REPETITION {repetition + 1}/10, FOLD {fold_within_rep + 1}/10 (global {fold_idx + 1}/100)")

    outer_cv = KFold(n_splits=10, shuffle=True, random_state=random_state + repetition)
    train_idx, val_idx = list(outer_cv.split(patient_ids))[fold_within_rep]
    train_data = score_data[score_data["id"].isin(patient_ids[train_idx])].copy()
    val_data = score_data[score_data["id"].isin(patient_ids[val_idx])].copy()

    scalers, covariate_cols, scaled_covariate_cols = scale_pipeline_data(train_data)

    output_dir = os.path.join(CONFIG["output_dir"], "training")
    hbm_df = fit_hbm(train_data, scalers, random_state, 1000, 1000, 4, output_dir, fold_idx + 1)

    train_data = train_data.merge(hbm_df[["id", "r_pred", "std_r_pred", "sample_weight"]], on="id", how="left")
    train_data["x_pred"] = train_data["x0"] + train_data["r_pred"] * train_data["time"]
    patient_level = train_data.groupby("id").first()

    train_ml_models(
        patient_level, val_data, scalers, covariate_cols, scaled_covariate_cols,
        output_dir, random_state, fold_suffix=f"_fold_{fold_idx + 1}",
    )


if __name__ == "__main__":
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of folds to run in parallel")
    args = parser.parse_args()

    if args.n_jobs > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=ctx) as executor:
            list(executor.map(main, range(100)))
    else:
        for fold_idx in range(100):
            print(f"Starting fold {fold_idx + 1}/100")
            main(fold_idx)
