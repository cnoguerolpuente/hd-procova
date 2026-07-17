#!/usr/bin/env python3
"""
Cross-Validation Pipeline: HBM + ML Models 
=============================================================
Two-stage pipeline for predicting individual progression rates:

Stage 1 - Hierarchical Bayesian Model (HBM):
    Fit a random-slope model via MCMC to extract per-patient
    progression rates from longitudinal scores.

Stage 2 - Machine Learning Models:
    Train supervised regressors to predict the progression rates from baseline covariates.

Usage:
    python scripts/train_cv.py <fold_idx>
    # fold_idx: 1-100 (10 repetitions x 10 folds)
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "mode=FAST_COMPILE,linker=py,cxx=")
os.environ["PYTENSOR_CXX"] = ""

import gc
import json
import sys
import time
import traceback
from typing import Dict, List

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytensor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR

pytensor.config.cxx = ""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG, LINEAR_MODELS, MODEL_MAP, PARAM_SPACE_MAP
from src.hbm import (
    check_convergence,
    fit_hierarchical_bayesian,
    save_hbm_convergence_plot,
    train_full_hbm_with_covariates,
)
from src.ml import (
    analyze_feature_importance,
    hyperparameter_tuning_weighted,
    transform_linear_coefficients,
)
from src.data import load_data, remove_baseline_visits, scale_pipeline_data

random_state: int = CONFIG.get("random_state", 42)
np.random.seed(random_state)

HBM_SAMPLES: int = 500
HBM_TUNE: int = 500
HBM_CHAINS: int = 2


def main(fold_idx: int = 0) -> None:
    """Run the two-level CV pipeline for a single outer fold."""

    # --- Load data ---
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    score_data = load_data(os.path.join(data_dir, "train.csv"))
    patient_ids = score_data["id"].unique()

    repetition = fold_idx // 10
    fold_within_rep = fold_idx % 10

    print("=" * 80)
    print(
        f"REPETITION {repetition + 1}/10, FOLD {fold_within_rep + 1}/10 "
        f"(global {fold_idx + 1}/100)"
    )
    print("=" * 80)

    # --- Outer CV split ---
    rep_seed = random_state + repetition
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=rep_seed)
    cv_splits = list(outer_cv.split(patient_ids))
    train_idx, val_idx = cv_splits[fold_within_rep]

    train_ids = patient_ids[train_idx]
    val_ids = patient_ids[val_idx]

    train_data = score_data[score_data["id"].isin(train_ids)].copy()
    test_data = score_data[score_data["id"].isin(val_ids)].copy()

    print(f"Train: {len(train_ids)} patients, Test: {len(val_ids)} patients")

    # --- Preprocessing ---
    scalers, covariate_cols, scaled_covariate_cols = scale_pipeline_data(train_data)
    train_data, test_data = remove_baseline_visits(train_data, test_data)

    unique_train_ids = sorted(train_data["id"].unique())
    train_id_to_code = {pid: idx for idx, pid in enumerate(unique_train_ids)}
    train_data["id_code"] = train_data["id"].map(train_id_to_code)

    unique_val_ids = sorted(test_data["id"].unique())
    val_id_to_code = {pid: idx for idx, pid in enumerate(unique_val_ids)}
    test_data["id_code"] = test_data["id"].map(val_id_to_code)

    # --- HBM training ---
    print(f"\nHBM Training ({len(unique_train_ids)} patients, "
          f"{HBM_SAMPLES} draws, {HBM_TUNE} tune, {HBM_CHAINS} chains)")

    r_pred_hbm, std_r_pred_hbm, gamma0, trace = fit_hierarchical_bayesian(
        train_data,
        train_data["id_code"].nunique(),
        random_state,
        hbm_samples=HBM_SAMPLES,
        hbm_tune=HBM_TUNE,
        hbm_chains=HBM_CHAINS,
        hbm_cores=1,
    )

    r_pred_hbm = r_pred_hbm * scalers["x"].scale_[0] / scalers["time"].scale_[0]
    std_r_pred_hbm = std_r_pred_hbm * scalers["x"].scale_[0] / scalers["time"].scale_[0]

    sample_weights = 1 / (std_r_pred_hbm ** 2)
    sample_weights = sample_weights / np.mean(sample_weights)

    # --- Convergence check ---
    main_params = ["mu_beta", "sigma_beta", "gamma0", "sigma_obs"]
    main_ok, max_rhat, min_ess = check_convergence(
        trace, var_names=main_params, rhat_thresh=1.1, ess_thresh=200
    )
    beta_ok, beta_rhat, beta_ess = check_convergence(
        trace, var_names=["beta"], rhat_thresh=1.1, ess_thresh=100
    )

    if not main_ok or not beta_ok:
        print(f"WARNING: Convergence issues — Main R-hat: {max_rhat:.3f}, Beta R-hat: {beta_rhat:.3f}")

    # Save HBM diagnostics
    hbm_dir = os.path.join(CONFIG["output_dir"], "training", "hbm")
    os.makedirs(hbm_dir, exist_ok=True)
    save_hbm_convergence_plot(trace, hbm_dir, fold_idx + 1)
    del trace
    gc.collect()

    # Merge HBM estimates into train_data
    hbm_estimates_df = pd.DataFrame({
        "id": unique_train_ids,
        "r_pred": r_pred_hbm,
        "std_r_pred": std_r_pred_hbm,
        "sample_weight": sample_weights,
    })
    train_data = train_data.merge(
        hbm_estimates_df[["id", "r_pred", "std_r_pred", "sample_weight"]],
        on="id", how="left",
    )
    train_data["x_pred"] = train_data["x0"] + train_data["r_pred"] * train_data["time"]

    raw_covariate_cols: List[str] = covariate_cols.copy()
    patient_level_train = train_data.groupby("id").first()

    # --- Full HBM with covariates ---
    print("\n" + "=" * 60)
    print("MODEL: HIERARCHICAL (Full Bayesian with Covariates)")
    print("=" * 60)

    model_times: Dict[str, float] = {}
    start_hbm = time.time()

    try:
        hierarchical_results, hierarchical_trace, hierarchical_importance = (
            train_full_hbm_with_covariates(
                train_data, test_data, covariate_cols, scalers,
                random_state=random_state,
            )
        )

        hierarchical_dir = os.path.join(CONFIG["output_dir"], "training", "hierarchical")
        os.makedirs(hierarchical_dir, exist_ok=True)

        pd.DataFrame(hierarchical_results).to_csv(
            f"{hierarchical_dir}/results_fold_{fold_idx + 1}.csv", index=False
        )
        hierarchical_importance.to_csv(
            f"{hierarchical_dir}/importance_fold_{fold_idx + 1}.csv", index=False
        )
        save_hbm_convergence_plot(hierarchical_trace, hierarchical_dir, fold_idx + 1)

        actual = np.array(hierarchical_results["actual_change"])
        predicted = np.array(hierarchical_results["predicted_change"])
        print(
            f"HBM Validation — Corr: {np.corrcoef(actual, predicted)[0,1]:.4f}, "
            f"RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.4f}"
        )

        model_times["hierarchical"] = time.time() - start_hbm

    except Exception:
        print(f"Full HBM failed:\n{traceback.format_exc()}")

    # --- Sklearn ML models ---
    for model_idx, model_config in MODEL_MAP.items():
        if model_config["class"] is None:
            continue

        model_class = model_config["class"]
        model_name: str = model_config["name"]
        param_space = PARAM_SPACE_MAP[model_idx]
        start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"MODEL {model_idx}: {model_name.upper()}")
        print("=" * 60)

        model_output_dir = os.path.join(CONFIG["output_dir"], "training", model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        r_pred_hbm_ml = patient_level_train["r_pred"].values
        sw = patient_level_train["sample_weight"].values

        if model_name in LINEAR_MODELS:
            feature_cols = scaled_covariate_cols + ["x0_scaled"]
            y = scalers["progression_rates"].fit_transform(
                r_pred_hbm_ml.reshape(-1, 1)
            ).ravel()
        else:
            feature_cols = raw_covariate_cols + ["x0"]
            y = r_pred_hbm_ml

        X = patient_level_train[feature_cols].reset_index(drop=True)

        best_params, best_score, _ = hyperparameter_tuning_weighted(
            X, y, model_class, param_space, model_name,
            CONFIG["inner_cv_folds"], CONFIG["scoring_metric"],
            CONFIG["optuna_trials"], sw,
        )

        deterministic_classes = (Lasso, Ridge, ElasticNet, LinearSVR)
        if model_class in deterministic_classes:
            final_model = model_class(**best_params)
        else:
            final_model = model_class(**best_params, random_state=random_state)

        X_train = X.values if hasattr(X, "values") else X
        final_model.fit(X_train, y, sample_weight=sw)

        if model_name in LINEAR_MODELS:
            transformed_coefs, transformed_intercept = transform_linear_coefficients(
                final_model, scalers, raw_covariate_cols + ["x0"]
            )
            X_test_unscaled = test_data.groupby("id")[raw_covariate_cols + ["x0"]].first()
            r_pred = (transformed_intercept + X_test_unscaled.dot(transformed_coefs)).values

            importance_df = pd.DataFrame({
                "feature": raw_covariate_cols + ["x0"],
                "importance_mean": transformed_coefs,
            }).sort_values("importance_mean", ascending=False)
        else:
            X_test_unscaled = test_data.groupby("id")[feature_cols].first()
            X_test_array = X_test_unscaled.values if hasattr(X_test_unscaled, "values") else X_test_unscaled
            r_pred = final_model.predict(X_test_array)

            r_test_true_dict: Dict[str, float] = {}
            for pid in test_data["id"].unique():
                pdata = test_data[test_data["id"] == pid].sort_values("time")
                if len(pdata) > 1:
                    r_test_true_dict[pid] = np.polyfit(pdata["time"].values, pdata["x"].values, 1)[0]
                else:
                    r_test_true_dict[pid] = 0.0

            r_test_true = np.array([r_test_true_dict[pid] for pid in X_test_unscaled.index])
            importance_df = analyze_feature_importance(
                final_model, X_test_unscaled, r_test_true, raw_covariate_cols + ["x0"],
            )

        id_to_pred_r = dict(zip(X_test_unscaled.index, r_pred))
        test_data_copy = test_data.copy(deep=True)
        test_data_copy["predicted_r"] = test_data_copy["id"].map(id_to_pred_r)
        test_data_copy["predicted_score"] = (
            test_data_copy["x0"] + test_data_copy["predicted_r"] * test_data_copy["time"]
        )
        test_data_copy["actual_change"] = test_data_copy["x"] - test_data_copy["x0"]
        test_data_copy["predicted_change"] = test_data_copy["predicted_score"] - test_data_copy["x0"]

        actual_change = test_data_copy["actual_change"].values
        predicted_change = test_data_copy["predicted_change"].values

        test_corr = np.corrcoef(actual_change, predicted_change)[0, 1]
        test_rmse = np.sqrt(mean_squared_error(actual_change, predicted_change))
        print(f"Validation — Corr: {test_corr:.4f}, RMSE: {test_rmse:.4f}")

        ml_results = {
            "baseline": test_data_copy["x0"].values,
            "actual": test_data_copy["x"].values,
            "predicted": test_data_copy["predicted_score"].values,
            "actual_change": actual_change,
            "predicted_change": predicted_change,
            "predicted_r": test_data_copy["predicted_r"].values,
            "NVISIT": test_data_copy["NVISIT"].values,
            "ID": test_data_copy["id"].values,
        }

        pd.DataFrame(ml_results).to_csv(
            f"{model_output_dir}/results_fold_{fold_idx + 1}.csv", index=False
        )
        importance_df.to_csv(
            os.path.join(model_output_dir, f"importance_fold_{fold_idx + 1}.csv"), index=False,
        )
        with open(f"{model_output_dir}/hyperparams_fold_{fold_idx + 1}.json", "w") as fh:
            json.dump({"fold": fold_idx + 1, "model": model_name,
                        "best_params": best_params, "best_cv_score": best_score}, fh, indent=2)

        model_times[model_name] = time.time() - start_time
        print(f"{model_name} completed in {model_times[model_name]:.1f} s")
        gc.collect()

    print("\nTIME SUMMARY:")
    for name, duration in model_times.items():
        print(f"  {name:<20}: {duration / 60:.2f} min")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/train_cv.py <fold_idx>")
        print("  fold_idx: 1-100 (1-based)")
        sys.exit(1)

    start_overall = time.time()

    try:
        fold_idx = int(sys.argv[1]) - 1
    except ValueError:
        print("Error: fold_idx must be an integer between 1 and 100")
        sys.exit(1)

    if fold_idx < 0 or fold_idx >= 100:
        print("Error: fold_idx must be between 1 and 100")
        sys.exit(1)

    repetition = fold_idx // 10 + 1
    fold_in_rep = fold_idx % 10 + 1
    print(f"Pipeline — repetition={repetition}/10, fold={fold_in_rep}/10 (global {fold_idx + 1}/100)")
    main(fold_idx=fold_idx)

    total = time.time() - start_overall
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    print(f"\nCompleted in {int(h)}h {int(m)}m {s:.1f}s")
