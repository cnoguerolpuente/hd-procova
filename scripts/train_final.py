#!/usr/bin/env python3
"""
Final Model Training: HBM + ML on Full Simulated Data
======================================================
Trains production-ready models on the full training set and evaluates
on the held-out test set.

Usage:
    python scripts/train_final.py
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
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytensor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
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

HBM_SAMPLES: int = 800
HBM_TUNE: int = 800
HBM_CHAINS: int = 2


def train_final_model() -> Dict[str, pd.DataFrame]:
    """Train and evaluate all models on full data."""
    pipeline_start = time.time()
    print("=" * 80)
    print("FINAL MODEL TRAINING PIPELINE")
    print(f"HBM: {HBM_SAMPLES} samples | {HBM_TUNE} tune | {HBM_CHAINS} chains")
    print("=" * 80)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    train_data = load_data(os.path.join(data_dir, "train.csv"))
    test_data = load_data(os.path.join(data_dir, "test.csv"))

    n_train = train_data["id"].nunique()
    n_test = test_data["id"].nunique()
    print(f"Data: {len(train_data)} train obs ({n_train} patients), "
          f"{len(test_data)} test obs ({n_test} patients)")

    # --- Preprocessing ---
    scalers, covariate_cols, scaled_covariate_cols = scale_pipeline_data(train_data)
    train_data, test_data = remove_baseline_visits(train_data, test_data, drop_test_avisit=False)

    unique_train_ids = sorted(train_data["id"].unique())
    train_id_to_code = {pid: idx for idx, pid in enumerate(unique_train_ids)}
    train_data["id_code"] = train_data["id"].map(train_id_to_code)

    # --- HBM training ---
    print(f"\nHBM Training ({len(unique_train_ids)} patients)")

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

    # Convergence
    main_params = ["mu_beta", "sigma_beta", "gamma0", "sigma_obs"]
    main_ok, max_rhat, min_ess = check_convergence(
        trace, var_names=main_params, rhat_thresh=1.1, ess_thresh=200
    )
    print(f"Convergence: {'OK' if main_ok else 'MARGINAL'} (R-hat: {max_rhat:.3f}, ESS: {min_ess})")

    hbm_dir = os.path.join(CONFIG["output_dir"], "final", "hbm")
    os.makedirs(hbm_dir, exist_ok=True)
    save_hbm_convergence_plot(trace, hbm_dir)
    del trace
    gc.collect()

    # Merge HBM estimates
    hbm_df = pd.DataFrame({
        "id": unique_train_ids,
        "r_pred": r_pred_hbm,
        "std_r_pred": std_r_pred_hbm,
        "sample_weight": sample_weights,
    })
    hbm_df.to_csv(os.path.join(hbm_dir, "hbm_estimates.csv"), index=False)

    train_data = train_data.merge(
        hbm_df[["id", "r_pred", "std_r_pred", "sample_weight"]],
        on="id", how="left",
    )
    train_data["x_pred"] = train_data["x0"] + train_data["r_pred"] * train_data["time"]

    train_r2 = r2_score(train_data["x"], train_data["x_pred"])
    print(f"In-sample HBM R2: {train_r2:.4f}")

    raw_covariate_cols: List[str] = covariate_cols.copy()
    patient_level_train = train_data.groupby("id").first()

    all_results: Dict[str, pd.DataFrame] = {}
    model_times: Dict[str, float] = {}

    # --- Full HBM with covariates ---
    print("\n" + "=" * 60)
    print("MODEL: HIERARCHICAL (Full Bayesian with Covariates)")
    print("=" * 60)

    start_hbm = time.time()
    try:
        hier_results, hier_trace, hier_importance = train_full_hbm_with_covariates(
            train_data, test_data, covariate_cols, scalers,
            random_state=random_state,
            hbm_samples=HBM_SAMPLES, hbm_tune=HBM_TUNE,
            hbm_chains=HBM_CHAINS, hbm_cores=1,
        )

        hier_dir = os.path.join(CONFIG["output_dir"], "final", "hierarchical")
        os.makedirs(hier_dir, exist_ok=True)
        pd.DataFrame(hier_results).to_csv(f"{hier_dir}/predictions.csv", index=False)
        hier_importance.to_csv(f"{hier_dir}/importance.csv", index=False)
        save_hbm_convergence_plot(hier_trace, hier_dir)

        if "actual_change" in hier_results:
            actual = np.array(hier_results["actual_change"])
            predicted = np.array(hier_results["predicted_change"])
            print(f"HBM Validation — Corr: {np.corrcoef(actual, predicted)[0,1]:.4f}, "
                  f"RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.4f}")

        all_results["hierarchical"] = pd.DataFrame(hier_results)
        model_times["hierarchical"] = time.time() - start_hbm

    except Exception:
        print(f"Full HBM failed:\n{traceback.format_exc()}")

    # --- Sklearn models ---
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

        model_dir = os.path.join(CONFIG["output_dir"], "final", model_name)
        os.makedirs(model_dir, exist_ok=True)

        r_pred_ml = patient_level_train["r_pred"].values
        sw = patient_level_train["sample_weight"].values

        if model_name in LINEAR_MODELS:
            feature_cols = scaled_covariate_cols + ["x0_scaled"]
            y = scalers["progression_rates"].fit_transform(
                r_pred_ml.reshape(-1, 1)
            ).ravel()
        else:
            feature_cols = raw_covariate_cols + ["x0"]
            y = r_pred_ml

        X = patient_level_train[feature_cols].reset_index(drop=True)

        best_params, best_score, study = hyperparameter_tuning_weighted(
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
            X_test = test_data.groupby("id")[raw_covariate_cols + ["x0"]].first()
            r_pred = (transformed_intercept + X_test.dot(transformed_coefs)).values

            importance_df = pd.DataFrame({
                "feature": raw_covariate_cols + ["x0"],
                "importance_mean": transformed_coefs,
            }).sort_values("importance_mean", ascending=False)
        else:
            X_test = test_data.groupby("id")[feature_cols].first()
            r_pred = final_model.predict(
                X_test.values if hasattr(X_test, "values") else X_test
            )

            feature_cols_imp = raw_covariate_cols + ["x0"]
            if "x" in test_data.columns:
                r_true = {}
                for pid in test_data["id"].unique():
                    pdata = test_data[test_data["id"] == pid].sort_values("time")
                    if len(pdata) > 1:
                        r_true[pid] = np.polyfit(pdata["time"].values, pdata["x"].values, 1)[0]
                    else:
                        r_true[pid] = 0.0
                r_test_true = np.array([r_true[pid] for pid in X_test.index])
                importance_df = analyze_feature_importance(
                    final_model, X_test, r_test_true, feature_cols_imp,
                )
            else:
                importance_df = pd.DataFrame({
                    "feature": feature_cols_imp,
                    "importance_mean": np.nan,
                })

        # Reconstruct predictions
        id_to_r = dict(zip(X_test.index, r_pred))
        tc = test_data.copy(deep=True)
        tc["predicted_r"] = tc["id"].map(id_to_r)
        tc["predicted_score"] = tc["x0"] + tc["predicted_r"] * tc["time"]
        tc["predicted_change"] = tc["predicted_score"] - tc["x0"]

        has_outcome = "x" in tc.columns
        results: Dict = {
            "ID": tc["id"].values,
            "NVISIT": tc["NVISIT"].values,
            "baseline": tc["x0"].values,
            "predicted": tc["predicted_score"].values,
            "predicted_change": tc["predicted_change"].values,
            "predicted_r": tc["predicted_r"].values,
        }
        if has_outcome:
            tc["actual_change"] = tc["x"] - tc["x0"]
            results["actual"] = tc["x"].values
            results["actual_change"] = tc["actual_change"].values

            actual_change = tc["actual_change"].values
            predicted_change = tc["predicted_change"].values
            print(f"Validation — Corr: {np.corrcoef(actual_change, predicted_change)[0,1]:.4f}, "
                  f"RMSE: {np.sqrt(mean_squared_error(actual_change, predicted_change)):.4f}")

        if "AVISIT" in tc.columns:
            results["AVISIT"] = tc["AVISIT"].values

        pd.DataFrame(results).to_csv(f"{model_dir}/predictions.csv", index=False)
        importance_df.to_csv(os.path.join(model_dir, "importance.csv"), index=False)
        joblib.dump(final_model, os.path.join(model_dir, "model.joblib"))
        with open(os.path.join(model_dir, "hyperparams.json"), "w") as fh:
            json.dump({"model": model_name, "best_params": best_params,
                        "best_cv_score": best_score}, fh, indent=2)

        all_results[model_name] = pd.DataFrame(results)
        model_times[model_name] = time.time() - start_time
        print(f"{model_name} completed in {model_times[model_name]:.1f} s")
        gc.collect()

    # --- Deployment artefacts ---
    deploy_dir = os.path.join(CONFIG["output_dir"], "final", "deployment")
    os.makedirs(deploy_dir, exist_ok=True)
    joblib.dump(scalers, os.path.join(deploy_dir, "scalers.joblib"))
    with open(os.path.join(deploy_dir, "gamma0.json"), "w") as fh:
        json.dump({"gamma0": float(gamma0)}, fh, indent=2)

    total = time.time() - pipeline_start
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)

    print("\nTIME SUMMARY:")
    for name, dur in model_times.items():
        print(f"  {name:<20}: {dur / 60:.2f} min")
    print(f"\nTotal: {int(h)}h {int(m)}m {s:.1f}s")

    return all_results


if __name__ == "__main__":
    train_final_model()
    print("Training complete.")
