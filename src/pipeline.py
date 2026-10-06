import gc
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR

from src.config import CONFIG, LINEAR_MODELS, MODEL_MAP, PARAM_SPACE_MAP
from src.hbm import fit_hierarchical_bayesian
from src.ml import analyze_feature_importance, hyperparameter_tuning_weighted, transform_linear_coefficients

def fit_hbm(train_data, scalers, random_state, hbm_samples, hbm_tune, hbm_chains, output_dir, fold_idx=None):
    unique_ids = sorted(train_data["id"].unique())
    id_to_code = {pid: idx for idx, pid in enumerate(unique_ids)}
    train_data["id_code"] = train_data["id"].map(id_to_code)

    print(f"\nHBM Training ({len(unique_ids)} patients, {hbm_samples} draws, {hbm_tune} tune, {hbm_chains} chains)")

    r_pred, std_r_pred = fit_hierarchical_bayesian(
        train_data, len(unique_ids), random_state,
        hbm_samples=hbm_samples, hbm_tune=hbm_tune, hbm_chains=hbm_chains, hbm_cores=1,
    )

    r_pred = r_pred * scalers["x"].scale_[0] / scalers["time"].scale_[0]
    std_r_pred = std_r_pred * scalers["x"].scale_[0] / scalers["time"].scale_[0]

    weights = 1 / (std_r_pred ** 2)
    weights = weights / np.mean(weights)

    hbm_df = pd.DataFrame({"id": unique_ids, "r_pred": r_pred, "std_r_pred": std_r_pred, "sample_weight": weights})
    return hbm_df


def _compute_test_true_rates(test_data, patient_ids):
    rates = {}
    for pid in patient_ids:
        pdata = test_data[test_data["id"] == pid].sort_values("time")
        rates[pid] = np.polyfit(pdata["time"].values, pdata["x"].values, 1)[0] if len(pdata) > 1 else 0.0
    return rates


def train_ml_models(patient_level_train, test_data, scalers, covariate_cols,
                    scaled_covariate_cols, output_dir, random_state,
                    fold_suffix=None, save_model=False, results_prefix="results"):
    all_results = {}

    for model_idx, model_config in MODEL_MAP.items():

        model_class = model_config["class"]
        model_name = model_config["name"]
        print(f"\n{'=' * 60}\nMODEL {model_idx}: {model_name.upper()}\n{'=' * 60}")

        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        is_linear = model_name in LINEAR_MODELS
        if is_linear:
            feature_cols = scaled_covariate_cols + ["x0_scaled"]
            y = scalers["progression_rates"].fit_transform(
                patient_level_train["r_pred"].values.reshape(-1, 1)
            ).ravel()
        else:
            feature_cols = covariate_cols + ["x0"]
            y = patient_level_train["r_pred"].values

        X = patient_level_train[feature_cols].reset_index(drop=True)
        sw = patient_level_train["sample_weight"].values

        best_params, best_score, _ = hyperparameter_tuning_weighted(
            X, y, model_class, PARAM_SPACE_MAP[model_idx], model_name,
            CONFIG["inner_cv_folds"], CONFIG["optuna_trials"], sw, output_dir
        )

        if is_linear:
            model = model_class(**best_params)
        else:
            model = model_class(**best_params, random_state=random_state)
        model.fit(X.values, y, sample_weight=sw)

        raw_features = covariate_cols + ["x0"]
        if is_linear:
            coefs, intercept = transform_linear_coefficients(model, scalers, raw_features)
            X_test = test_data.groupby("id")[raw_features].first()
            r_pred = (intercept + X_test.dot(coefs)).values
            importance_df = pd.DataFrame({"feature": raw_features, "importance_mean": coefs}).sort_values("importance_mean", ascending=False)
        else:
            X_test = test_data.groupby("id")[feature_cols].first()
            r_pred = model.predict(X_test.values)
            if "x" in test_data.columns:
                r_true = _compute_test_true_rates(test_data, X_test.index)
                r_test_true = np.array([r_true[pid] for pid in X_test.index])
                importance_df = analyze_feature_importance(model, X_test, r_test_true, raw_features)
            else:
                importance_df = pd.DataFrame({"feature": raw_features, "importance_mean": np.nan})

        id_to_r = dict(zip(X_test.index, r_pred))
        tc = test_data.copy(deep=True)
        tc["predicted_r"] = tc["id"].map(id_to_r)
        tc["predicted_score"] = tc["x0"] + tc["predicted_r"] * tc["time"]
        tc["predicted_change"] = tc["predicted_score"] - tc["x0"]

        results = {
            "ID": tc["id"].values, "time": tc["time"].values,
            "baseline": tc["x0"].values, "predicted": tc["predicted_score"].values,
            "predicted_change": tc["predicted_change"].values, "predicted_r": tc["predicted_r"].values,
        }
        if "x" in tc.columns:
            tc["actual_change"] = tc["x"] - tc["x0"]
            results["actual"] = tc["x"].values
            results["actual_change"] = tc["actual_change"].values
            corr = np.corrcoef(tc["actual_change"].values, tc["predicted_change"].values)[0, 1]
            rmse = np.sqrt(mean_squared_error(tc["actual_change"].values, tc["predicted_change"].values))
            print(f"Validation — Corr: {corr:.4f}, RMSE: {rmse:.4f}")

        suffix = fold_suffix or ""
        pd.DataFrame(results).to_csv(os.path.join(model_dir, f"{results_prefix}{suffix}.csv"), index=False)
        importance_df.to_csv(os.path.join(model_dir, f"importance{suffix}.csv"), index=False)

        all_results[model_name] = pd.DataFrame(results)

    return all_results
