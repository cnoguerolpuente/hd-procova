from src.config import CONFIG
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import numpy as np
import pandas as pd
import warnings

RANDOM_STATE = CONFIG["random_state"]
SCORING = CONFIG["scoring_metric"]


def hyperparameter_tuning_weighted(
    X, y, model_class, param_space_fn, model_name,
    inner_cv, scoring, n_trials, sample_weights,
):
    """Hyperparameter tuning via Optuna with precision-weighted CV."""
    warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    scoring = (scoring or "neg_mean_squared_error").lower()

    def objective(trial):
        params = param_space_fn(trial)
        cv_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []

        for train_idx, val_idx in cv_splitter.split(X):
            if hasattr(X, "iloc"):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]

            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            w_train_fold = sample_weights[train_idx]

            fold_model = model_class(**params)
            fold_model.fit(X_train_fold, y_train_fold, sample_weight=w_train_fold)
            y_pred = fold_model.predict(X_val_fold)

            if scoring == "neg_mean_squared_error":
                fold_score = -mean_squared_error(y_val_fold, y_pred)
            elif scoring == "r2":
                fold_score = r2_score(y_val_fold, y_pred)
            elif scoring in ("pearson", "pearson_correlation", "corr", "correlation"):
                y_val_std = np.std(y_val_fold)
                y_pred_std = np.std(y_pred)
                if y_val_std < 1e-12 or y_pred_std < 1e-12:
                    fold_score = 0.0
                else:
                    fold_score = float(np.corrcoef(y_val_fold, y_pred)[0, 1])
                    if not np.isfinite(fold_score):
                        fold_score = 0.0
            else:
                raise ValueError(f"Unsupported scoring metric: {scoring}")

            cv_scores.append(fold_score)

        mean_score = float(np.nanmean(cv_scores))
        if not np.isfinite(mean_score):
            mean_score = -1e12
        return mean_score

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    best_score = study.best_value

    print(f"Tuning complete for {model_name.upper()} | Best score ({scoring}): {best_score:.4f}")
    print(f"Best hyperparameters: {best_params}")

    return best_params, best_score, study


def analyze_feature_importance(model, X_test, y_test, features):
    """Feature importance via native importances or permutation importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = pd.DataFrame(
            {"feature": features, "importance_mean": importances}
        ).sort_values("importance_mean", ascending=True)
    else:
        result = permutation_importance(
            estimator=model,
            X=X_test,
            y=y_test,
            scoring="neg_mean_squared_error",
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        importance_df = pd.DataFrame(
            {"feature": features, "importance_mean": result.importances_mean}
        ).sort_values("importance_mean", ascending=True)

    return importance_df


def transform_linear_coefficients(
    model,
    scalers: dict,
    covariate_cols: list,
) -> tuple[np.ndarray, float]:
    """Transform linear model coefficients from scaled to original space.

    The last element of covariate_cols must be "x0".
    """
    covariate_scaler = scalers["covariates"]
    progression_rate_scaler = scalers["progression_rates"]

    scaled_coefs = model.coef_
    scale_r: float = progression_rate_scaler.scale_[0]
    center_r: float = progression_rate_scaler.mean_[0]

    scales = np.append(covariate_scaler.scale_, scalers["x"].scale_[0])
    centers = np.append(covariate_scaler.mean_, scalers["x"].mean_[0])

    transformed_coefs = (scaled_coefs * scale_r) / scales

    intercept_val = (
        model.intercept_[0]
        if hasattr(model.intercept_, "__len__")
        else model.intercept_
    )
    transformed_intercept = (
        intercept_val * scale_r + center_r
    ) - np.dot(transformed_coefs, centers)

    return transformed_coefs, transformed_intercept
