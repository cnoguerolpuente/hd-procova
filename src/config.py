import os

CONFIG = {
    "random_state": 42,
    "inner_cv_folds": 5,
    "optuna_trials": 30,
    "scoring_metric": "neg_mean_squared_error",
    "output_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs"),
}

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

LINEAR_MODELS: list = ["lasso", "ridge", "elasticnet", "svr"]

MODEL_MAP = {
    1: {"class": Lasso, "name": "lasso"},
    2: {"class": Ridge, "name": "ridge"},
    3: {"class": ElasticNet, "name": "elasticnet"},
    4: {"class": LinearSVR, "name": "svr"},
    5: {"class": DecisionTreeRegressor, "name": "decision_tree"},
    6: {"class": RandomForestRegressor, "name": "random_forest"},
    7: {"class": XGBRegressor, "name": "xgboost"},
    8: {"class": LGBMRegressor, "name": "lightgbm"},
    9: {"class": HistGradientBoostingRegressor, "name": "histgradientboosting"},
}


def param_space_lasso(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True)}


def param_space_ridge(trial):
    return {"alpha": trial.suggest_int("alpha", 0.1, 10)}


def param_space_elasticnet(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1.0, step=0.1),
    }


def param_space_svr(trial):
    return {
        "C": trial.suggest_float("C", 0.001, 10, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 0.5, step=0.01),
        "random_state": CONFIG["random_state"],
        "max_iter": 30000,
        "tol": 1e-4,
    }


def param_space_decision_tree(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_samples_split": trial.suggest_categorical(
            "min_samples_split", [2, 5, 10]
        ),
        "min_samples_leaf": trial.suggest_categorical(
            "min_samples_leaf", [5, 10, 20, 50, 100]
        ),
    }


def param_space_random_forest(trial):
    return {
        "random_state": CONFIG["random_state"],
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def param_space_xgboost(trial):
    return {
        "tree_method": "hist",
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
    }


def param_space_lightgbm(trial):
    return {
        "random_state": CONFIG["random_state"],
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "verbosity": -1,
    }


def param_space_histgradientboosting(trial):
    return {
        "random_state": CONFIG["random_state"],
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
    }


PARAM_SPACE_MAP = {
    1: param_space_lasso,
    2: param_space_ridge,
    3: param_space_elasticnet,
    4: param_space_svr,
    5: param_space_decision_tree,
    6: param_space_random_forest,
    7: param_space_xgboost,
    8: param_space_lightgbm,
    9: param_space_histgradientboosting,
}
