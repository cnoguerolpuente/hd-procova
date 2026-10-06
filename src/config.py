import os

CONFIG = {
    "random_state": 42,
    "inner_cv_folds": 3,
    "optuna_trials": 20,
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
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "random_state": CONFIG["random_state"],
    }


def param_space_ridge(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 1e5, log=True),
        "random_state": CONFIG["random_state"],
    }


def param_space_elasticnet(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        "random_state": CONFIG["random_state"],
    }


def param_space_svr(trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 5, log=True),
        "max_iter": 30000,
        "tol": 1e-4,
        "random_state": CONFIG["random_state"],
    }


def param_space_decision_tree(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "random_state": CONFIG["random_state"],
    }


def param_space_random_forest(trial):
    return {
        "n_estimators": 300,
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": 0.5,
        "n_jobs": 1,
        "random_state": CONFIG["random_state"],
    }


def param_space_xgboost(trial):
    return {
        "tree_method": "hist",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_jobs": 1,
        "random_state": CONFIG["random_state"],
    }


def param_space_lightgbm(trial):
    return {
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 60),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "max_depth": -1,
        "verbosity": -1,
        "n_jobs": 1,
        "random_state": CONFIG["random_state"],
    }


def param_space_histgradientboosting(trial):
    return {
        "early_stopping": True,
        "n_iter_no_change": 10,
        "max_iter": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 31),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
        "random_state": CONFIG["random_state"],
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
