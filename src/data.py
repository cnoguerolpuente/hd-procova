import logging
import sys
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG

RANDOM_STATE = CONFIG["random_state"]


def setup_logging(fold_idx: int, model_name: str) -> logging.Logger:
    logger = logging.getLogger(f"fold_{fold_idx}_{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | FOLD-%(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_data(path: str) -> pd.DataFrame:
    """Load a simulated dataset CSV and prepare it for the pipeline.

    Expects columns: id, x0, x, time, NVISIT, AVISIT, plus covariate columns.
    """
    df = pd.read_csv(path)

    if "ID" in df.columns and "id" not in df.columns:
        df.rename(columns={"ID": "id"}, inplace=True)

    if "time" not in df.columns and "NVISIT" in df.columns:
        df["time"] = df["NVISIT"].astype(float)

    print(f"Loaded {len(df)} rows, {df['id'].nunique()} patients from {path}")
    return df


def scale_pipeline_data(train_data: pd.DataFrame) -> tuple[dict, list, list]:
    """Fit scalers on train_data and add scaled columns in-place.

    Returns:
        scalers: Dict of StandardScalers.
        covariate_cols: Original covariate column names.
        scaled_covariate_cols: Corresponding *_scaled column names.
    """
    scalers: dict = {
        "x": StandardScaler(),
        "time": StandardScaler(with_mean=False),
        "covariates": StandardScaler(),
        "progression_rates": StandardScaler(),
    }

    non_covariate_cols = {"id", "time", "x0", "x", "STATUS", "AVISIT", "NVISIT"}
    covariate_cols: list = [
        col for col in train_data.columns if col not in non_covariate_cols
    ]
    scaled_covariate_cols: list = [col + "_scaled" for col in covariate_cols]

    train_data[scaled_covariate_cols] = scalers["covariates"].fit_transform(
        train_data[covariate_cols]
    )

    scalers["x"].fit(train_data[["x"]])
    train_data["x0_scaled"] = (
        train_data[["x0"]].values - scalers["x"].mean_[0]
    ) / scalers["x"].scale_[0]
    train_data["x_scaled"] = scalers["x"].transform(train_data[["x"]])

    train_data["time_scaled"] = scalers["time"].fit_transform(train_data[["time"]])

    return scalers, covariate_cols, scaled_covariate_cols


def remove_baseline_visits(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    drop_test_avisit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop BASELINE rows and the AVISIT column from both DataFrames."""
    if "AVISIT" not in train_data.columns:
        return train_data, test_data

    train_data.drop(
        train_data[train_data["AVISIT"] == "BASELINE"].index, inplace=True
    )
    test_data.drop(
        test_data[test_data["AVISIT"] == "BASELINE"].index, inplace=True
    )
    train_data.drop(columns=["AVISIT"], inplace=True)
    if drop_test_avisit:
        test_data.drop(columns=["AVISIT"], inplace=True)

    return train_data, test_data
