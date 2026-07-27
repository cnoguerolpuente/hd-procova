import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import norm, pearsonr


def generate_longitudinal_structure(
    n_individuals: int = 1000,
    min_timepoints: int = 1,
    max_timepoints: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_timepoints = np.random.randint(min_timepoints, max_timepoints + 1, n_individuals)
    n_datapoints = np.sum(n_timepoints)

    id_array = np.empty(n_datapoints, dtype=int)
    x0_array = np.empty(n_datapoints, dtype=int)
    t_array = np.empty(n_datapoints, dtype=float)

    current_idx = 0
    for i in range(n_individuals):
        individual_id = i + 1
        x0_i = np.random.randint(0, 21)
        n_points = n_timepoints[i]

        end_idx = current_idx + n_points
        id_array[current_idx:end_idx] = individual_id
        x0_array[current_idx:end_idx] = x0_i
        t_array[current_idx:end_idx] = np.arange(1, n_points + 1, dtype=float)

        current_idx = end_idx

    return id_array, x0_array, t_array, n_timepoints


def generate_covariates(
    n_individuals: int,
    n_irrelevant: int = 30,
    n_relevant: int = 20,
    beta_range: Tuple[float, float] = (0.02, 0.25),
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    beta_irrelevant = np.zeros(n_irrelevant)
    beta_relevant = np.linspace(beta_range[0], beta_range[1], n_relevant)
    beta = np.concatenate([beta_irrelevant, beta_relevant])

    n_covariates = len(beta)

    Z = norm.rvs(size=(n_individuals, n_covariates), random_state=42)

    covariate_columns = [f"Z{j+1}" for j in range(n_covariates)]
    covariate_df = pd.DataFrame(Z, columns=covariate_columns)
    covariate_df["id"] = np.arange(1, n_individuals + 1)

    return beta, Z, covariate_df


def calculate_progression_rates(
    id_array: np.ndarray,
    Z: np.ndarray,
    beta: np.ndarray,
    n_individuals: int,
) -> np.ndarray:
    beta0 = norm.rvs(size=n_individuals, random_state=42)

    beta0_per_observation = beta0[id_array - 1]
    Z_per_observation = Z[id_array - 1, :]

    r = beta0_per_observation + np.dot(Z_per_observation, beta)

    return r


def calculate_observed_changes(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    epsilon = norm.rvs(size=len(t), random_state=42)

    delta_y = r * t + epsilon

    return delta_y


def create_simulated_dataset(
    n_individuals: int = 1000,
    min_timepoints: int = 1,
    max_timepoints: int = 10,
    n_irrelevant: int = 30,
    n_relevant: int = 20,
    beta_range: Tuple[float, float] = (0.02, 0.25),
) -> pd.DataFrame:
    id_array, x0_array, t_array, n_timepoints = generate_longitudinal_structure(
        n_individuals, min_timepoints, max_timepoints
    )

    beta, Z, covariate_df = generate_covariates(
        n_individuals, n_irrelevant, n_relevant, beta_range
    )

    r = calculate_progression_rates(id_array, Z, beta, n_individuals)
    delta_y = calculate_observed_changes(r, t_array)
    x_array = x0_array + delta_y

    df = pd.DataFrame(
        {
            "id": id_array,
            "x0": x0_array,
            "x": x_array,
            "time": t_array,
            "NVISIT": t_array.astype(int),
        }
    )

    # Add AVISIT labels (BASELINE row + visit labels)
    df["AVISIT"] = "VISIT_" + df["NVISIT"].astype(str)

    # Add baseline rows (one per patient, NVISIT=0, x=x0)
    baseline = df.groupby("id").first().reset_index()
    baseline["x"] = baseline["x0"]
    baseline["time"] = 0.0
    baseline["NVISIT"] = 0
    baseline["AVISIT"] = "BASELINE"

    df = pd.concat([baseline, df], ignore_index=True).sort_values(
        ["id", "NVISIT"]
    ).reset_index(drop=True)

    df = df.merge(covariate_df, on="id", how="left")

    return df


def analyze_covariate_effects(
    df: pd.DataFrame,
    target_col: str = "x",
    top_n: int = 10,
) -> pd.DataFrame:
    z_cols = [col for col in df.columns if col.startswith("Z")]
    correlations = []

    for col in z_cols:
        corr, p_val = pearsonr(df[col], df[target_col])
        covar_num = int(col[1:])
        is_relevant = covar_num > 30

        correlations.append(
            {
                "covariate": col,
                "correlation": corr,
                "abs_correlation": abs(corr),
                "p_value": p_val,
                "is_relevant": is_relevant,
            }
        )

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("abs_correlation", ascending=False)

    return corr_df.head(top_n) if top_n else corr_df
