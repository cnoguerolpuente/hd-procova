import numpy as np
import pandas as pd
import os

def _cuda_available() -> bool:
    import importlib.util
    import subprocess
    cuda_plugins = ("jax_cuda12_plugin", "jax_cuda11_plugin", "jaxlib_cuda")
    if not any(importlib.util.find_spec(p) is not None for p in cuda_plugins):
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


def _normalize_jax_platforms_env() -> None:
    raw = os.environ.get("JAX_PLATFORMS", "")
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    aliases = {"gpu": "cuda", "nvidia": "cuda", "amd": "rocm"}
    normalized = [aliases.get(p, p) for p in parts]
    normalized = ["cuda" if p == "rocm" else p for p in normalized]

    if "cuda" in normalized and not _cuda_available():
        normalized = [p for p in normalized if p != "cuda"]

    seen = set()
    deduped = []
    for p in normalized:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    if "cpu" not in seen:
        deduped.append("cpu")

    os.environ["JAX_PLATFORMS"] = ",".join(deduped)


_normalize_jax_platforms_env()
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ.pop("JAX_PLATFORM_NAME", None)
os.environ.setdefault("PYTENSOR_FLAGS", "mode=FAST_COMPILE,linker=py,cxx=")
os.environ["PYTENSOR_CXX"] = ""

import numpyro
numpyro.set_host_device_count(4)

import pymc as pm
import pytensor
pytensor.config.cxx = ""

from typing import Tuple
import arviz as az
from matplotlib import pyplot as plt


def _effective_hbm_chains(requested_chains: int, logger=None) -> int:
    requested = max(1, int(requested_chains))
    if logger is not None:
        logger(f"Using {requested} vectorized chain(s).")
    return requested


def fit_hierarchical_bayesian(
    train_data_fold: pd.DataFrame,
    n_patients_fold: int,
    random_state: int,
    hbm_samples: int = 500,
    hbm_tune: int = 500,
    hbm_chains: int = 2,
    hbm_cores: int = 1,
) -> Tuple[np.ndarray, float, float, az.InferenceData]:
    """Fit the PyMC HBM and extract per-patient progression rates."""
    id_code = train_data_fold["id_code"].to_numpy(dtype=np.int32)
    x0_by_patient = (
        train_data_fold.groupby("id_code")["x0_scaled"]
        .first()
        .sort_index()
        .to_numpy(dtype=np.float64)
    )
    x0_obs = x0_by_patient[id_code]
    time_scaled = train_data_fold["time_scaled"].to_numpy(dtype=np.float64)
    x_scaled_obs = train_data_fold["x_scaled"].to_numpy(dtype=np.float64)
    effective_chains = _effective_hbm_chains(hbm_chains)

    with pm.Model():
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=1)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)
        gamma0 = pm.TruncatedNormal("gamma0", mu=1, sigma=0.5, lower=0)

        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=n_patients_fold)
        beta = pm.Deterministic("beta", mu_beta + beta_raw * sigma_beta)

        progression_rate_i = beta[id_code]
        mu = gamma0 * x0_obs + progression_rate_i * time_scaled

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
        pm.Normal("x_obs", mu=mu, sigma=sigma_obs, observed=x_scaled_obs)

        trace = pm.sample(
            draws=hbm_samples,
            tune=hbm_tune,
            chains=effective_chains,
            cores=1,
            random_seed=random_state,
            target_accept=0.8,
            nuts_sampler="numpyro",
            chain_method="sequential",
            return_inferencedata=True,
        )

    beta_values = trace.posterior["beta"].mean(dim=("chain", "draw")).values
    gamma0_val = trace.posterior["gamma0"].mean(dim=("chain", "draw")).values
    sigma_beta_values = trace.posterior["beta"].std(dim=("chain", "draw")).values

    return beta_values, sigma_beta_values, gamma0_val, trace


def check_convergence(
    trace, var_names=None, rhat_thresh=1.05, ess_thresh=400
) -> Tuple[bool, float, float]:
    """Check MCMC convergence via R-hat and ESS."""
    summary = az.summary(trace, var_names=var_names, kind="diagnostics")
    max_rhat = summary["r_hat"].max()
    min_ess = summary["ess_bulk"].min()

    n_chains = int(trace.posterior.sizes.get("chain", 1))
    if n_chains < 2 or np.isnan(max_rhat):
        convergence = min_ess > ess_thresh
    else:
        convergence = (max_rhat < rhat_thresh) and (min_ess > ess_thresh)
    return convergence, max_rhat, min_ess


def save_hbm_convergence_plot(
    trace, output_dir: str, fold_idx: int = None
):
    """Save MCMC diagnostic trace plots."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if fold_idx is not None:
        title_prefix = f"Fold {fold_idx}: "
        file_prefix = f"fold_{fold_idx}_"
    else:
        title_prefix = ""
        file_prefix = ""

    hyper_params = ["sigma_obs", "gamma0", "mu_beta", "sigma_beta"]
    num_hyper = len(hyper_params)

    total_rows = num_hyper + 1
    fig, axes = plt.subplots(total_rows, 2, figsize=(12, 2 * total_rows))

    az.plot_trace(
        trace, var_names=hyper_params, axes=axes[:num_hyper, :], compact=True
    )

    beta_indices = np.arange(5)
    coords = {"beta_dim_0": beta_indices}
    az.plot_trace(
        trace, var_names=["beta"], coords=coords,
        axes=axes[num_hyper:, :], compact=True,
    )

    fig.suptitle(
        f"{title_prefix}HBM Trace and Posterior Plots", fontsize=16, y=1.001
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, f"{file_prefix}trace.png"),
        dpi=150, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    print(f"Saved diagnostic plot to {plots_dir}")
