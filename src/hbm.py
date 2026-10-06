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


def check_convergence(
    trace, var_names=None, rhat_thresh=1.05, ess_thresh=400
) -> Tuple[bool, float, float]:
    """Check MCMC convergence via R-hat and ESS."""
    summary = az.summary(trace, var_names=var_names, kind="diagnostics")
    max_rhat = summary["r_hat"].max()
    min_ess_bulk = summary["ess_bulk"].min()
    min_ess_tail = summary["ess_tail"].min()

    convergence = (max_rhat < rhat_thresh) and (min_ess_bulk > ess_thresh) and (min_ess_tail > ess_thresh)

    if not convergence:
        print(f"Warning: HBM did not converge (R-hat: {max_rhat:.3f}, ESS bulk: {min_ess_bulk}, ESS tail: {min_ess_tail})")

def fit_hierarchical_bayesian(
    train_data_fold: pd.DataFrame,
    n_patients_fold: int,
    random_state: int,
    hbm_samples: int,
    hbm_tune: int,
    hbm_chains: int,
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

    with pm.Model():
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=1)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)
        gamma0 = pm.TruncatedNormal("gamma0", mu=1, sigma=0.5, lower=0)

        beta_i = pm.Normal("beta_i", mu=mu_beta, sigma=sigma_beta, shape=n_patients_fold)

        progression_rate_i = beta_i[id_code]
        mu = gamma0 * x0_obs + progression_rate_i * time_scaled

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
        pm.Normal("x_obs", mu=mu, sigma=sigma_obs, observed=x_scaled_obs)

        trace = pm.sample(
            draws=hbm_samples,
            tune=hbm_tune,
            chains=hbm_chains,
            cores=hbm_cores,
            random_seed=random_state,
            target_accept=0.8,
            nuts_sampler="numpyro",
            chain_method="sequential",
            return_inferencedata=True,
        )

    beta_values = trace.posterior["beta_i"].mean(dim=("chain", "draw")).values
    sigma_beta_values = trace.posterior["beta_i"].std(dim=("chain", "draw")).values

    check_convergence(trace, var_names=["mu_beta", "sigma_beta", "gamma0", "sigma_obs", "beta_i"])

    return beta_values, sigma_beta_values
