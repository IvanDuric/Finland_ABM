"""Calibration design, loss, recoverability, and identifiability diagnostics."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr


def calibration_design(
    n_samples: int,
    parameter_specs: Sequence[tuple[float, float, str]],
    seed: int = 42,
) -> np.ndarray:
    """Generate a physical Latin-hypercube design with applied integer rounding."""
    if n_samples < 2 or not parameter_specs:
        raise ValueError("Calibration requires at least two samples and one parameter.")
    rng = np.random.default_rng(seed)
    design = np.empty((n_samples, len(parameter_specs)), dtype=float)
    for column, (low, high, kind) in enumerate(parameter_specs):
        if high <= low:
            raise ValueError("Every calibration upper bound must exceed its lower bound.")
        strata = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
        values = low + strata * (high - low)
        if kind == "int":
            values = np.rint(values)
        design[:, column] = values
    return design


def waste_rate_percent(sales: np.ndarray, waste: np.ndarray) -> float:
    """Return waste as a share of physical product throughput, never currency."""
    sales_total = float(np.sum(np.asarray(sales, dtype=float)))
    waste_total = float(np.sum(np.asarray(waste, dtype=float)))
    denominator = sales_total + waste_total
    return 100.0 * waste_total / denominator if denominator > 0 else 0.0


def standardized_rmse(
    simulated: Sequence[float],
    target: Sequence[float],
    scales: Sequence[float],
) -> float:
    """RMSE of residuals standardized by declared measurement/tolerance scales."""
    simulated_arr = np.asarray(simulated, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    scale_arr = np.asarray(scales, dtype=float)
    if simulated_arr.shape != target_arr.shape or target_arr.shape != scale_arr.shape:
        raise ValueError("Simulated values, targets, and scales must have equal shape.")
    if simulated_arr.size == 0 or np.any(~np.isfinite(simulated_arr)):
        return math.inf
    if np.any(~np.isfinite(target_arr)) or np.any(scale_arr <= 0):
        raise ValueError("Targets must be finite and every calibration scale positive.")
    return float(np.sqrt(np.mean(((simulated_arr - target_arr) / scale_arr) ** 2)))


def identifiability_diagnostics(
    parameter_names: Sequence[str],
    design: np.ndarray,
    replicate_outputs: np.ndarray,
    min_points_per_parameter: int = 10,
) -> dict:
    """Diagnose whether target outputs can recover parameters from synthetic data.

    ``replicate_outputs`` has shape ``(design_points, replicates, target_features)``.
    The diagnostic combines target-space rank, parameter-output association,
    nearest-neighbour synthetic recovery, design adequacy, and stochastic noise.
    Thresholds are conservative screening rules, not mathematical proofs of
    structural identifiability.
    """
    x = np.asarray(design, dtype=float)
    y_rep = np.asarray(replicate_outputs, dtype=float)
    if x.ndim != 2 or y_rep.ndim != 3:
        raise ValueError("Design must be 2-D and replicate outputs must be 3-D.")
    n_points, n_parameters = x.shape
    if len(parameter_names) != n_parameters or y_rep.shape[0] != n_points:
        raise ValueError("Parameter names/design/output dimensions do not align.")
    if n_points < 3 or y_rep.shape[1] < 1 or y_rep.shape[2] < 1:
        raise ValueError("Insufficient design points, replicates, or target features.")

    y = np.mean(y_rep, axis=1)
    x_span = np.ptp(x, axis=0)
    x_scaled = (x - np.min(x, axis=0)) / np.where(x_span > 0, x_span, 1.0)
    y_sd = np.std(y, axis=0, ddof=1)
    informative = y_sd > 1e-12
    y_scaled = (y[:, informative] - np.mean(y[:, informative], axis=0)) / y_sd[informative]

    if y_scaled.shape[1] == 0:
        target_rank = 0
        nearest = np.zeros(n_points, dtype=int)
        nearest[:] = np.arange(n_points)
    else:
        target_rank = int(np.linalg.matrix_rank(y_scaled))
        distances = np.sqrt(np.sum((y_scaled[:, None, :] - y_scaled[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(distances, np.inf)
        nearest = np.argmin(distances, axis=1)

    recovery_error = np.abs(x_scaled - x_scaled[nearest])
    parameter_rows = []
    for column, name in enumerate(parameter_names):
        correlations = []
        for outcome_column in range(y.shape[1]):
            if np.std(y[:, outcome_column]) <= 1e-12:
                continue
            corr = spearmanr(x[:, column], y[:, outcome_column]).statistic
            if np.isfinite(corr):
                correlations.append(abs(float(corr)))
        max_corr = max(correlations, default=0.0)
        median_error = float(np.median(recovery_error[:, column]))
        p90_error = float(np.quantile(recovery_error[:, column], 0.90))
        individually_recoverable = bool(
            max_corr >= 0.15 and median_error <= 0.25 and p90_error <= 0.60
        )
        parameter_rows.append({
            "parameter": str(name),
            "max_abs_spearman": max_corr,
            "median_recovery_error": median_error,
            "p90_recovery_error": p90_error,
            "individually_recoverable": individually_recoverable,
        })

    sample_adequate = n_points >= min_points_per_parameter * n_parameters
    rank_adequate = target_rank >= n_parameters
    if y_rep.shape[1] > 1:
        within_var = np.mean(np.var(y_rep, axis=1, ddof=1), axis=0)
    else:
        within_var = np.zeros(y.shape[1], dtype=float)
    between_var = np.var(y, axis=0, ddof=1)
    variance_ratio = between_var / np.maximum(within_var, 1e-12)
    informative_ratios = variance_ratio[informative]
    median_signal_noise = (
        float(np.median(informative_ratios)) if informative_ratios.size else 0.0
    )
    noise_adequate = bool(y_rep.shape[1] >= 2 and median_signal_noise >= 1.0)
    all_recoverable = all(row["individually_recoverable"] for row in parameter_rows)
    recommendation_allowed = bool(
        sample_adequate and rank_adequate and noise_adequate and all_recoverable
    )
    reasons = []
    if not sample_adequate:
        reasons.append(
            f"design too small ({n_points} points; require at least "
            f"{min_points_per_parameter * n_parameters})"
        )
    if not rank_adequate:
        reasons.append(
            f"target-space rank {target_rank} is below {n_parameters} free parameters"
        )
    if not noise_adequate:
        reasons.append(
            "parameter signal does not reliably exceed stochastic noise or fewer than two replicates were used"
        )
    failed = [row["parameter"] for row in parameter_rows if not row["individually_recoverable"]]
    if failed:
        reasons.append("synthetic recovery failed for: " + ", ".join(failed))

    return {
        "n_points": n_points,
        "n_replicates": int(y_rep.shape[1]),
        "n_target_features": int(y.shape[1]),
        "target_rank": target_rank,
        "n_parameters": n_parameters,
        "sample_adequate": sample_adequate,
        "rank_adequate": rank_adequate,
        "noise_adequate": noise_adequate,
        "median_signal_to_noise": median_signal_noise,
        "recommendation_allowed": recommendation_allowed,
        "reasons": reasons,
        "parameters": parameter_rows,
        "method": "nearest-neighbour synthetic recovery + target rank + replicate variance",
    }
