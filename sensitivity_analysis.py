"""Statistical helpers for replicated global sensitivity analysis.

The functions in this module are deliberately independent of Streamlit and the
ABM so their statistical behaviour can be unit tested in isolation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr, t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def latin_hypercube(n_samples: int, n_parameters: int, seed: int) -> np.ndarray:
    """Return a reproducible maximin-free Latin hypercube in the unit cube."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if n_parameters < 1:
        raise ValueError("n_parameters must be positive")
    rng = np.random.default_rng(seed)
    design = np.empty((n_samples, n_parameters), dtype=float)
    for column in range(n_parameters):
        strata = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        design[:, column] = strata[rng.permutation(n_samples)]
    return design


def scale_design(unit_design: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """Scale a unit-cube design to physical parameter bounds."""
    design = np.asarray(unit_design, dtype=float)
    if design.ndim != 2 or design.shape[1] != len(bounds):
        raise ValueError("design columns must match bounds")
    lows = np.array([bound[0] for bound in bounds], dtype=float)
    highs = np.array([bound[1] for bound in bounds], dtype=float)
    if np.any(highs <= lows):
        raise ValueError("every upper bound must exceed its lower bound")
    return lows + design * (highs - lows)


def partial_rank_correlations(x: np.ndarray, y: np.ndarray) -> list[dict]:
    """Calculate PRCCs after residualising ranked inputs and ranked output.

    The reported p-value uses the conventional partial-correlation t test with
    ``n - k - 1`` degrees of freedom. It is descriptive for screening and is
    not corrected for multiple comparisons.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(y) != len(x):
        raise ValueError("x must be 2D and y must be a matching 1D vector")
    n, k = x.shape
    if n <= k + 2:
        raise ValueError("PRCC requires more observations than parameters + 2")

    ranked_x = np.column_stack([rankdata(x[:, j]) for j in range(k)])
    ranked_y = rankdata(y)
    results: list[dict] = []
    for j in range(k):
        others = np.delete(ranked_x, j, axis=1)
        design = np.column_stack([np.ones(n), others])
        residual_x = ranked_x[:, j] - design @ np.linalg.lstsq(
            design, ranked_x[:, j], rcond=None
        )[0]
        residual_y = ranked_y - design @ np.linalg.lstsq(
            design, ranked_y, rcond=None
        )[0]
        denom = float(np.linalg.norm(residual_x) * np.linalg.norm(residual_y))
        coefficient = (
            float(np.dot(residual_x, residual_y) / denom) if denom > 1e-12 else 0.0
        )
        coefficient = float(np.clip(coefficient, -1.0, 1.0))
        degrees_freedom = n - k - 1
        if abs(coefficient) >= 1.0:
            p_value = 0.0
        else:
            statistic = coefficient * np.sqrt(
                degrees_freedom / max(1e-15, 1.0 - coefficient**2)
            )
            p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
        results.append({"coefficient": coefficient, "p_value": p_value})
    return results


def bootstrap_prcc(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 300,
    seed: int = 0,
) -> list[dict]:
    """Add non-parametric 95% bootstrap intervals to PRCC estimates."""
    point = partial_rank_correlations(x, y)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, x.shape[1]), dtype=float)
    accepted = 0
    attempts = 0
    while accepted < n_bootstrap and attempts < n_bootstrap * 5:
        attempts += 1
        indices = rng.integers(0, len(y), len(y))
        try:
            estimate = partial_rank_correlations(x[indices], y[indices])
        except (ValueError, np.linalg.LinAlgError):
            continue
        draws[accepted] = [row["coefficient"] for row in estimate]
        accepted += 1
    if accepted == 0:
        raise ValueError("bootstrap PRCC failed for every resample")
    draws = draws[:accepted]
    for j, row in enumerate(point):
        row["ci_low"], row["ci_high"] = np.quantile(
            draws[:, j], [0.025, 0.975]
        ).tolist()
        row["bootstrap_samples"] = accepted
    return point


def nonlinear_permutation_importance(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
) -> dict:
    """Held-out random-forest permutation importance for nonlinear screening."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 20:
        return {
            "heldout_r2": float("nan"),
            "importance": np.zeros(x.shape[1]),
            "importance_sd": np.zeros(x.shape[1]),
            "status": "insufficient_sample",
        }
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed
    )
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=max(2, len(y_train) // 25),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    heldout_r2 = float(model.score(x_test, y_test))
    if not np.isfinite(heldout_r2) or heldout_r2 <= 0:
        return {
            "heldout_r2": heldout_r2,
            "importance": np.zeros(x.shape[1]),
            "importance_sd": np.zeros(x.shape[1]),
            "status": "rejected_no_predictive_skill",
        }
    permutation = permutation_importance(
        model, x_test, y_test, n_repeats=30, random_state=seed, scoring="r2"
    )
    positive = np.clip(permutation.importances_mean, 0.0, None)
    total = float(positive.sum())
    normalized = positive / total if total > 0 else positive
    return {
        "heldout_r2": heldout_r2,
        "importance": normalized,
        "importance_sd": permutation.importances_std,
        "status": "retained",
    }


def convergence_diagnostics(x: np.ndarray, y: np.ndarray) -> dict:
    """Compare PRCC estimates and importance rankings at nested sample sizes."""
    n, k = x.shape
    minimum = max(k + 3, int(np.ceil(n * 0.5)))
    sizes = sorted(set([minimum, max(minimum, int(np.ceil(n * 0.75))), n]))
    estimates = []
    for size in sizes:
        coefficients = np.array([
            row["coefficient"] for row in partial_rank_correlations(x[:size], y[:size])
        ])
        estimates.append(coefficients)
    final = estimates[-1]
    rows = []
    for size, coefficients in zip(sizes, estimates):
        rank_stability = spearmanr(np.abs(coefficients), np.abs(final)).statistic
        rows.append({
            "n": size,
            "max_abs_change": float(np.max(np.abs(coefficients - final))),
            "rank_stability": float(rank_stability) if np.isfinite(rank_stability) else 0.0,
        })
    return {"rows": rows, "stable": rows[0]["rank_stability"] >= 0.8}


def variance_decomposition(replicate_outputs: np.ndarray) -> dict:
    """Separate variation between parameter points from stochastic run noise."""
    values = np.asarray(replicate_outputs, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("replicate_outputs must have at least two replicates per design")
    design_means = values.mean(axis=1)
    observed_between = (
        float(np.var(design_means, ddof=1)) if len(design_means) > 1 else 0.0
    )
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    # Design means still contain within-point Monte Carlo error. Subtract its
    # expected contribution before attributing variation to parameter ranges.
    between = max(0.0, observed_between - within / values.shape[1])
    total = between + within
    return {
        "observed_variance_of_design_means": observed_between,
        "between_parameter_variance": between,
        "within_stochastic_variance": within,
        "parameter_variance_share": between / total if total > 0 else 0.0,
    }
