"""Pure, testable behavioral transformations shared by calibration and the ABM."""

from __future__ import annotations


TPB_TRANSFER_WEIGHTS = (0.49, 0.26, 0.39)


def normalized_tpb_weights(
    weights: tuple[float, float, float] = TPB_TRANSFER_WEIGHTS,
) -> tuple[float, float, float]:
    """Normalize non-negative transferred TPB weights to a convex combination."""
    if len(weights) != 3 or any(weight < 0 for weight in weights):
        raise ValueError("TPB weights must contain three non-negative values.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one TPB weight must be positive.")
    return tuple(weight / total for weight in weights)


def tpb_intention(
    attitude: float,
    subjective_norm: float,
    perceived_control: float,
    weights: tuple[float, float, float] = TPB_TRANSFER_WEIGHTS,
) -> float:
    """Return bounded intention as a normalized convex combination of TPB inputs."""
    att_w, norm_w, pbc_w = normalized_tpb_weights(weights)
    inputs = (
        min(1.0, max(0.0, float(attitude))),
        min(1.0, max(0.0, float(subjective_norm))),
        min(1.0, max(0.0, float(perceived_control))),
    )
    return att_w * inputs[0] + norm_w * inputs[1] + pbc_w * inputs[2]


def effective_hoarding_multiplier(
    maximum_multiplier: float,
    household_propensity: float,
    panic_level: float,
) -> float:
    """Continuously scale demand from 1 to the scenario maximum.

    Panic and the cross-fitted household propensity must both be positive for
    amplification. This avoids an arbitrary activation discontinuity.
    """
    maximum = max(1.0, float(maximum_multiplier))
    propensity = min(1.0, max(0.0, float(household_propensity)))
    panic = min(1.0, max(0.0, float(panic_level)))
    return 1.0 + (maximum - 1.0) * propensity * panic
