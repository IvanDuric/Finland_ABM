"""Evidence-aware validation utilities for GROCERYsim.

The module deliberately separates software/model checks from empirical validation.
Only preregistered targets based on data that were not used for calibration can
support an external-validation claim, and even then the claim is limited to the
declared targets.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


EVIDENCE_TIERS = {
    "internal_invariant",
    "calibration_holdout",
    "external_independent",
    "scenario_plausibility",
}

AGGREGATIONS = {"mean", "median", "sum", "min", "max", "last"}

REQUIRED_COLUMNS = (
    "target_id",
    "metric",
    "label",
    "evidence_tier",
    "scenario",
    "aggregation",
    "day_start",
    "day_end",
    "multiplier",
    "lower",
    "upper",
    "unit",
    "source_name",
    "source_reference",
    "source_period",
    "source_population",
    "independent_of_calibration",
    "preregistered",
    "registration_reference",
    "notes",
)


def daily_validation_observables(
    sales_units: float,
    nominal_revenue: float,
    consumers: int,
    waste_units: float,
    sku_stockout_flags: Iterable[bool],
) -> dict[str, float]:
    """Create explicitly denominated daily model counterparts for validation."""
    flags = [bool(value) for value in sku_stockout_flags]
    consumers = int(consumers)
    throughput = float(sales_units) + float(waste_units)
    return {
        "MeanDairyBasketUnits": float(sales_units) / consumers if consumers > 0 else 0.0,
        "MeanDairyBasketValue": float(nominal_revenue) / consumers if consumers > 0 else 0.0,
        "DairyWasteRate": float(waste_units) / throughput if throughput > 0 else 0.0,
        "StockoutSkuDayRate": sum(flags) / len(flags) if flags else 0.0,
    }


def evaluate_baseline_reproduction(
    targets: Mapping[str, object], simulation: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate whether repeated baseline operation preserves phase-one patterns.

    Targets come from the same GROCERYsim phase-one observations that initialise
    the agents. Consequently this is an internal reproduction test, not external
    validation. The warm-up excludes one model-implied revisit interval.
    """
    if not isinstance(simulation, pd.DataFrame) or simulation.empty:
        raise ValueError("simulation must be a non-empty DataFrame")
    required = {
        "Day", "Consumers", "RequestedDemandUnits", "Sales", "NominalRevenue",
        "DomesticSales", "ImportSales", "OrganicSalesUnits",
        "CategorySalesUnits", "ConsumptionFulfillmentRate",
        "VisitorCapacityCapped", "ExpectedVisitIntervalDays",
    }
    missing = sorted(required - set(simulation.columns))
    if missing:
        raise ValueError("baseline reproduction columns are missing: " + ", ".join(missing))
    if targets.get("status") != "ok":
        raise ValueError("baseline observation targets are unavailable")

    interval = float(pd.to_numeric(
        simulation["ExpectedVisitIntervalDays"], errors="coerce"
    ).dropna().median())
    warmup_days = max(1, int(np.ceil(interval)))
    evaluation = simulation[
        pd.to_numeric(simulation["Day"], errors="coerce") > warmup_days
    ].copy()
    if evaluation.empty:
        raise ValueError(
            f"simulation must extend beyond the {warmup_days}-day warm-up"
        )

    def total(column: str) -> float:
        return float(pd.to_numeric(evaluation[column], errors="coerce").fillna(0).sum())

    visits = total("Consumers")
    if visits <= 0:
        raise ValueError("no household visits remain after warm-up")

    checks: list[dict[str, object]] = []

    def interval_check(
        metric: str, label: str, model_value: float, target_value: float,
        tolerance: float, unit: str, kind: str = "relative",
    ) -> None:
        if kind == "relative":
            lower = target_value * (1.0 - tolerance)
            upper = target_value * (1.0 + tolerance)
        else:
            lower = target_value - tolerance
            upper = target_value + tolerance
            if unit == "share":
                lower = max(0.0, lower)
                upper = min(1.0, upper)
        checks.append({
            "metric": metric,
            "label": label,
            "target": round(float(target_value), 4),
            "model": round(float(model_value), 4),
            "lower": round(float(lower), 4),
            "upper": round(float(upper), 4),
            "unit": unit,
            "status": "pass" if lower <= model_value <= upper else "fail",
            "evidence_tier": "internal_reproduction",
        })

    observed_units = float(targets["mean_linked_basket_units"])
    observed_value = float(targets["mean_linked_basket_value"])
    interval_check(
        "requested_units_per_visit", "Generated routine units per visit",
        total("RequestedDemandUnits") / visits, observed_units, 0.05, "units/visit",
    )
    interval_check(
        "purchased_units_per_visit", "Purchased units per visit",
        total("Sales") / visits, observed_units, 0.10, "units/visit",
    )
    interval_check(
        "purchased_value_per_visit", "Purchased value per visit",
        total("NominalRevenue") / visits, observed_value, 0.10, "EUR/visit",
    )

    sales = total("Sales")
    domestic_total = total("DomesticSales") + total("ImportSales")
    interval_check(
        "organic_unit_share", "Organic unit share",
        total("OrganicSalesUnits") / sales if sales > 0 else 0.0,
        float(targets["organic_unit_share"]), 0.05, "share", "absolute",
    )
    interval_check(
        "domestic_unit_share", "Domestic unit share",
        total("DomesticSales") / domestic_total if domestic_total > 0 else 0.0,
        float(targets["domestic_unit_share"]), 0.05, "share", "absolute",
    )

    category_totals: dict[str, float] = {}
    for value in evaluation["CategorySalesUnits"]:
        if not isinstance(value, Mapping):
            continue
        for category, units in value.items():
            category_totals[str(category)] = (
                category_totals.get(str(category), 0.0) + float(units)
            )
    category_sales = sum(category_totals.values())
    for category, target_share in targets.get("category_unit_shares", {}).items():
        interval_check(
            f"category_share:{category}", f"{category} unit share",
            category_totals.get(str(category), 0.0) / category_sales
            if category_sales > 0 else 0.0,
            float(target_share), 0.05, "share", "absolute",
        )

    fulfillment = sales / max(total("RequestedDemandUnits"), 1e-12)
    checks.append({
        "metric": "baseline_fulfillment", "label": "Baseline unit fulfilment",
        "target": 0.95, "model": round(fulfillment, 4), "lower": 0.95,
        "upper": 1.0, "unit": "share",
        "status": "pass" if 0.95 <= fulfillment <= 1.0 + 1e-9 else "fail",
        "evidence_tier": "internal_invariant",
    })
    consumption = float(pd.to_numeric(
        evaluation["ConsumptionFulfillmentRate"], errors="coerce"
    ).mean())
    checks.append({
        "metric": "consumption_fulfillment", "label": "Pantry consumption fulfilment",
        "target": 0.95, "model": round(consumption, 4), "lower": 0.95,
        "upper": 1.0, "unit": "share",
        "status": "pass" if 0.95 <= consumption <= 1.0 + 1e-9 else "fail",
        "evidence_tier": "internal_invariant",
    })
    capacity_capped_days = int(pd.to_numeric(
        evaluation["VisitorCapacityCapped"], errors="coerce"
    ).fillna(0).gt(0).sum())
    checks.append({
        "metric": "visitor_capacity_capped_days",
        "label": "Days footfall exceeded represented households",
        "target": 0, "model": capacity_capped_days, "lower": 0, "upper": 0,
        "unit": "days", "status": "pass" if capacity_capped_days == 0 else "fail",
        "evidence_tier": "internal_invariant",
    })

    passed = sum(check["status"] == "pass" for check in checks)
    return {
        "status": "pass" if passed == len(checks) else "fail",
        "passed": passed,
        "total": len(checks),
        "warmup_days": warmup_days,
        "evaluation_days": len(evaluation),
        "expected_visit_interval_days": round(interval, 4),
        "checks": checks,
        "claim": (
            "Internal phase-one pattern reproduction only; revisit timing, daily "
            "consumption, and store capacity are not identified by GROCERYsim."
        ),
    }


def evaluate_phase2_reproduction(
    targets: Mapping[str, object], simulation: pd.DataFrame,
) -> dict[str, object]:
    """Compare controlled one-occasion predictions with phase-two holdouts."""
    if targets.get("status") != "ok":
        raise ValueError("phase-two holdout targets are unavailable")
    if not isinstance(simulation, pd.DataFrame) or simulation.empty:
        raise ValueError("simulation must be a non-empty DataFrame")
    if "source_id" not in simulation or "Run" not in simulation:
        raise ValueError("simulation needs source_id and Run columns")

    checks = []
    for metric, target in targets.get("metrics", {}).items():
        model_column = f"model_{metric}"
        observed_column = f"observed_{metric}"
        if model_column not in simulation or observed_column not in simulation:
            raise ValueError(f"phase-two metric columns missing for {metric}")
        participant = simulation.groupby("source_id", sort=True).agg(
            model=(model_column, "mean"),
            observed=(observed_column, "first"),
        )
        model_mean = float(participant["model"].mean())
        observed_mean = float(target["mean"])
        tolerance = float(target.get("absolute_tolerance", 0.10))
        lower = max(0.0, observed_mean - tolerance)
        upper = min(1.5, observed_mean + tolerance)
        mae = float(np.mean(np.abs(
            participant["model"] - participant["observed"]
        )))
        training_mean = float(target.get("training_mean", observed_mean))
        naive_mae = float(np.mean(np.abs(
            participant["observed"] - training_mean
        )))
        mean_pass = lower <= model_mean <= upper
        skill_pass = bool(mae < naive_mae - 1e-12) if naive_mae > 1e-12 else False
        individual_claimed = bool(target.get("individual_model_retained", False))
        overall_pass = mean_pass and (skill_pass if individual_claimed else True)
        checks.append({
            "metric": metric,
            "holdout_mean": round(observed_mean, 4),
            "model_mean": round(model_mean, 4),
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "participant_mae": round(mae, 4),
            "naive_training_mean_mae": round(naive_mae, 4),
            "mean_gate": "pass" if mean_pass else "fail",
            "individual_skill_gate": (
                "pass" if skill_pass else "fail"
            ) if individual_claimed else "not_claimed",
            "status": "pass" if overall_pass else "fail",
            "evidence_tier": "calibration_holdout",
        })

    passed = sum(row["status"] == "pass" for row in checks)
    return {
        "status": "pass" if passed == len(checks) else "fail",
        "passed": passed,
        "total": len(checks),
        "n_holdout": int(simulation["source_id"].nunique()),
        "n_replicates": int(simulation["Run"].nunique()),
        "checks": checks,
        "claim": targets.get("caution", "Calibration-holdout evidence only."),
    }


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def _as_bool(value: object) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def validation_target_template() -> pd.DataFrame:
    """Return a schema/example template; examples are not validation evidence."""
    rows = [
        {
            "target_id": "EXAMPLE_baseline_waste",
            "metric": "Waste",
            "label": "Example only — mean daily waste",
            "evidence_tier": "external_independent",
            "scenario": "Baseline",
            "aggregation": "mean",
            "day_start": 15,
            "day_end": 60,
            "multiplier": 1.0,
            "lower": 0.0,
            "upper": 1.0,
            "unit": "units/day",
            "source_name": "REPLACE with dataset/study name",
            "source_reference": "REPLACE with DOI, URL, or archive identifier",
            "source_period": "REPLACE with observation dates",
            "source_population": "REPLACE with geography/sample/retail scope",
            "independent_of_calibration": True,
            "preregistered": True,
            "registration_reference": "REPLACE with timestamped protocol/OSF reference",
            "notes": "Replace every example value before use.",
        },
        {
            "target_id": "EXAMPLE_crisis_sales",
            "metric": "Sales",
            "label": "Example only — crisis-period cumulative sales",
            "evidence_tier": "external_independent",
            "scenario": "Crisis",
            "aggregation": "sum",
            "day_start": 1,
            "day_end": 60,
            "multiplier": 1.0,
            "lower": 0.0,
            "upper": 1.0,
            "unit": "units",
            "source_name": "REPLACE with dataset/study name",
            "source_reference": "REPLACE with DOI, URL, or archive identifier",
            "source_period": "REPLACE with observation dates",
            "source_population": "REPLACE with geography/sample/retail scope",
            "independent_of_calibration": True,
            "preregistered": True,
            "registration_reference": "REPLACE with timestamped protocol/OSF reference",
            "notes": "Replace every example value before use.",
        },
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def validate_target_definitions(targets: pd.DataFrame) -> list[str]:
    """Return human-readable schema and evidence-integrity errors."""
    errors: list[str] = []
    if not isinstance(targets, pd.DataFrame):
        return ["Validation targets must be supplied as a table."]
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in targets.columns]
    if missing_columns:
        return ["Missing required columns: " + ", ".join(missing_columns)]
    if targets.empty:
        return ["The validation plan contains no targets."]

    ids = targets["target_id"].astype(str).str.strip()
    if ids.eq("").any():
        errors.append("Every row needs a non-empty target_id.")
    duplicates = sorted(ids[ids.duplicated(keep=False)].unique())
    if duplicates:
        errors.append("target_id values must be unique: " + ", ".join(duplicates))

    for idx, row in targets.iterrows():
        target_id = str(row.get("target_id", f"row {idx + 1}")).strip() or f"row {idx + 1}"
        tier = str(row["evidence_tier"]).strip()
        aggregation = str(row["aggregation"]).strip().lower()
        if tier not in EVIDENCE_TIERS:
            errors.append(f"{target_id}: unsupported evidence_tier '{tier}'.")
        if aggregation not in AGGREGATIONS:
            errors.append(f"{target_id}: unsupported aggregation '{aggregation}'.")
        if _missing(row["metric"]):
            errors.append(f"{target_id}: metric is required.")
        try:
            lower, upper = float(row["lower"]), float(row["upper"])
            multiplier = float(row["multiplier"])
            if not np.isfinite([lower, upper, multiplier]).all():
                raise ValueError
            if lower > upper:
                errors.append(f"{target_id}: lower must be no greater than upper.")
        except (TypeError, ValueError):
            errors.append(f"{target_id}: lower, upper, and multiplier must be finite numbers.")

        for day_column in ("day_start", "day_end"):
            if not _missing(row[day_column]):
                try:
                    if not np.isfinite(float(row[day_column])):
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(f"{target_id}: {day_column} must be numeric or blank.")

        independent = _as_bool(row["independent_of_calibration"])
        preregistered = _as_bool(row["preregistered"])
        if independent is None:
            errors.append(f"{target_id}: independent_of_calibration must be true or false.")
        if preregistered is None:
            errors.append(f"{target_id}: preregistered must be true or false.")

        if tier == "external_independent":
            for column in ("source_name", "source_reference", "source_period", "source_population"):
                if _missing(row[column]):
                    errors.append(f"{target_id}: {column} is required for external evidence.")
            if independent is not True:
                errors.append(f"{target_id}: external evidence must be independent of calibration.")
            if preregistered is not True:
                errors.append(f"{target_id}: external targets must be preregistered.")
            if _missing(row["registration_reference"]):
                errors.append(f"{target_id}: a timestamped registration_reference is required.")

        placeholder_values = [str(row.get(column, "")).strip().upper()
                              for column in ("target_id", "source_name", "source_reference", "notes")]
        if tier == "external_independent" and any(
            value.startswith("EXAMPLE") or value.startswith("REPLACE")
            for value in placeholder_values
        ):
            errors.append(f"{target_id}: template/example placeholders cannot be used as evidence.")

    return errors


def _aggregate(series: pd.Series, method: str) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("no numeric observations")
    if method == "last":
        return float(values.iloc[-1])
    return float(getattr(values, method)())


def evaluate_targets(targets: pd.DataFrame, simulation: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a valid target plan against a simulation-output table."""
    errors = validate_target_definitions(targets)
    if errors:
        raise ValueError("Invalid validation plan: " + " | ".join(errors))
    if not isinstance(simulation, pd.DataFrame):
        raise TypeError("simulation must be a pandas DataFrame")

    output = []
    for _, row in targets.iterrows():
        record = row.to_dict()
        metric = str(row["metric"]).strip()
        subset = simulation.copy()
        reason = ""
        observed = np.nan
        simulation_lower = np.nan
        simulation_upper = np.nan
        n_replicates = 0
        status = "not_evaluated"

        if metric not in subset.columns:
            reason = f"metric column '{metric}' is absent"
        else:
            scenario = "" if _missing(row["scenario"]) else str(row["scenario"]).strip()
            if scenario:
                if "Scenario" not in subset.columns:
                    reason = "Scenario column is absent"
                else:
                    subset = subset[subset["Scenario"].astype(str) == scenario]
            if not reason and not _missing(row["day_start"]):
                if "Day" not in subset.columns:
                    reason = "Day column is absent"
                else:
                    subset = subset[pd.to_numeric(subset["Day"], errors="coerce") >= float(row["day_start"])]
            if not reason and not _missing(row["day_end"]):
                if "Day" not in subset.columns:
                    reason = "Day column is absent"
                else:
                    subset = subset[pd.to_numeric(subset["Day"], errors="coerce") <= float(row["day_end"])]
            if not reason and subset.empty:
                reason = "no simulation rows match the declared scenario/day window"
            if not reason:
                try:
                    method = str(row["aggregation"]).strip().lower()
                    multiplier = float(row["multiplier"])
                    if "Run" in subset.columns:
                        replicate_values = np.asarray([
                            _aggregate(group[metric], method) * multiplier
                            for _, group in subset.groupby("Run", sort=True)
                        ], dtype=float)
                    else:
                        replicate_values = np.asarray([
                            _aggregate(subset[metric], method) * multiplier
                        ], dtype=float)
                    n_replicates = int(len(replicate_values))
                    observed = float(np.mean(replicate_values))
                    simulation_lower, simulation_upper = (
                        np.quantile(replicate_values, [0.025, 0.975])
                        if n_replicates >= 2 else (observed, observed)
                    )
                    inside = float(row["lower"]) <= observed <= float(row["upper"])
                    interval_inside = (
                        float(row["lower"]) <= simulation_lower
                        and simulation_upper <= float(row["upper"])
                    )
                    if str(row["evidence_tier"]).strip() == "external_independent" and n_replicates < 3:
                        reason = "external evaluation requires at least three stochastic replicates with a Run column"
                    else:
                        status = "pass" if inside and interval_inside else "fail"
                        if inside and not interval_inside:
                            reason = "model mean is in range but its 95% replicate interval crosses an acceptance bound"
                except ValueError as exc:
                    reason = str(exc)

        record.update({
            "observed": observed,
            "simulation_lower_95": float(simulation_lower),
            "simulation_upper_95": float(simulation_upper),
            "n_replicates": n_replicates,
            "status": status,
            "reason": reason,
        })
        output.append(record)
    return pd.DataFrame(output)


def validation_summary(evaluated: pd.DataFrame) -> dict:
    """Summarise evidence without allowing lower tiers to imply external validity."""
    if evaluated is None or evaluated.empty:
        return {
            "claim_status": "no_external_targets",
            "claim": "No independent external validation targets were evaluated.",
            "external_total": 0,
            "external_passed": 0,
            "external_failed": 0,
            "external_not_evaluated": 0,
        }
    external = evaluated[evaluated["evidence_tier"] == "external_independent"]
    counts = external["status"].value_counts()
    total = len(external)
    passed = int(counts.get("pass", 0))
    failed = int(counts.get("fail", 0))
    missing = int(counts.get("not_evaluated", 0))
    if total == 0:
        status = "no_external_targets"
        claim = "Only internal, calibration-holdout, or plausibility evidence was evaluated; external validity was not tested."
    elif missing:
        status = "external_incomplete"
        claim = "External validation is incomplete because one or more preregistered targets could not be evaluated."
    elif failed:
        status = "external_targets_not_met"
        claim = "The model did not meet all preregistered independent external targets."
    else:
        status = "external_targets_met"
        claim = "The model met all declared preregistered independent targets; this supports validity only for these targets, population, period, and scenarios."
    return {
        "claim_status": status,
        "claim": claim,
        "external_total": total,
        "external_passed": passed,
        "external_failed": failed,
        "external_not_evaluated": missing,
    }


def evidence_tier_counts(evaluated: pd.DataFrame) -> list[Mapping[str, object]]:
    """Return compact tier/status counts for an audit export."""
    if evaluated is None or evaluated.empty:
        return []
    grouped = evaluated.groupby(["evidence_tier", "status"], dropna=False).size().reset_index(name="count")
    return grouped.to_dict(orient="records")
