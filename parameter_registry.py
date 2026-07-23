"""Machine-readable evidence registry for influential GROCERYsim parameters.

The registry is deliberately conservative: a parameter is called empirical only
when its value is read directly from GROCERYsim data, and calibrated only when a
documented estimation/validation step determines its value. Literature transfer
and engineering choices remain assumptions until validated in the target setting.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


EVIDENCE_CLASSES = {
    "observed_data",
    "heldout_calibrated",
    "cross_fitted_calibrated",
    "literature_transfer",
    "scenario_input",
    "engineering_assumption",
}
PRIORITIES = {"critical", "high", "medium", "low"}


def _row(
    parameter_id: str,
    parameter: str,
    component: str,
    current_value: Any,
    unit: str,
    evidence_class: str,
    source: str,
    identifiable: bool,
    validation: str,
    minimum: float | int | None,
    maximum: float | int | None,
    priority: str,
    uncertainty_treatment: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "parameter_id": parameter_id,
        "parameter": parameter,
        "component": component,
        "current_value": current_value,
        "unit": unit,
        "evidence_class": evidence_class,
        "source": source,
        "identifiable_from_current_data": identifiable,
        "validation": validation,
        "admissible_min": minimum,
        "admissible_max": maximum,
        "priority": priority,
        "uncertainty_treatment": uncertainty_treatment,
        "interpretation": interpretation,
    }


def build_parameter_registry(
    stats: dict | None = None,
    runtime_params: dict | None = None,
) -> list[dict[str, Any]]:
    """Return the evidence registry, updated with this dataset/run where possible."""
    stats = stats or {}
    p = runtime_params or {}
    behavioural = stats.get("behavioral_calibration", {})
    dce = stats.get("dce_choice_validation", {})
    substitution_choice = stats.get("substitution_choice_validation", {})
    reliability = stats.get("questionnaire_reliability", {})
    archetype_audit = stats.get("archetype_stability", {})
    policy = p.get("policy_cfg", {}) or {}

    dce_ok = (
        dce.get("status") == "ok"
        and dce.get("beats_null_benchmark", dce.get("beats_majority_benchmark", False))
        and dce.get("model_converged", True)
    )
    behavioural_ok = behavioural.get("status") == "ok"
    exploratory = bool(p.get("exploratory_behaviour", False))
    archetypes_supported = bool(archetype_audit.get("archetypes_supported", False))

    rows = [
        _row("product.base_price", "SKU base price", "Products", "catalogue", "EUR/unit",
             "observed_data", "Unity/master-products catalogue", True,
             "Schema and duplicate-SKU checks; no external price validation", 0, None, "high",
             "Use observed SKU values; refresh for each study period", "Baseline shelf price before scenarios and policies"),
        _row("product.attributes", "SKU origin/organic/fat attributes", "Products", "catalogue", "mixed",
             "observed_data", "Unity/master-products catalogue", True,
             "Canonical-name consistency checks only", None, None, "high",
             "Audit against retailer/product records", "Attributes entering product utility and policy eligibility"),
        _row("product.shelf_life", "Maximum shelf life", "Inventory", "catalogue", "days",
             "observed_data", "Unity/master-products catalogue", True,
             "No batch-level spoilage validation", 1, None, "high",
             "Vary by SKU; validate with retailer shrink records", "Age at which stock is recorded as waste"),
        _row("household.baseline_basket", "Phase-one basket quantities", "Demand", "participant-specific", "units/visit",
             "observed_data", "GROCERYsim phase-one shopping task", True,
             "Catalogue linkage and basket accounting checks", 0, None, "critical",
             "Bootstrap participants; report sampling uncertainty", "Agent needs and initial pantry composition"),
        _row("household.budget", "Shopping budget", "Demand", "participant-specific", "EUR/visit",
             "observed_data", "GROCERYsim phase-one and phase-two tasks", True,
             "Hard budget invariant tested", 0, None, "critical",
             "Preserve empirical distribution", "Maximum spending per simulated visit"),
        _row("household.questionnaire_factors", "Belief/motive factor scores", "Behaviour", "participant-specific", "0-1",
             "observed_data", "GROCERYsim 21-item beliefs questionnaire", True,
             ("Reliability audit reported; not all constructs passed" if reliability.get("status") == "ok" and not reliability.get("all_constructs_acceptable")
              else "Reliability audit passed" if reliability.get("all_constructs_acceptable")
              else "Construct reliability not available"), 0, 1, "high",
             "Propagate measurement uncertainty; verify item keys and factor structure", "Inputs to exploratory clusters and behavioural prediction"),
        _row("population.pool_size", "Simulation household draw size", "Population", p.get("pool_size", stats.get("pool_size", 2000)), "households",
             "scenario_input", "Analyst-selected simulation scale", False,
             "Complete empirical profiles resampled per model seed; empirical n reported separately", 100, 50000, "high",
             "Test convergence and retain participant-level outer uncertainty", "Number of persistent resampled household instances"),
        _row("population.archetypes", "Number of behavioural archetypes", "Population", p.get("n_archetypes", archetype_audit.get("selected_k", 4)), "clusters",
             "engineering_assumption", "K-means design choice", False,
             ("Operational gate passed" if archetype_audit.get("archetypes_supported")
              else "Operational gate failed or unavailable; categorical modifiers disabled"), 2, 5, "high",
             "Bootstrap ARI, silhouette, minimum size, and alternative-k comparison", "Exploratory labels; operational only when all gates pass"),
        _row("population.profile_resampling", "Participant-profile resampling", "Population", "complete profiles; no jitter", "method",
             "engineering_assumption", "Seeded empirical bootstrap", False,
             "Preserves all within-participant joint relationships and source provenance", None, None, "medium",
             "Resample independently across Monte Carlo seeds; report empirical n", "Propagates participant sampling uncertainty without synthetic attributes"),
        _row("behaviour.evidence_mode", "Behavioural evidence mode", "Behaviour",
             "exploratory extensions" if exploratory else "empirical only", "mode",
             "scenario_input", "Explicit analyst opt-in", False,
             ("Unidentified dynamic mechanisms enabled" if exploratory
              else "Unidentified dynamic mechanisms disabled"), None, None, "critical",
             "Report the mode with every result; use exploratory mode only in global sensitivity analysis",
             "Controls panic, TPB, Prospect Theory, archetype modifiers, and preference learning"),
        _row("dce.origin_weight", "Finnish-origin pooled logit coefficient", "Choice evidence", dce.get("origin_coefficient"), "log-odds",
             "heldout_calibrated" if dce_ok else "engineering_assumption", "GROCERYsim choice experiment; participant holdout", dce_ok,
             "Held-out choice accuracy beats majority benchmark" if dce_ok else "DCE unavailable or did not beat benchmark", None, None, "critical",
             "Bootstrap coefficient uncertainty and repeat participant holdouts", "Pooled milk-domain coefficient estimated jointly with recorded price and opt-out; not individual WTP"),
        _row("dce.organic_weight", "Organic pooled logit coefficient", "Choice evidence", dce.get("organic_coefficient"), "log-odds",
             "heldout_calibrated" if dce_ok else "engineering_assumption", "GROCERYsim choice experiment; participant holdout", dce_ok,
             "Held-out choice accuracy beats majority benchmark" if dce_ok else "DCE unavailable or did not beat benchmark", None, None, "critical",
             "Bootstrap coefficient uncertainty and repeat participant holdouts", "Pooled milk-domain coefficient estimated jointly with recorded price and opt-out; not individual WTP"),
        _row("dce.fat_weight", "Fat pooled logit coefficients", "Choice evidence",
             f"{dce.get('fat_linear_coefficient')}/{dce.get('fat_quadratic_coefficient')}", "linear/quadratic log-odds",
             "heldout_calibrated" if dce_ok else "engineering_assumption", "GROCERYsim choice experiment; participant holdout", dce_ok,
             "Held-out choice accuracy beats majority benchmark" if dce_ok else "DCE unavailable or did not beat benchmark", None, None, "critical",
             "Bootstrap coefficient uncertainty and inspect functional form", "Pooled milk-domain coefficient estimated jointly with recorded price and opt-out"),
        _row("choice.substitute_ranking", "Substitute attribute compatibility ranking", "Choice allocation",
             ("DCE multinomial probabilities for Milk; participant compatibility in " + ", ".join(substitution_choice.get("supported_ranking_categories", [])))
             if substitution_choice.get("supported_ranking_categories")
             else "DCE multinomial probabilities for Milk; validated phase-transition shares or seeded uniform elsewhere", "allocation rule",
             "heldout_calibrated" if dce_ok or substitution_choice.get("supported_transition_categories") else "engineering_assumption",
             "DCE participant holdout plus phase-one/phase-two basket transitions", bool(dce_ok or substitution_choice.get("supported_transition_categories")),
             f"{substitution_choice.get('n_unambiguous_events', 0)} unambiguous events; {substitution_choice.get('ranking_gate', 'validation gate unavailable')}", None, None, "high",
             "Collect explicit candidate choice sets or substantially more replacement events", "Seeded stochastic fallback is used when no category clears the predictive gate"),
        _row("choice.substitute_price_gate", "Retention-price threshold transferred to substitute SKUs", "Choice allocation",
             "enabled" if substitution_choice.get("candidate_price_gate_supported") else "disabled", "gate",
             "heldout_calibrated" if substitution_choice.get("candidate_price_gate_supported") else "engineering_assumption",
             "Observed replacement-target coverage", bool(substitution_choice.get("candidate_price_gate_supported")),
             f"{substitution_choice.get('n_unambiguous_events', 0)} events; observed target coverage {substitution_choice.get('candidate_price_gate_target_coverage')}; gate requires n>=100 and coverage>=0.90", None, None, "critical",
             "Use explicit replacement candidate sets and prices", "Remaining visit budget is always enforced separately"),
        _row("dce.price_coefficient", "DCE price coefficient", "Choice utility", dce.get("price_coefficient"), "utility/EUR",
             "heldout_calibrated" if dce.get("price_coefficient_estimable") and dce_ok else "engineering_assumption",
             dce.get("price_source", "Displayed DCE prices absent"),
             bool(dce.get("price_coefficient_estimable") and dce_ok),
             (f"Held-out log loss {dce.get('validation_log_loss')} vs null {dce.get('null_model_log_loss')}"
              if dce.get("price_coefficient_estimable") else "Structurally non-identifiable in current export"),
             None, None, "critical",
             "Bootstrap the participant-held-out estimate and avoid individual WTP claims", "Pooled milk-choice price effect in utility per displayed EUR"),
        _row("behaviour.price_response", "Household price-response intensity", "Behaviour", "cross-fitted profiles" if behavioural_ok else 0.5, "0-1",
             "cross_fitted_calibrated" if behavioural_ok else "engineering_assumption", "Phase-one predictors and phase-two quantity response", behavioural_ok,
             ("Held-out MAE vs naive; individual model " + ("retained" if behavioural.get("price_response_model_retained") else "rejected")) if behavioural_ok else "Insufficient phase-two data",
             0, 1, "critical", "Use population mean when held-out model loses; bootstrap the mean", "Response to price loss, not a demand elasticity"),
        _row("behaviour.substitution", "Household substitution propensity", "Behaviour", "cross-fitted profiles" if behavioural_ok else 0.5, "0-1",
             "cross_fitted_calibrated" if behavioural_ok else "engineering_assumption", "Phase-one predictors and phase-two product changes", behavioural_ok,
             ("Nested line-action MAE vs naive; operational model " + ("retained" if behavioural.get("substitution_action_model_retained") else "rejected")) if behavioural_ok else "Insufficient phase-two data",
             0, 1, "critical", "Use population mean when held-out model loses; bootstrap predictions", "Probability/tolerance used when desired SKU is unavailable"),
        _row("behaviour.hoarding_propensity", "Household phase-transition quantity increase", "Behaviour", "cross-fitted profiles" if behavioural_ok else 0.0, "0-1",
             "cross_fitted_calibrated" if behavioural_ok else "engineering_assumption", "Phase-one predictors and phase-two quantity increase", behavioural_ok,
             ("Held-out MAE vs naive; individual model " + ("retained" if behavioural.get("hoarding_model_retained") else "rejected")) if behavioural_ok else "Insufficient phase-two data",
             0, 1, "critical", "Distinguish experiment response from real-world panic hoarding", "Scales scenario-level maximum hoarding"),
        _row("behaviour.budget_utilization", "Phase-two reservation spending share", "Behaviour", "cross-fitted profiles" if behavioural_ok else 1.0, "share of maximum budget",
             "cross_fitted_calibrated" if behavioural_ok else "engineering_assumption", "Phase-one predictors and phase-two budget utilization", behavioural_ok,
             ("Held-out MAE vs naive; individual model " + ("retained" if behavioural.get("budget_utilization_model_retained") else "rejected in favour of fold mean")) if behavioural_ok else "Insufficient phase-two data",
             0, 1, "critical", "Bootstrap calibration uncertainty and validate on repeated occasions", "Separates maximum available budget from the amount a household is willing to spend"),
        _row("behaviour.revealed_margin", "Revealed-preference purchase margin", "Choice utility", behavioural.get("revealed_preference_margin", 0.0), "utility",
             "heldout_calibrated" if behavioural_ok else "engineering_assumption", "Phase-two aggregate quantity retention", behavioural_ok,
             "Tuned on training participants; evaluated on held-out participants" if behavioural_ok else "Insufficient phase-two data", 0, 0.25, "critical",
             "Propagate calibration error; reject individual interpretation", "Maximum accepted incremental price loss in the one-occasion rule"),
        _row("behaviour.loss_aversion", "Loss-aversion coefficient", "Choice utility", 2.25 if exploratory else "disabled", "ratio",
             "literature_transfer", "Tversky & Kahneman (1992)", False,
             "Not validated in the GROCERYsim sample", 1, 4, "high",
             "Global sensitivity and target-sample estimation", "Weights price losses relative to gains"),
        _row("behaviour.value_curvature", "Prospect-theory curvature", "Choice utility", 0.88 if exploratory else "disabled", "exponent",
             "literature_transfer", "Tversky & Kahneman (1992)", False,
             "Not validated in the GROCERYsim sample", 0.5, 1.2, "high",
             "Global sensitivity and target-sample estimation", "Curvature of price gain/loss value"),
        _row("behaviour.tpb_weights", "TPB attitude/norm/control weights", "Choice utility", "0.430/0.228/0.342" if exploratory else "disabled", "normalized weights",
             "literature_transfer", "Armitage & Conner (2001) meta-analysis", False,
             "Transferred relative weights normalized to sum to one; not target-sample estimates", 0, 1, "critical",
             "Re-estimate in the target sample and compare alternative specifications", "Convex combination of attitude, subjective norm and perceived control"),
        _row("behaviour.learning_rate", "Preference learning rate", "Learning", 0.015 if exploratory and archetypes_supported else "disabled", "fraction/day",
             "engineering_assumption", "Model design choice", False,
             "No longitudinal validation", 0, 0.1, "high",
             "Sensitivity analysis; calibrate with repeated observations", "Daily movement toward experienced attributes"),
        _row("behaviour.utility_threshold", "Price acceptance rule", "Choice acceptance", "price_response × proportional price change ≤ calibrated margin", "incremental price loss",
             "heldout_calibrated" if behavioural_ok else "engineering_assumption", "Phase-transition retention calibration", behavioural_ok,
             "One-occasion held-out mean and individual skill reported", 0, 1, "critical",
             "Bootstrap calibration error; validate multi-day trajectories separately", "Determines whether the requested SKU is price-acceptable; substitute transfer is separately gated"),
        _row("demand.daily_visitors", "Base daily consumers", "Demand", p.get("base_con", 100), "visitors/day",
             "scenario_input", "Analyst-selected store scenario", False,
             "No retailer footfall calibration", 10, 5000, "critical",
             "Calibrate to store transaction counts", "Scale of daily household visits"),
        _row("demand.seasonality", "Month and weekday traffic factors", "Demand", "enabled" if p.get("traffic_variation", False) else "disabled", "multiplier",
             "engineering_assumption", "Model design choice", False,
             "No retailer footfall validation", 0.5, 1.5, "high",
             "Replace with empirical calendar effects", "Systematic variation in visitors"),
        _row("demand.visit_interval", "Expected household revisit interval", "Demand", "derived", "days",
             "engineering_assumption", "Pool size / expected daily traffic", False,
             "No observed shopping-frequency variable", 1, None, "critical",
             "Collect/calibrate shopping frequency", "Schedules persistent household visits and pantry depletion"),
        _row("demand.traffic_variation", "Calendar and random footfall variation", "Demand",
             "enabled" if p.get("traffic_variation", False) else "disabled", "mode",
             "engineering_assumption", "Fixed weekday/month tables and ±10% seeded noise", False,
             "No GROCERYsim footfall time series; disabled by default", None, None, "critical",
             "Estimate from store visit or transaction counts", "When disabled, declared base daily consumers is exact"),
        _row("crisis.inflation", "Crisis price inflation", "Crisis", p.get("inf", 25.0), "percent",
             "scenario_input", "Analyst-defined stress scenario", False,
             "Scenario value, not estimated from participant data", 0, 150, "critical",
             "Use documented scenario distributions", "Uniform crisis price shock"),
        _row("crisis.disruption_days", "Supply disruption duration", "Crisis", p.get("dis", 5), "days",
             "scenario_input", "Analyst-defined stress scenario", False,
             "Scenario value, not estimated from participant data", 0, 30, "critical",
             "Use documented scenario distributions", "Blocks inbound deliveries"),
        _row("crisis.panic_sensitivity", "Panic sensitivity", "Crisis", p.get("panic", 0.5), "0-1",
             "scenario_input", "Analyst-defined; no direct panic-belief item in current export", False,
             "Not empirically identified", 0, 1, "critical",
             "Always vary globally; collect validated panic/scarcity measures", "Scales panic response to scarcity and inflation"),
        _row("crisis.hoarding_multiplier", "Maximum hoarding multiplier", "Crisis", p.get("hoard", 1.5), "multiplier",
             "scenario_input", "Analyst-defined maximum scaled by household propensity", False,
             "Not empirically identified as a real-world multiplier", 1, 3, "critical",
             "Always vary globally; validate against crisis transaction panels", "Upper bound on crisis purchase quantities"),
        _row("crisis.panic_exposure_floor", "Normal scarcity exposure floor", "Crisis", p.get("panic_exposure_floor", 0.10), "shopper share",
             "engineering_assumption", "Model design choice", False,
             "No time-series panic validation", 0, 1, "critical",
             "Global sensitivity; calibrate jointly to longitudinal data", "Scarcity exposure treated as ordinary retail friction"),
        _row("crisis.panic_growth_rate", "Scarcity-to-panic growth rate", "Crisis", p.get("panic_growth_rate", 0.50), "per day",
             "engineering_assumption", "Model design choice", False,
             "No time-series panic validation", 0, 1, "critical",
             "Global sensitivity; calibrate jointly to longitudinal data", "Amplifies scarcity exposure above the floor"),
        _row("crisis.panic_decay_active", "Active-phase panic decay", "Crisis", p.get("panic_decay_active", 0.05), "per day",
             "engineering_assumption", "Model design choice", False,
             "No time-series panic validation", 0, 1, "critical",
             "Global sensitivity; calibrate jointly to longitudinal data", "Reduces panic each active-crisis day"),
        _row("crisis.panic_decay_recovery", "Recovery-phase panic decay", "Crisis", p.get("panic_decay_recovery", 0.10), "per day",
             "engineering_assumption", "Model design choice", False,
             "No time-series panic validation", 0, 1, "critical",
             "Global sensitivity; calibrate jointly to longitudinal data", "Reduces panic after crisis conditions normalize"),
        _row("crisis.inflation_panic_rate", "Inflation-to-panic rate", "Crisis", p.get("inflation_panic_rate", 0.40), "per day",
             "engineering_assumption", "Model design choice", False,
             "No time-series panic validation", 0, 1, "critical",
             "Global sensitivity; calibrate jointly to longitudinal data", "Maps scenario inflation into the daily panic signal"),
        _row("crisis.media_panic_rate", "Panic-media amplification rate", "Crisis", 0.12, "per day",
             "engineering_assumption", "Model design choice", False,
             "No communication treatment-effect validation", 0, 1, "high",
             "Expose and estimate from communication experiments", "Maps panic-framed media intensity into panic"),
        _row("crisis.media_calming_rate", "Calming-media reduction rate", "Crisis", 0.10, "per day",
             "engineering_assumption", "Model design choice", False,
             "No communication treatment-effect validation", 0, 1, "high",
             "Expose and estimate from communication experiments", "Maps calming-media intensity into panic reduction"),
        _row("inventory.reorder_point", "Storage reorder point", "Logistics", p.get("reorder", 0.30), "capacity share",
             "scenario_input", "Analyst/store-policy input", False,
             "Inventory conservation tested; performance not externally validated", 0.10, 0.90, "high",
             "Optimize only within retailer-feasible ranges", "Triggers supplier order"),
        _row("inventory.target_stock", "Restock target", "Logistics", p.get("target", 0.90), "capacity share",
             "scenario_input", "Analyst/store-policy input", False,
             "Inventory conservation tested; performance not externally validated", 0.50, 1.00, "high",
             "Optimize only within retailer-feasible ranges", "Target storage after delivery"),
        _row("inventory.lead_time", "Supplier lead time", "Logistics", p.get("lead", 2), "days",
             "scenario_input", "Analyst/store-policy input", False,
             "No supplier-record calibration", 1, 14, "high",
             "Use empirical supplier lead-time distribution", "Delay between order and delivery"),
        _row("inventory.capacity_rules", "Shelf/storage cover and initial fill", "Logistics", "1.5/2.5/4 days; 75%/60%", "mixed",
             "engineering_assumption", "Traffic-scaled engineering heuristic", False,
             "Inventory accounting tested; capacities not retailer-validated", 0, None, "critical",
             "Calibrate per SKU/store; include delivery variance and service target", "Sets capacity and initial inventory"),
        _row("inventory.near_expiry", "Near-expiry window and discount", "Inventory", "2 days; 50%", "mixed",
             "engineering_assumption", "Model design choice", False,
             "No retailer markdown validation", 0, None, "high",
             "Estimate SKU/store markdown policy and demand response", "Markdown timing and price reduction"),
        _row("welfare.access_thresholds", "Consumption-shortfall severity thresholds", "Welfare", "0/25/50/90%", "daily shortfall",
             "engineering_assumption", "Model-specific access-stress index; not FIES", False,
             "Internal consistency tests only", 0, 1, "critical",
             "Validate against an external food-access measure; retain non-FIES label", "Classifies unmet daily home consumption"),
        _row("environment.co2_factors", "Production and waste CO2 factors", "Environment", "0.8-3.5", "kg CO2e/unit",
             "engineering_assumption", "Provisional model constants; no traceable product LCA mapping", False,
             "No unit/mass or SKU-specific LCA validation", 0, None, "critical",
             "Replace with mass-based, sourced SKU/category LCA factors", "Attributes emissions to sales and waste"),
        _row("policy.fat_tax", "Fat-tax threshold/rate", "Policy", f"{policy.get('fat_tax_threshold', 3.5)}/{policy.get('fat_tax_rate', 0.20)}", "g/100ml; share",
             "scenario_input", "Analyst-defined policy counterfactual", False,
             "Mechanism check only; no demand-policy validation", 0, None, "high",
             "Use enacted/design values and price-elasticity uncertainty", "Price surcharge for eligible products"),
        _row("policy.subsidy", "Domestic/organic subsidy rate", "Policy", policy.get("subsidy_rate", 0.15), "price share",
             "scenario_input", "Analyst-defined policy counterfactual", False,
             "Mechanism check only; no demand-policy validation", 0, 0.40, "high",
             "Use documented policy design and behavioural uncertainty", "Discount applied to eligible products"),
        _row("policy.labelling_boost", "Labelling compatibility-weight boosts", "Policy",
             (f"{policy.get('labelling_health_boost', 0.15)}/{policy.get('labelling_organic_boost', 0.10)}" if exploratory else "disabled"), "ranking-weight increment",
             "engineering_assumption", "Model design choice", False,
             "No treatment-effect estimate", 0, 0.40, "critical",
             "Estimate from randomized information treatment", "Exploratory milk-candidate ranking effect only; disabled in empirical-only mode"),
    ]
    return rows


def validate_parameter_registry(rows: list[dict[str, Any]]) -> list[str]:
    """Return validation errors; an empty list means the registry is well formed."""
    errors: list[str] = []
    required = {
        "parameter_id", "parameter", "component", "current_value", "unit",
        "evidence_class", "source", "identifiable_from_current_data",
        "validation", "admissible_min", "admissible_max", "priority",
        "uncertainty_treatment", "interpretation",
    }
    ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            errors.append(f"row {index} missing: {', '.join(sorted(missing))}")
            continue
        pid = str(row["parameter_id"])
        if pid in ids:
            errors.append(f"duplicate parameter_id: {pid}")
        ids.add(pid)
        if row["evidence_class"] not in EVIDENCE_CLASSES:
            errors.append(f"{pid}: invalid evidence_class")
        if row["priority"] not in PRIORITIES:
            errors.append(f"{pid}: invalid priority")
        lo, hi = row["admissible_min"], row["admissible_max"]
        if lo is not None and hi is not None and lo > hi:
            errors.append(f"{pid}: admissible_min exceeds admissible_max")
        if not str(row["source"]).strip() or not str(row["uncertainty_treatment"]).strip():
            errors.append(f"{pid}: source and uncertainty treatment are required")
        if row["evidence_class"] in {"engineering_assumption", "scenario_input", "literature_transfer"} and row["identifiable_from_current_data"]:
            errors.append(f"{pid}: assumption/transfer cannot be marked identifiable")
    return errors


def parameter_registry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["evidence_class"] for row in rows)
    unresolved = [
        row for row in rows
        if row["priority"] in {"critical", "high"}
        and not row["identifiable_from_current_data"]
    ]
    return {
        "n_parameters": len(rows),
        "evidence_counts": dict(counts),
        "n_identifiable": sum(bool(row["identifiable_from_current_data"]) for row in rows),
        "n_unresolved_high_priority": len(unresolved),
        "policy_grade_ready": len(unresolved) == 0,
    }
