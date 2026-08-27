"""Preliminary Portugal fruit case-study calibration.

The source is a GROCERYsim Firebase export.  This module deliberately keeps the
raw export outside the repository: it returns a de-identified model
configuration containing only the variables required by the ABM.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

from data_processor import (
    archetype_stability_diagnostics,
    assign_archetypes,
    calibrate_behavioral_profiles,
    compute_price_elasticity,
    substitution_choice_diagnostics,
    summarize_baseline_observations,
    summarize_phase2_holdout_targets,
)


PORTUGAL_FRUIT_PIPELINE_VERSION = 1
ORANGE_TOKENS = ("laranja", "laranjas")
CARROT_TOKENS = ("carrot", "cenoura", "cenouras")


def _fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()


def _slug(value: str) -> str:
    folded = re.sub(r"[^a-z0-9]+", "-", _fold(value)).strip("-")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"pt-fruit-{folded[:48]}-{digest}"


def _contains_any(value: object, tokens: tuple[str, ...]) -> bool:
    folded = _fold(value)
    return any(token in folded for token in tokens)


def _count_carrot_material(value: object) -> int:
    """Count carrot-labelled keys/text recursively for the exclusion audit."""
    if isinstance(value, dict):
        return sum(
            int(_contains_any(key, CARROT_TOKENS)) + _count_carrot_material(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_count_carrot_material(item) for item in value)
    return int(isinstance(value, str) and _contains_any(value, CARROT_TOKENS))


def _is_finished(record: dict) -> bool:
    return _fold(record.get("metadata", {}).get("status")) == "finished"


def _is_halle(record: dict) -> bool:
    location = record.get("metadata", {}).get("location", {}) or {}
    return _fold(location.get("city")).strip() == "halle"


def _income_midpoint(value: object) -> float:
    text = str(value or "").replace("€", "").replace(" ", "")
    numbers = [float(x) for x in re.findall(r"\d+(?:[.,]\d+)?", text.replace(",", "."))]
    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2.0
    if numbers:
        return numbers[0]
    return 2500.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _product_group(name: str) -> str:
    return "Orange" if _contains_any(name, ORANGE_TOKENS) else "Other fruit"


def _shelf_life(name: str) -> int:
    folded = _fold(name)
    if any(token in folded for token in ("banana", "manga", "papaia", "abacate")):
        return 7
    if any(token in folded for token in ("melao", "melancia")):
        return 8
    if any(token in folded for token in ("laranja", "limao", "clementina")):
        return 14
    if any(token in folded for token in ("maca", "pera", "kiwi")):
        return 21
    return 12


def _choice_attributes(name: str) -> dict[str, float]:
    folded = _fold(name)
    is_orange = _contains_any(name, ORANGE_TOKENS)
    return {
        "local_portugal": float(is_orange),
        "algarve": float("algarve" in folded),
        # The retail names do not identify size. Keep it structurally missing
        # instead of inventing a value from the DCE photographs.
        "small": 0.0,
        "imperfect": float("zero desperdicio" in folded),
    }


def _cart_rows(record: dict, key: str) -> list[dict]:
    cart = record.get(key, {}).get("cart", [])
    return cart if isinstance(cart, list) else []


def _build_catalogue(records: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    observations: dict[str, list[float]] = {}
    for record in records:
        for key in ("task1_shopping", "task3_intervention"):
            for row in _cart_rows(record, key):
                name = str(row.get("name", "")).strip()
                if not name or _contains_any(name, CARROT_TOKENS):
                    continue
                price = _safe_float(row.get("unitPrice"), 0.0)
                if price > 0:
                    observations.setdefault(name, []).append(price)

    products = []
    for name in sorted(observations):
        attrs = _choice_attributes(name)
        origin = "Portugal (Algarve)" if attrs["algarve"] else "Portugal"
        products.append({
            "id": _slug(name),
            "name": name,
            "category": _product_group(name),
            "price": round(float(np.median(observations[name])), 2),
            "origin": origin,
            "is_domestic": True,
            "is_bio": "biologic" in _fold(name),
            "is_plant_based": True,
            "fat_content": 0.0,
            "shelf_life_days": _shelf_life(name),
            "choice_attributes": attrs,
        })
    return products, {row["name"]: row for row in products}


def _clean_cart(record: dict, key: str, product_map: dict[str, dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for row in _cart_rows(record, key):
        name = str(row.get("name", "")).strip()
        if not name or _contains_any(name, CARROT_TOKENS) or name not in product_map:
            continue
        product = product_map[name]
        quantity = max(1, _safe_int(row.get("qty"), 1))
        price = _safe_float(row.get("unitPrice"), float(product["price"]))
        product_id = product["id"]
        if product_id not in by_id:
            by_id[product_id] = {
                "product_id": product_id,
                "product_name": name,
                "quantity": quantity,
                "price": price,
                "category": product["category"],
                "fat_content": 0.0,
                "is_bio": bool(product["is_bio"]),
                "is_plant_based": True,
            }
        else:
            existing = by_id[product_id]
            total_value = existing["price"] * existing["quantity"] + price * quantity
            existing["quantity"] += quantity
            existing["price"] = total_value / existing["quantity"]
    return list(by_id.values())


def _dce_item_features(item: dict) -> np.ndarray:
    origin = _fold(item.get("origin"))
    image = _fold(item.get("image"))
    return np.asarray([
        _safe_float(item.get("price"), 0.0),
        float("portugal" in origin),
        float("algarve" in origin),
        float("small" in image),
        float("dots" in image),
        0.0,
    ], dtype=float)


def _valid_orange_choices(record: dict) -> list[dict]:
    choices = record.get("task2_choices", [])
    if not isinstance(choices, list):
        return []
    valid = []
    for choice in choices:
        left = choice.get("leftItem")
        right = choice.get("rightItem")
        selected = _fold(choice.get("userSelected"))
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        if selected not in {"left", "right", "none"}:
            continue
        if _safe_float(left.get("price"), 0.0) <= 0 or _safe_float(right.get("price"), 0.0) <= 0:
            continue
        valid.append({"left": left, "right": right, "selected": selected})
    return valid


def _participant_dce_descriptors(record: dict) -> dict:
    choices = _valid_orange_choices(record)
    picked = []
    cheaper = []
    optouts = 0
    for choice in choices:
        if choice["selected"] == "none":
            optouts += 1
            continue
        selected = choice[choice["selected"]]
        other = choice["right" if choice["selected"] == "left" else "left"]
        picked.append(_dce_item_features(selected))
        if abs(_safe_float(selected.get("price")) - _safe_float(other.get("price"))) > 1e-9:
            cheaper.append(float(_safe_float(selected.get("price")) < _safe_float(other.get("price"))))
    means = np.mean(picked, axis=0) if picked else np.asarray([0.0, 0.5, 0.25, 0.5, 0.5, 0.0])
    return {
        "dce_cheaper_bundle_choice_rate": float(np.mean(cheaper)) if cheaper else 0.5,
        "dce_price_sensitivity": float(np.mean(cheaper)) if cheaper else 0.5,
        "local_preference": float(means[1]),
        "algarve_preference": float(means[2]),
        "small_orange_preference": float(means[3]),
        "imperfect_orange_preference": float(means[4]),
        "dce_optout_rate": optouts / max(1, len(choices)),
        "dce_preferences_use_recorded_prices": bool(choices),
    }


def _fit_orange_dce(records: list[dict], random_state: int = 42) -> dict:
    sets = []
    for index, record in enumerate(records):
        for choice in _valid_orange_choices(record):
            X = np.vstack([
                _dce_item_features(choice["left"]),
                _dce_item_features(choice["right"]),
                np.asarray([0, 0, 0, 0, 0, 1], dtype=float),
            ])
            selected = {"left": 0, "right": 1, "none": 2}[choice["selected"]]
            sets.append({"participant": index, "X": X, "chosen": selected})
    participant_ids = sorted({row["participant"] for row in sets})
    if len(participant_ids) < 10:
        return {"status": "insufficient_data", "n_participants": len(participant_ids)}
    train_ids, validation_ids = train_test_split(
        participant_ids, test_size=0.20, random_state=random_state
    )
    training = [row for row in sets if row["participant"] in set(train_ids)]
    validation = [row for row in sets if row["participant"] in set(validation_ids)]

    def objective(beta: np.ndarray) -> float:
        loss = 0.0
        for row in training:
            utility = row["X"] @ beta
            utility -= np.max(utility)
            loss += math.log(float(np.exp(utility).sum())) - float(utility[row["chosen"]])
        return loss / max(1, len(training)) + 0.005 * float(beta @ beta)

    fitted = minimize(
        objective, np.zeros(6), method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7},
    )
    beta = np.asarray(fitted.x, dtype=float)
    validation_loss = 0.0
    correct = 0
    train_shares = np.bincount(
        [row["chosen"] for row in training], minlength=3
    ).astype(float)
    train_shares /= max(1.0, train_shares.sum())
    null_loss = 0.0
    null_correct = 0
    for row in validation:
        utility = row["X"] @ beta
        utility -= np.max(utility)
        probabilities = np.exp(utility) / np.exp(utility).sum()
        validation_loss -= math.log(max(float(probabilities[row["chosen"]]), 1e-12))
        null_loss -= math.log(max(float(train_shares[row["chosen"]]), 1e-12))
        correct += int(int(np.argmax(probabilities)) == row["chosen"])
        null_correct += int(int(np.argmax(train_shares)) == row["chosen"])
    n_validation = max(1, len(validation))
    names = ["price", "local_portugal", "algarve", "small", "imperfect", "optout"]
    model_loss = validation_loss / n_validation
    benchmark_loss = null_loss / n_validation
    return {
        "status": "ok",
        "n_participants": len(participant_ids),
        "n_train_choices": len(training),
        "n_validation_choices": len(validation),
        "validation_accuracy": round(correct / n_validation, 4),
        "null_model_accuracy": round(null_correct / n_validation, 4),
        "validation_log_loss": round(model_loss, 4),
        "null_model_log_loss": round(benchmark_loss, 4),
        "beats_null_benchmark": bool(model_loss < benchmark_loss),
        "beats_majority_benchmark": bool(model_loss < benchmark_loss),
        "model_converged": bool(fitted.success),
        "feature_coefficients": {
            name: round(float(value), 6) for name, value in zip(names, beta)
        },
        # Existing fields keep the engine's price-choice gate generic.
        "price_coefficient": round(float(beta[0]), 6),
        "price_coefficient_estimable": True,
        "utility_scale_compatible_with_price": True,
        "price_source": "recorded_orange_dce_prices_eur_per_kg",
        "price_unit": "EUR_per_displayed_kg",
        "applicable_categories": ["Orange"],
        "choice_model_type": "generic_product_attributes",
        "attribute_scope": "orange_products_only",
        "operational_use": "pooled_orange_candidate_multinomial_probabilities",
        "estimation_sample": "training_participants_only",
        "caution": (
            "Preliminary pooled model. Product-level size is unavailable in the retail "
            "basket catalogue, so the DCE size coefficient is reported but contributes "
            "zero to current retail-SKU ranking."
        ),
    }


def _questionnaire_scores(record: dict) -> dict:
    questionnaire = record.get("task4_questionnaire", {})
    if not isinstance(questionnaire, dict):
        questionnaire = {}
    groups = {
        "price": ("barat", "caro", "qualidadepre", "promoc"),
        "health": ("vitamin", "minera", "natural", "aditivo", "saude", "fibr", "nutrit"),
        "environment": ("ecologic", "recicl", "agricult", "economia_nacional", "localmente"),
        "sensory_habit": ("sabor", "textura", "cheir", "aspeto", "prepar", "dispon"),
    }
    result = {}
    for group, tokens in groups.items():
        values = [
            _safe_float(value)
            for key, value in questionnaire.items()
            if not _contains_any(key, CARROT_TOKENS)
            and any(token in _fold(key) for token in tokens)
            and 1 <= _safe_float(value) <= 5
        ]
        result[f"q_{group}"] = (float(np.mean(values)) - 1.0) / 4.0 if values else 0.5
        result[f"q_{group}_observed_items"] = len(values)
        result[f"q_{group}_fallback"] = not bool(values)
    result["q_animal_welfare"] = 0.5
    result["q_animal_welfare_observed_items"] = 0
    result["q_animal_welfare_fallback"] = True
    return result


def _profile(record: dict, index: int, product_map: dict[str, dict]) -> dict | None:
    baseline = _clean_cart(record, "task1_shopping", product_map)
    if not baseline:
        return None
    phase_two = _clean_cart(record, "task3_intervention", product_map)
    total = _safe_float(
        record.get("task1_shopping", {}).get("total_spent"),
        sum(row["price"] * row["quantity"] for row in baseline),
    )
    max_budget = _safe_float(
        record.get("task3_intervention", {}).get("max_budget_limit"), total
    )
    elasticity = compute_price_elasticity(baseline, phase_two, total, max_budget)
    repeated_one = {row["product_id"]: row["price"] for row in baseline}
    repeated_two = {row["product_id"]: row["price"] for row in phase_two}
    ratios = [
        repeated_two[pid] / repeated_one[pid]
        for pid in repeated_one.keys() & repeated_two.keys()
        if repeated_one[pid] > 0
    ]
    demo = record.get("demographics", {}) or {}
    dce = _participant_dce_descriptors(record)
    return {
        # Sequential local identifiers avoid retaining Firebase IDs.
        "source_id": f"pt_preliminary_{index:03d}",
        "empirical_source_id": f"pt_preliminary_{index:03d}",
        "is_real": True,
        "age": _safe_int(demo.get("idade"), 35),
        "gender": str(demo.get("sexo", "Unknown")),
        "income_midpoint": _income_midpoint(demo.get("rendimento")),
        "household_size": max(1, _safe_int(demo.get("agregado_familiar"), 2)),
        "children": max(0, _safe_int(demo.get("menores"), 0)),
        "employment": str(demo.get("empregados", "Unknown")),
        "buys_oranges": _fold(demo.get("compra_laranjas")) in {"sim", "yes", "1", "true"},
        "consumes_oranges": _fold(demo.get("consome_laranjas")) in {"sim", "yes", "1", "true"},
        "baseline_basket": baseline,
        # Stored for diagnostics only; the ABM always starts from baseline needs.
        "crisis_basket": phase_two,
        "budget": total,
        "crisis_budget": max_budget,
        "has_crisis_observation": "task3_intervention" in record,
        **dce,
        # Backward-compatible engine names; meanings are documented in stats.
        "finnish_preference": dce["local_preference"],
        "organic_preference": 0.2,
        "preferred_fat": 0.0,
        "price_sensitivity": dce["dce_cheaper_bundle_choice_rate"],
        "substitution_rate": 0.5,
        "observed_spending_reduction": float(elasticity["spending_reduction"]),
        "observed_budget_utilization": float(elasticity["budget_utilization"]),
        "observed_substitution_rate": float(elasticity["substitution_rate"]),
        "observed_substitution_lines": int(elasticity["substitution_lines"]),
        "observed_phase2_choice_lines": int(elasticity["phase2_choice_lines"]),
        "baseline_choice_lines": int(elasticity["baseline_choice_lines"]),
        "observed_quantity_retention": float(elasticity["quantity_retention"]),
        "observed_round2_spend": float(elasticity["round2_actual_spend"]),
        "observed_price_shock": float(np.median(ratios) - 1.0) if ratios else None,
        **_questionnaire_scores(record),
        "reference_price": float(np.mean([row["price"] for row in baseline])),
        "archetype": None,
        "cluster_id": -1,
    }


def build_portugal_fruit_config(
    export: dict,
    pool_size: int = 2000,
    n_archetypes: int = 4,
) -> dict:
    """Return an ABM configuration from a preliminary Portugal Firebase export."""
    if not isinstance(export, dict):
        raise TypeError("Portugal Firebase export must be a JSON object keyed by session.")
    raw_records = [record for record in export.values() if isinstance(record, dict)]
    finished = [record for record in raw_records if _is_finished(record)]
    halle = [record for record in finished if _is_halle(record)]
    eligible = [record for record in finished if not _is_halle(record)]
    products, product_map = _build_catalogue(eligible)
    profiles = []
    for index, record in enumerate(eligible, start=1):
        built = _profile(record, index, product_map)
        if built is not None:
            profiles.append(built)
    if not profiles:
        raise ValueError("No eligible completed Portugal fruit shopping profiles were found.")

    dce = _fit_orange_dce(eligible)
    stability = archetype_stability_diagnostics(profiles, selected_k=n_archetypes)
    profiles = assign_archetypes(
        profiles, n_clusters=n_archetypes,
        operational=stability.get("archetypes_supported", False),
    )
    profiles, calibration = calibrate_behavioral_profiles(profiles)
    substitution = substitution_choice_diagnostics(profiles, products)
    baseline_targets = summarize_baseline_observations(profiles, products)
    phase2_targets = summarize_phase2_holdout_targets(profiles, calibration)
    carrot_fields_in_eligible = sum(
        _count_carrot_material(record.get(section_name, {}))
        for record in eligible
        for section_name in (
            "task2_followup_question_scales",
            "task2_followup_statement_scales",
            "task4_questionnaire",
        )
    )
    carrot_fields_ignored = sum(
        _count_carrot_material(record)
        for record in raw_records
    )
    categories = Counter(product["category"] for product in products)
    orange_products = sum(1 for product in products if product["category"] == "Orange")
    return {
        "products": products,
        "population": profiles,
        "population_target_size": int(pool_size),
        "stats": {
            "population_pipeline_version": 9,
            "case_study_pipeline_version": PORTUGAL_FRUIT_PIPELINE_VERSION,
            "case_study": "Portugal — Fruits (Oranges)",
            "data_status": "preliminary_ongoing_collection",
            "raw_sessions": len(raw_records),
            "finished_sessions": len(finished),
            "halle_finished_excluded": len(halle),
            "carrot_fields_ignored": carrot_fields_ignored,
            "carrot_fields_in_eligible_profiles": carrot_fields_in_eligible,
            "n_real": len(profiles),
            "n_skipped": len(eligible) - len(profiles),
            "pool_size": int(pool_size),
            "empirical_sampling_units": len(profiles),
            "population_method": "seeded_complete_profile_resampling_with_replacement",
            "synthetic_attribute_jitter": False,
            "catalogue_rows_raw": len(products),
            "catalogue_skus": len(products),
            "catalogue_duplicates_collapsed": 0,
            "catalogue_categories": dict(categories),
            "orange_skus": orange_products,
            "dce_choice_validation": dce,
            "questionnaire_reliability": {
                "status": "descriptive_preliminary",
                "method": "keyword-mapped fruit-value constructs; no CFA",
                "caution": "Construct mapping must be replaced by the final Portuguese codebook.",
            },
            "archetype_stability": stability,
            "behavioral_calibration": calibration,
            "substitution_choice_validation": substitution,
            "baseline_reproduction_targets": baseline_targets,
            "phase2_reproduction_targets": phase2_targets,
            "field_semantics": {
                "finnish_preference": "compatibility alias for local Portuguese origin preference",
                "preferred_fat": "unused compatibility field for fruit model",
            },
            "scientific_cautions": [
                "Data collection is ongoing; all estimates are preliminary.",
                "Halle participants and carrot-specific fields are excluded.",
                "Orange DCE coefficients are pooled and do not identify individual WTP heterogeneity.",
                "Retail SKU names do not encode orange size, so DCE size is not operationally mapped.",
                "Cross-type fruit substitution remains an exploratory allocation rule where validation gates fail.",
            ],
        },
    }


def deidentified_calibration_summary(config: dict) -> dict:
    """Small JSON-serialisable summary suitable for display or audit."""
    stats = deepcopy(config.get("stats", {}))
    return {
        "case_study": stats.get("case_study"),
        "data_status": stats.get("data_status"),
        "sample": {
            "raw_sessions": stats.get("raw_sessions"),
            "finished_sessions": stats.get("finished_sessions"),
            "halle_finished_excluded": stats.get("halle_finished_excluded"),
            "eligible_profiles": stats.get("n_real"),
            "carrot_fields_ignored": stats.get("carrot_fields_ignored"),
            "carrot_fields_in_eligible_profiles": stats.get(
                "carrot_fields_in_eligible_profiles"
            ),
        },
        "catalogue": {
            "skus": stats.get("catalogue_skus"),
            "orange_skus": stats.get("orange_skus"),
            "categories": stats.get("catalogue_categories"),
        },
        "orange_dce": stats.get("dce_choice_validation", {}),
        "phase2_calibration": stats.get("behavioral_calibration", {}),
        "cautions": stats.get("scientific_cautions", []),
    }
