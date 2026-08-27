"""Synthetic development configuration for the Greece dairy case study.

No participant-level or retail data are currently available for this case.  The
generator therefore creates a deterministic, auditable demonstration population
and catalogue.  Its outputs are scenario illustrations, not calibrated estimates.
"""

from __future__ import annotations

import random
from collections import Counter


GREECE_DAIRY_SYNTHETIC_VERSION = 1
SYNTHETIC_TEMPLATE_COUNT = 240
SYNTHETIC_SIMULATION_POOL = 1200


_PRODUCT_SPECS = (
    # id, name, category, price EUR/unit, fat %, organic, domestic, shelf life
    ("gr-milk-fresh-15", "Fresh cow milk 1.5% — 1 L", "Milk", 1.62, 1.5, False, True, 8),
    ("gr-milk-fresh-35", "Fresh whole cow milk 3.5% — 1 L", "Milk", 1.78, 3.5, False, True, 8),
    ("gr-milk-lactose-free", "Lactose-free milk — 1 L", "Milk", 2.18, 1.5, False, True, 12),
    ("gr-milk-organic", "Organic fresh cow milk — 1 L", "Milk", 2.35, 3.5, True, True, 8),
    ("gr-yogurt-strained-2", "Greek strained yogurt 2% — 200 g", "Yogurt", 1.18, 2.0, False, True, 24),
    ("gr-yogurt-strained-10", "Greek strained yogurt 10% — 200 g", "Yogurt", 1.32, 10.0, False, True, 24),
    ("gr-yogurt-sheep", "Traditional sheep yogurt — 220 g", "Yogurt", 1.58, 6.5, False, True, 18),
    ("gr-cheese-feta", "Feta PDO — 200 g", "Cheese", 3.45, 21.0, False, True, 35),
    ("gr-cheese-feta-light", "Reduced-fat feta-style cheese — 200 g", "Cheese", 3.25, 12.0, False, True, 30),
    ("gr-cheese-kasseri", "Kasseri PDO — 200 g", "Cheese", 4.10, 25.0, False, True, 45),
    ("gr-cream-cooking", "Cooking cream — 200 ml", "Cream", 1.52, 15.0, False, True, 25),
    ("gr-butter-cow", "Cow butter — 250 g", "Butter", 3.65, 82.0, False, True, 60),
)


def _products() -> list[dict]:
    products = []
    for product_id, name, category, price, fat, organic, domestic, shelf_life in _PRODUCT_SPECS:
        products.append({
            "id": product_id,
            "name": name,
            "category": category,
            "price": price,
            "origin": "Greece" if domestic else "EU import",
            "is_domestic": domestic,
            "is_bio": organic,
            "is_plant_based": False,
            "fat_content": fat,
            "shelf_life_days": shelf_life,
        })
    return products


def _basket_item(product: dict, quantity: int) -> dict:
    return {
        "product_id": product["id"],
        "product_name": product["name"],
        "category": product["category"],
        "quantity": int(quantity),
        "price": float(product["price"]),
        "fat_content": float(product["fat_content"]),
        "is_bio": bool(product["is_bio"]),
        "is_plant_based": False,
    }


def _synthetic_profiles(products: list[dict], count: int) -> list[dict]:
    """Create deterministic whole-household templates from declared assumptions."""
    rng = random.Random(20260827)
    by_category: dict[str, list[dict]] = {}
    for product in products:
        by_category.setdefault(product["category"], []).append(product)

    income_points = (900.0, 1400.0, 2100.0, 3000.0, 4200.0)
    price_sensitivities = (0.30, 0.42, 0.55, 0.68, 0.80)
    profiles = []
    for index in range(count):
        household_size = 1 + (index % 4)
        milk = rng.choice(by_category["Milk"])
        yogurt = rng.choice(by_category["Yogurt"])
        cheese = rng.choice(by_category["Cheese"])
        basket = [
            _basket_item(milk, 1 + int(household_size >= 3)),
            _basket_item(yogurt, 1 + int(household_size >= 2)),
            _basket_item(cheese, 1),
        ]
        if index % 3 == 0:
            basket.append(_basket_item(by_category["Cream"][0], 1))
        if index % 4 == 0:
            basket.append(_basket_item(by_category["Butter"][0], 1))
        basket_value = sum(row["price"] * row["quantity"] for row in basket)
        income = income_points[index % len(income_points)]
        price_sensitivity = price_sensitivities[index % len(price_sensitivities)]
        profiles.append({
            "source_id": f"synthetic-gr-dairy-{index + 1:03d}",
            "empirical_source_id": f"synthetic-gr-dairy-{index + 1:03d}",
            "is_real": False,
            "is_synthetic": True,
            "age": 20 + (index * 7) % 55,
            "gender": ("Woman", "Man", "Other / not specified")[index % 3],
            "household_size": household_size,
            "income_midpoint": income,
            "baseline_basket": basket,
            "crisis_basket": [dict(row) for row in basket],
            "budget": round(max(basket_value * 1.35, 12.0), 2),
            "crisis_budget": round(max(basket_value * 1.45, 13.0), 2),
            "budget_utilization_propensity": 0.92,
            "price_sensitivity": price_sensitivity,
            "revealed_preference_margin": 0.08,
            "substitution_rate": 0.35,
            "sub_tolerance": 0.50,
            "finnish_preference": 0.72,  # compatibility alias: Greek-origin preference
            "organic_preference": 0.18,
            "preferred_fat": float(milk["fat_content"]),
            "reference_price": round(basket_value / max(1, sum(x["quantity"] for x in basket)), 2),
            "stockpile_days": 1.0,
            "hoarding_propensity": 0.0,
            "archetype": "synthetic_household",
            "cluster_id": -1,
            "has_crisis_observation": False,
        })
    return profiles


def build_greece_dairy_synthetic_config(
    template_count: int = SYNTHETIC_TEMPLATE_COUNT,
    pool_size: int = SYNTHETIC_SIMULATION_POOL,
) -> dict:
    """Return a deterministic synthetic ABM configuration for local prototyping."""
    products = _products()
    profiles = _synthetic_profiles(products, int(template_count))
    categories = Counter(product["category"] for product in products)
    return {
        "products": products,
        "population": profiles,
        "population_target_size": int(pool_size),
        "stats": {
            "population_pipeline_version": 9,
            "case_study_pipeline_version": GREECE_DAIRY_SYNTHETIC_VERSION,
            "case_study": "Greece — Dairy Supply Chain",
            "data_status": "fully_synthetic_no_empirical_data",
            "n_real": 0,
            "n_synthetic_templates": len(profiles),
            "pool_size": int(pool_size),
            "empirical_sampling_units": 0,
            "population_method": "seeded_resampling_of_declared_synthetic_household_templates",
            "catalogue_skus": len(products),
            "catalogue_categories": dict(categories),
            "dce_choice_validation": {
                "status": "not_estimated_no_greece_data",
                "n_participants": 0,
                "price_coefficient_estimable": False,
                "utility_scale_compatible_with_price": False,
                "beats_null_benchmark": False,
                "applicable_categories": [],
            },
            "substitution_choice_validation": {
                "status": "not_estimated_no_greece_data",
                "candidate_price_gate_supported": False,
                "supported_ranking_categories": [],
                "supported_transition_categories": [],
                "operational_fallback": "seeded_uniform_affordable_same_category",
            },
            "archetype_stability": {
                "status": "not_applicable_synthetic_templates",
                "archetypes_supported": False,
            },
            "synthetic_assumptions": {
                "seed": 20260827,
                "templates": len(profiles),
                "simulated_pool": int(pool_size),
                "catalogue_prices": "illustrative EUR per package; not observed retail prices",
                "basket_construction": "3-5 dairy lines per household with deterministic household-size variation",
                "price_sensitivity": "five declared values from 0.30 to 0.80; not estimated",
                "substitution": "35% propensity and affordable same-category allocation; not estimated",
                "panic_and_hoarding": "disabled in the default preset",
            },
            "scientific_cautions": [
                "No Greek participant, DCE, POS, inventory, delivery, or waste data are used.",
                "All catalogue prices, baskets, budgets, preferences, and substitution rates are synthetic assumptions.",
                "The case study is suitable for software and scenario-workflow testing only.",
                "Outputs must not be interpreted as estimates, predictions, forecasts, or policy evidence for Greece.",
                "Replace the synthetic layer and validate the model when Greek GROCERYsim data become available.",
            ],
        },
    }

