"""Synthetic development configuration for the Greece fish case study.

No Greek GROCERYsim fish observations are currently available.  This module
creates deterministic, auditable demonstration inputs so the interface and ABM
workflow can be tested without presenting synthetic values as empirical evidence.
"""

from __future__ import annotations

import random
from collections import Counter


GREECE_FISH_SYNTHETIC_VERSION = 1
SYNTHETIC_TEMPLATE_COUNT = 240
SYNTHETIC_SIMULATION_POOL = 1200


_PRODUCT_SPECS = (
    # id, name, category, price EUR/unit, domestic, shelf life in days
    ("gr-fish-seabream", "Greek farmed sea bream — 500 g", "Fresh farmed fish", 6.80, True, 6),
    ("gr-fish-seabass", "Greek farmed sea bass — 500 g", "Fresh farmed fish", 7.40, True, 6),
    ("gr-fish-trout", "Greek freshwater trout — 500 g", "Fresh farmed fish", 6.20, True, 5),
    ("gr-fish-sardine", "Fresh sardines — 500 g", "Small pelagic fish", 3.20, True, 4),
    ("gr-fish-anchovy", "Fresh anchovies — 500 g", "Small pelagic fish", 3.60, True, 4),
    ("gr-fish-mackerel", "Fresh mackerel — 500 g", "Small pelagic fish", 4.30, True, 5),
    ("gr-fish-hake-frozen", "Frozen hake fillets — 500 g", "Frozen fish", 5.90, False, 180),
    ("gr-fish-cod-frozen", "Frozen cod fillets — 500 g", "Frozen fish", 7.20, False, 180),
    ("gr-fish-salmon-frozen", "Frozen salmon portions — 400 g", "Frozen fish", 8.90, False, 150),
    ("gr-fish-tuna-can", "Canned tuna in water — 160 g", "Canned fish", 2.35, False, 730),
    ("gr-fish-sardine-can", "Canned sardines in olive oil — 120 g", "Canned fish", 2.10, True, 730),
    ("gr-fish-mackerel-can", "Canned mackerel — 160 g", "Canned fish", 2.25, False, 730),
)


def _products() -> list[dict]:
    return [
        {
            "id": product_id,
            "name": name,
            "category": category,
            "price": price,
            "origin": "Greece" if domestic else "Imported",
            "is_domestic": domestic,
            "is_bio": False,
            "is_plant_based": False,
            "fat_content": 0.0,
            "shelf_life_days": shelf_life,
        }
        for product_id, name, category, price, domestic, shelf_life in _PRODUCT_SPECS
    ]


def _basket_item(product: dict, quantity: int) -> dict:
    return {
        "product_id": product["id"],
        "product_name": product["name"],
        "category": product["category"],
        "quantity": int(quantity),
        "price": float(product["price"]),
        "fat_content": 0.0,
        "is_bio": False,
        "is_plant_based": False,
    }


def _synthetic_profiles(products: list[dict], count: int) -> list[dict]:
    rng = random.Random(20260828)
    by_category: dict[str, list[dict]] = {}
    for product in products:
        by_category.setdefault(product["category"], []).append(product)

    income_points = (900.0, 1400.0, 2100.0, 3000.0, 4200.0)
    price_sensitivities = (0.28, 0.40, 0.52, 0.65, 0.78)
    profiles = []
    for index in range(count):
        household_size = 1 + (index % 4)
        primary_group = (
            "Small pelagic fish" if index % 4 in (0, 1)
            else "Fresh farmed fish"
        )
        primary = rng.choice(by_category[primary_group])
        frozen = rng.choice(by_category["Frozen fish"])
        canned = rng.choice(by_category["Canned fish"])
        basket = [
            _basket_item(primary, 1 + int(household_size >= 4)),
            _basket_item(frozen, 1),
            _basket_item(canned, 1 + int(household_size >= 3)),
        ]
        if index % 3 == 0:
            secondary_group = (
                "Fresh farmed fish" if primary_group == "Small pelagic fish"
                else "Small pelagic fish"
            )
            basket.append(_basket_item(rng.choice(by_category[secondary_group]), 1))
        basket_value = sum(row["price"] * row["quantity"] for row in basket)
        income = income_points[index % len(income_points)]
        profiles.append({
            "source_id": f"synthetic-gr-fish-{index + 1:03d}",
            "empirical_source_id": f"synthetic-gr-fish-{index + 1:03d}",
            "is_real": False,
            "is_synthetic": True,
            "age": 20 + (index * 11) % 55,
            "gender": ("Woman", "Man", "Other / not specified")[index % 3],
            "household_size": household_size,
            "income_midpoint": income,
            "baseline_basket": basket,
            "crisis_basket": [dict(row) for row in basket],
            "budget": round(max(basket_value * 1.30, 16.0), 2),
            "crisis_budget": round(max(basket_value * 1.42, 18.0), 2),
            "budget_utilization_propensity": 0.92,
            "price_sensitivity": price_sensitivities[index % len(price_sensitivities)],
            "revealed_preference_margin": 0.08,
            "substitution_rate": 0.40,
            "sub_tolerance": 0.50,
            "finnish_preference": 0.68,  # compatibility alias: Greek-origin preference
            "organic_preference": 0.0,
            "preferred_fat": 0.0,
            "reference_price": round(basket_value / max(1, sum(x["quantity"] for x in basket)), 2),
            "stockpile_days": 1.0,
            "hoarding_propensity": 0.0,
            "archetype": "synthetic_household",
            "cluster_id": -1,
            "has_crisis_observation": False,
        })
    return profiles


def build_greece_fish_synthetic_config(
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
            "case_study_pipeline_version": GREECE_FISH_SYNTHETIC_VERSION,
            "case_study": "Greece — Fish Supply Chain",
            "data_status": "fully_synthetic_no_empirical_data",
            "n_real": 0,
            "n_synthetic_templates": len(profiles),
            "pool_size": int(pool_size),
            "empirical_sampling_units": 0,
            "population_method": "seeded_resampling_of_declared_synthetic_household_templates",
            "catalogue_skus": len(products),
            "catalogue_categories": dict(categories),
            "dce_choice_validation": {
                "status": "not_estimated_no_greece_fish_data",
                "n_participants": 0,
                "price_coefficient_estimable": False,
                "utility_scale_compatible_with_price": False,
                "beats_null_benchmark": False,
                "applicable_categories": [],
            },
            "substitution_choice_validation": {
                "status": "not_estimated_no_greece_fish_data",
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
                "seed": 20260828,
                "templates": len(profiles),
                "simulated_pool": int(pool_size),
                "catalogue_prices": "illustrative EUR per sales unit; not observed Greek retail prices",
                "basket_construction": "3-4 fish lines spanning fresh, frozen and canned products",
                "price_sensitivity": "five declared values from 0.28 to 0.78; not estimated",
                "substitution": "40% propensity and affordable same-category allocation; not estimated",
                "panic_and_hoarding": "disabled in the default preset",
                "shelf_life": "4-6 days fresh, 150-180 days frozen and 730 days canned",
            },
            "scientific_cautions": [
                "No Greek participant, DCE, POS, landing, aquaculture, inventory, delivery, cold-chain, or waste data are used.",
                "All catalogue prices, baskets, budgets, preferences, shelf lives, and substitution rates are synthetic assumptions.",
                "The case study is suitable for software and scenario-workflow testing only.",
                "Outputs must not be interpreted as estimates, predictions, forecasts, or policy evidence for Greece.",
                "Replace the synthetic layer and validate product choice, cold-chain loss, landings, and aquaculture dynamics when data become available.",
            ],
        },
    }

