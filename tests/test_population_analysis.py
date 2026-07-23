import unittest

import numpy as np

from data_processor import (
    archetype_stability_diagnostics,
    assign_archetypes,
    bootstrap_population,
    parse_questionnaire,
    questionnaire_reliability,
)
from model import SupermarketModel


def _ratings(values):
    return [{"value": value} for value in values]


def _profile(source_id, level=0.2):
    return {
        "source_id": source_id,
        "is_real": True,
        "age": 30,
        "household_size": 2,
        "children": 0,
        "q_price": level,
        "q_health": level,
        "q_environment": level,
        "q_animal_welfare": level,
        "q_sensory_habit": level,
        "organic_preference": level,
        "finnish_preference": level,
        "preferred_fat": 3.8 * level,
        "price_sensitivity": level,
        "baseline_basket": [{
            "product_id": "sku-1", "product_name": "Milk",
            "category": "Milk", "quantity": 1, "price": 2.0,
        }],
        "crisis_basket": [],
        "budget": 10.0,
        "crisis_budget": 10.0,
    }


def _product():
    return {
        "id": "sku-1", "name": "Milk", "category": "Milk", "price": 2.0,
        "origin": "Suomi", "is_bio": False, "fat_content": 1.5,
        "is_plant_based": False, "shelf_life_days": 10,
    }


class PopulationAnalysisTests(unittest.TestCase):
    def test_missing_questionnaire_items_are_not_silently_neutral_imputed(self):
        scores = parse_questionnaire(_ratings([1.0, 5.0]))
        self.assertTrue(scores["q_price_fallback"])
        self.assertEqual(scores["q_price_observed_items"], 1)
        self.assertEqual(scores["q_price"], 0.5)

    def test_reliability_audit_reports_declared_constructs_and_missingness(self):
        participants = {}
        for participant in range(30):
            base = 1.0 + (participant % 5)
            values = [base] * 21
            participants[str(participant)] = {"questionnaireRatings": _ratings(values)}
        participants["missing"] = {"questionnaireRatings": _ratings([3.0] * 5)}
        audit = questionnaire_reliability({"participants": participants})
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(len(audit["constructs"]), 5)
        self.assertTrue(any(row["missing_cell_rate"] > 0 for row in audit["constructs"]))

    def test_clear_two_group_structure_passes_stability_gate(self):
        rng = np.random.default_rng(7)
        profiles = []
        for index in range(80):
            centre = 0.15 if index < 40 else 0.85
            profiles.append(_profile(str(index), float(np.clip(
                centre + rng.normal(0, 0.025), 0, 1
            ))))
        audit = archetype_stability_diagnostics(
            profiles, selected_k=2, random_state=4, n_bootstrap=30,
        )
        self.assertEqual(audit["recommended_k"], 2)
        self.assertTrue(audit["archetypes_supported"])

    def test_failed_gate_keeps_clusters_exploratory_only(self):
        profiles = [_profile(str(index), index / 20) for index in range(20)]
        assigned = assign_archetypes(profiles, n_clusters=2, operational=False)
        self.assertTrue(all(p["archetype"] == "continuous_profile" for p in assigned))
        self.assertTrue(all(p["exploratory_archetype"] for p in assigned))

    def test_resampling_preserves_complete_observed_profiles(self):
        profiles = [_profile("a", 0.2), _profile("b", 0.8)]
        population = bootstrap_population(profiles, 50, jitter_seed=3)
        self.assertEqual(len(population), 50)
        self.assertTrue(all(p["empirical_source_id"] in {"a", "b"} for p in population))
        for resampled in population:
            original = next(p for p in profiles if p["source_id"] == resampled["empirical_source_id"])
            self.assertEqual(resampled["age"], original["age"])
            self.assertEqual(resampled["q_price"], original["q_price"])
            self.assertEqual(resampled["baseline_basket"], original["baseline_basket"])

    def test_model_seed_controls_participant_resample(self):
        profiles = [_profile(str(index), index / 10) for index in range(10)]
        config = {
            "products": [_product()],
            "population": profiles,
            "population_target_size": 40,
        }
        first = SupermarketModel(config_data=config, base_consumers=2, fixed_seed=10)
        repeat = SupermarketModel(config_data=config, base_consumers=2, fixed_seed=10)
        other = SupermarketModel(config_data=config, base_consumers=2, fixed_seed=11)
        first_ids = [p["empirical_source_id"] for p in first.population_pool]
        repeat_ids = [p["empirical_source_id"] for p in repeat.population_pool]
        other_ids = [p["empirical_source_id"] for p in other.population_pool]
        self.assertEqual(first_ids, repeat_ids)
        self.assertNotEqual(first_ids, other_ids)
        self.assertEqual(first.empirical_sampling_units, 10)


if __name__ == "__main__":
    unittest.main()
