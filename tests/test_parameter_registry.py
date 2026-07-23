import unittest

from parameter_registry import (
    build_parameter_registry,
    parameter_registry_summary,
    validate_parameter_registry,
)


class ParameterRegistryTests(unittest.TestCase):
    def test_registry_is_machine_valid_and_ids_are_unique(self):
        rows = build_parameter_registry()
        self.assertEqual(validate_parameter_registry(rows), [])
        ids = [row["parameter_id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_current_data_does_not_claim_dce_price_identification(self):
        row = next(
            row for row in build_parameter_registry()
            if row["parameter_id"] == "dce.price_coefficient"
        )
        self.assertFalse(row["identifiable_from_current_data"])
        self.assertIsNone(row["current_value"])
        self.assertIn("absent", row["source"])

    def test_validated_clean_dce_price_is_registered_as_identified(self):
        stats = {"dce_choice_validation": {
            "status": "ok",
            "beats_null_benchmark": True,
            "model_converged": True,
            "price_coefficient_estimable": True,
            "price_coefficient": -0.14,
            "price_source": "cleaned_dce_csv_recorded_prices",
            "validation_log_loss": 0.91,
            "null_model_log_loss": 0.98,
        }}

        row = next(
            row for row in build_parameter_registry(stats)
            if row["parameter_id"] == "dce.price_coefficient"
        )

        self.assertTrue(row["identifiable_from_current_data"])
        self.assertEqual(row["current_value"], -0.14)
        self.assertEqual(row["evidence_class"], "heldout_calibrated")

    def test_runtime_scenario_values_are_reported(self):
        rows = build_parameter_registry(
            runtime_params={
                "base_con": 321,
                "panic": 0.8,
                "lead": 6,
                "panic_growth_rate": 0.7,
            }
        )
        values = {row["parameter_id"]: row["current_value"] for row in rows}
        self.assertEqual(values["demand.daily_visitors"], 321)
        self.assertEqual(values["crisis.panic_sensitivity"], 0.8)
        self.assertEqual(values["inventory.lead_time"], 6)
        self.assertEqual(values["crisis.panic_growth_rate"], 0.7)

    def test_calibration_status_changes_provenance_and_records_rejection(self):
        stats = {
            "behavioral_calibration": {
                "status": "ok",
                "price_response_model_retained": False,
                "substitution_model_retained": True,
                "hoarding_model_retained": False,
                "revealed_preference_margin": 0.11,
            },
            "dce_choice_validation": {
                "status": "ok",
                "beats_majority_benchmark": True,
                "origin_coefficient": 0.6,
                "organic_coefficient": 0.15,
                "fat_linear_coefficient": 0.2,
                "fat_quadratic_coefficient": -0.1,
            },
        }
        rows = {row["parameter_id"]: row for row in build_parameter_registry(stats)}
        self.assertEqual(rows["dce.origin_weight"]["evidence_class"], "heldout_calibrated")
        self.assertEqual(rows["dce.origin_weight"]["current_value"], 0.6)
        self.assertIn("rejected", rows["behaviour.price_response"]["validation"])
        self.assertEqual(rows["behaviour.revealed_margin"]["current_value"], 0.11)

    def test_readiness_stays_false_while_critical_assumptions_remain(self):
        summary = parameter_registry_summary(build_parameter_registry())
        self.assertFalse(summary["policy_grade_ready"])
        self.assertGreater(summary["n_unresolved_high_priority"], 0)

    def test_behavioral_extensions_require_explicit_runtime_opt_in(self):
        empirical = {
            row["parameter_id"]: row
            for row in build_parameter_registry(runtime_params={})
        }
        exploratory = {
            row["parameter_id"]: row
            for row in build_parameter_registry(
                runtime_params={"exploratory_behaviour": True}
            )
        }

        self.assertEqual(
            empirical["behaviour.evidence_mode"]["current_value"],
            "empirical only",
        )
        self.assertEqual(empirical["behaviour.loss_aversion"]["current_value"], "disabled")
        self.assertEqual(
            exploratory["behaviour.evidence_mode"]["current_value"],
            "exploratory extensions",
        )
        self.assertEqual(exploratory["behaviour.loss_aversion"]["current_value"], 2.25)


if __name__ == "__main__":
    unittest.main()
