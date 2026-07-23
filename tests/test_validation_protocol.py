import unittest
from pathlib import Path

import pandas as pd

from validation_protocol import (
    daily_validation_observables,
    evaluate_baseline_reproduction,
    evaluate_phase2_reproduction,
    evaluate_targets,
    validate_target_definitions,
    validation_summary,
    validation_target_template,
)


def _valid_target(**overrides):
    row = validation_target_template().iloc[0].to_dict()
    row.update(
        {
            "target_id": "waste_external_01",
            "metric": "Waste",
            "label": "Mean daily waste",
            "source_name": "Independent retail panel",
            "source_reference": "doi:10.example/archive",
            "source_period": "2024-01 to 2024-12",
            "source_population": "Finnish supermarkets",
            "registration_reference": "https://osf.example/registered-2025-01-01",
            "notes": "Locked before model evaluation.",
            "day_start": 1,
            "day_end": 3,
            "lower": 8.0,
            "upper": 12.0,
        }
    )
    row.update(overrides)
    return row


class ValidationProtocolTests(unittest.TestCase):
    def test_phase2_audit_does_not_claim_rejected_individual_model(self):
        targets = {
            "status": "ok",
            "metrics": {
                "quantity_retention": {
                    "mean": 0.7,
                    "training_mean": 0.7,
                    "absolute_tolerance": 0.1,
                    "individual_model_retained": False,
                }
            },
            "caution": "Internal holdout only.",
        }
        simulation = pd.DataFrame({
            "Run": [0, 0],
            "source_id": ["a", "b"],
            "model_quantity_retention": [0.7, 0.7],
            "observed_quantity_retention": [0.2, 1.2],
        })

        audit = evaluate_phase2_reproduction(targets, simulation)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(
            audit["checks"][0]["individual_skill_gate"], "not_claimed"
        )

    def test_phase2_retained_model_must_beat_naive_individual_prediction(self):
        targets = {
            "status": "ok",
            "metrics": {
                "substitution_rate": {
                    "mean": 0.5,
                    "training_mean": 0.5,
                    "absolute_tolerance": 0.1,
                    "individual_model_retained": True,
                }
            },
        }
        simulation = pd.DataFrame({
            "Run": [0, 0],
            "source_id": ["a", "b"],
            "model_substitution_rate": [0.5, 0.5],
            "observed_substitution_rate": [0.0, 1.0],
        })

        audit = evaluate_phase2_reproduction(targets, simulation)

        self.assertEqual(audit["status"], "fail")
        self.assertEqual(
            audit["checks"][0]["individual_skill_gate"], "fail"
        )

    def test_baseline_reproduction_is_scoped_internal_evidence(self):
        targets = {
            "status": "ok",
            "mean_linked_basket_units": 2.0,
            "mean_linked_basket_value": 4.0,
            "organic_unit_share": 0.5,
            "domestic_unit_share": 0.75,
            "category_unit_shares": {"Milk": 1.0},
        }
        simulation = pd.DataFrame({
            "Day": list(range(1, 7)),
            "Consumers": [10] * 6,
            "RequestedDemandUnits": [20] * 6,
            "Sales": [20] * 6,
            "NominalRevenue": [40] * 6,
            "DomesticSales": [15] * 6,
            "ImportSales": [5] * 6,
            "OrganicSalesUnits": [10] * 6,
            "CategorySalesUnits": [{"Milk": 20}] * 6,
            "ConsumptionFulfillmentRate": [1.0] * 6,
            "VisitorCapacityCapped": [0] * 6,
            "ExpectedVisitIntervalDays": [2.0] * 6,
        })

        audit = evaluate_baseline_reproduction(targets, simulation)

        self.assertEqual(audit["status"], "pass")
        self.assertTrue(all(
            row["evidence_tier"].startswith("internal")
            for row in audit["checks"]
        ))
        self.assertIn("not identified", audit["claim"])

    def test_baseline_reproduction_detects_temporal_demand_inflation(self):
        targets = {
            "status": "ok",
            "mean_linked_basket_units": 2.0,
            "mean_linked_basket_value": 4.0,
            "organic_unit_share": 0.5,
            "domestic_unit_share": 0.75,
            "category_unit_shares": {"Milk": 1.0},
        }
        simulation = pd.DataFrame({
            "Day": [1, 2, 3, 4], "Consumers": [10] * 4,
            "RequestedDemandUnits": [40] * 4, "Sales": [40] * 4,
            "NominalRevenue": [80] * 4, "DomesticSales": [30] * 4,
            "ImportSales": [10] * 4, "OrganicSalesUnits": [20] * 4,
            "CategorySalesUnits": [{"Milk": 40}] * 4,
            "ConsumptionFulfillmentRate": [1.0] * 4,
            "VisitorCapacityCapped": [0] * 4,
            "ExpectedVisitIntervalDays": [1.0] * 4,
        })

        audit = evaluate_baseline_reproduction(targets, simulation)
        generated = next(
            row for row in audit["checks"]
            if row["metric"] == "requested_units_per_visit"
        )

        self.assertEqual(audit["status"], "fail")
        self.assertEqual(generated["status"], "fail")
    def test_daily_validation_observables_use_declared_denominators(self):
        values = daily_validation_observables(
            sales_units=80,
            nominal_revenue=240,
            consumers=20,
            waste_units=20,
            sku_stockout_flags=[False, True, True, False],
        )
        self.assertEqual(values["MeanDairyBasketUnits"], 4.0)
        self.assertEqual(values["MeanDairyBasketValue"], 12.0)
        self.assertEqual(values["DairyWasteRate"], 0.2)
        self.assertEqual(values["StockoutSkuDayRate"], 0.5)

    def test_external_target_requires_independence_and_registration(self):
        targets = pd.DataFrame([_valid_target(
            independent_of_calibration=False,
            preregistered=False,
            registration_reference="",
        )])
        errors = validate_target_definitions(targets)
        self.assertTrue(any("independent of calibration" in error for error in errors))
        self.assertTrue(any("preregistered" in error for error in errors))
        self.assertTrue(any("registration_reference" in error for error in errors))

    def test_template_placeholders_cannot_be_evidence(self):
        errors = validate_target_definitions(validation_target_template())
        self.assertTrue(any("template/example" in error for error in errors))

    def test_interval_bounds_are_inclusive(self):
        targets = pd.DataFrame([_valid_target(lower=10.0, upper=10.0)])
        simulation = pd.DataFrame({
            "Day": [1, 2, 3] * 3,
            "Run": [0] * 3 + [1] * 3 + [2] * 3,
            "Scenario": ["Baseline"] * 9,
            "Waste": [9.0, 10.0, 11.0] * 3,
        })
        evaluated = evaluate_targets(targets, simulation)
        self.assertEqual(evaluated.loc[0, "observed"], 10.0)
        self.assertEqual(evaluated.loc[0, "status"], "pass")

    def test_missing_metric_blocks_external_claim(self):
        targets = pd.DataFrame([_valid_target(metric="UnknownMetric")])
        simulation = pd.DataFrame({"Day": [1], "Scenario": ["Baseline"], "Waste": [10.0]})
        summary = validation_summary(evaluate_targets(targets, simulation))
        self.assertEqual(summary["claim_status"], "external_incomplete")

    def test_calibration_holdout_cannot_imply_external_validity(self):
        target = _valid_target(
            evidence_tier="calibration_holdout",
            independent_of_calibration=False,
            preregistered=False,
            registration_reference="",
        )
        targets = pd.DataFrame([target])
        simulation = pd.DataFrame({
            "Day": [1, 2, 3], "Scenario": ["Baseline"] * 3, "Waste": [10.0] * 3,
        })
        summary = validation_summary(evaluate_targets(targets, simulation))
        self.assertEqual(summary["claim_status"], "no_external_targets")

    def test_all_external_targets_pass_is_scoped_claim(self):
        targets = pd.DataFrame([_valid_target()])
        simulation = pd.DataFrame({
            "Day": [1, 2, 3] * 3,
            "Run": [0] * 3 + [1] * 3 + [2] * 3,
            "Scenario": ["Baseline"] * 9,
            "Waste": [9.0, 10.0, 11.0] * 3,
        })
        summary = validation_summary(evaluate_targets(targets, simulation))
        self.assertEqual(summary["claim_status"], "external_targets_met")
        self.assertIn("only for these targets", summary["claim"])

    def test_single_stochastic_run_cannot_support_external_claim(self):
        targets = pd.DataFrame([_valid_target()])
        simulation = pd.DataFrame({
            "Day": [1, 2, 3], "Scenario": ["Baseline"] * 3, "Waste": [9.0, 10.0, 11.0],
        })
        evaluated = evaluate_targets(targets, simulation)
        self.assertEqual(evaluated.loc[0, "status"], "not_evaluated")
        self.assertIn("replicates", evaluated.loc[0, "reason"])

    def test_external_pass_requires_replicate_interval_inside_bounds(self):
        targets = pd.DataFrame([_valid_target(lower=9.5, upper=10.5)])
        simulation = pd.DataFrame({
            "Day": [1, 2, 3] * 3,
            "Run": [0] * 3 + [1] * 3 + [2] * 3,
            "Scenario": ["Baseline"] * 9,
            "Waste": [8.0] * 3 + [10.0] * 3 + [12.0] * 3,
        })
        evaluated = evaluate_targets(targets, simulation)
        self.assertEqual(evaluated.loc[0, "observed"], 10.0)
        self.assertEqual(evaluated.loc[0, "status"], "fail")
        self.assertIn("95% replicate interval", evaluated.loc[0, "reason"])

    def test_bundled_draft_is_not_misrepresented_as_registered_evidence(self):
        root = Path(__file__).resolve().parents[1]
        draft = pd.read_csv(root / "validation" / "validation_plan_DRAFT.csv")
        external = draft[draft["evidence_tier"] == "external_independent"]
        self.assertTrue(external["preregistered"].eq(False).all())
        self.assertTrue(external["lower"].isna().all())
        self.assertTrue(external["upper"].isna().all())
        self.assertTrue(validate_target_definitions(draft))

    def test_evidence_catalogue_records_access_and_blockers(self):
        root = Path(__file__).resolve().parents[1]
        catalogue = pd.read_csv(root / "validation" / "evidence_catalogue.csv")
        self.assertIn("FIN_LOCARD", set(catalogue["source_id"]))
        self.assertIn("FIN_RETAIL_OPS", set(catalogue["source_id"]))
        self.assertFalse(catalogue["blocking_issue"].isna().any())


if __name__ == "__main__":
    unittest.main()
