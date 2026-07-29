import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from data_processor import (
    _clean_basket,
    calibrate_dce_choice_model,
    calibrate_behavioral_profiles,
    canonicalize_products,
    compute_price_elasticity,
    summarize_baseline_observations,
    substitution_choice_diagnostics,
)
from model import ConsumerAgent, SupermarketModel, is_low_income_access_stressed


def product(*, product_id="sku-1", name="Milk", price=2.0, organic=False):
    return {
        "id": product_id,
        "name": name,
        "category": "Milk",
        "price": price,
        "origin": "Suomi",
        "is_bio": organic,
        "fat_content": 1.5,
        "is_plant_based": False,
        "shelf_life_days": 10,
    }


def profile(*, quantity=1, budget=20.0, product_id="sku-1", name="Milk"):
    basket = [{
        "product_id": product_id,
        "product_name": name,
        "category": "Milk",
        "quantity": quantity,
    }]
    return {
        "baseline_basket": basket,
        "crisis_basket": basket,
        "budget": budget,
        "crisis_budget": budget,
        "price_sensitivity": 0.0,
        "finnish_preference": 1.0,
        "preferred_fat": 1.5,
        "stockpile_days": 1.0,
    }


class AccessStressDiagnosticTests(unittest.TestCase):
    def test_low_income_stockout_is_stress_even_without_budget_exhaustion(self):
        shopper = SimpleNamespace(
            income_midpoint=1200.0,
            items_wanted=10,
            items_purchased=2,
            budget_exhausted=False,
        )
        self.assertTrue(is_low_income_access_stressed(shopper))

    def test_no_requested_demand_is_not_automatically_stress(self):
        shopper = SimpleNamespace(
            income_midpoint=1200.0,
            items_wanted=0,
            items_purchased=0,
            budget_exhausted=False,
        )
        self.assertFalse(is_low_income_access_stressed(shopper))

    def test_high_income_shopper_is_outside_low_income_diagnostic(self):
        shopper = SimpleNamespace(
            income_midpoint=3000.0,
            items_wanted=10,
            items_purchased=0,
            budget_exhausted=True,
        )
        self.assertFalse(is_low_income_access_stressed(shopper))


class CatalogueCompletenessTests(unittest.TestCase):
    def test_bundled_catalogues_have_analysis_ready_category_and_origin(self):
        project_root = Path(__file__).resolve().parents[1]
        for relative_path in ("data/master_products.json", "data/product_catalogue.json"):
            payload = json.loads((project_root / relative_path).read_text(encoding="utf-8"))
            products = payload["products"]
            for product_row in products:
                with self.subTest(catalogue=relative_path, product=product_row.get("id")):
                    self.assertTrue(str(product_row.get("category", "")).strip())
                    self.assertTrue(str(product_row.get("origin", "")).strip())


class CatalogueIdentityTests(unittest.TestCase):
    def test_repeated_basket_rows_are_consolidated_by_canonical_sku(self):
        raw = [
            {"productName": "Milk", "quantity": 1, "price": 2.0, "category": "Milk"},
            {"productName": "Milk", "quantity": 2, "price": 2.3, "category": "Milk"},
        ]

        clean = _clean_basket(raw, {"Milk": "sku-1"})

        self.assertEqual(len(clean), 1)
        self.assertEqual(clean[0]["quantity"], 3)
        self.assertAlmostEqual(clean[0]["price"], 2.2)

    def test_baseline_target_treats_basket_as_household_outcome(self):
        p = profile(quantity=3, budget=20.0)
        p["household_size"] = 5
        p["baseline_basket"][0].update({"price": 2.0, "is_bio": False})

        targets = summarize_baseline_observations([p], [product()])

        self.assertEqual(targets["mean_linked_basket_units"], 3.0)
        self.assertEqual(targets["mean_linked_basket_value"], 6.0)
        self.assertIn("not_rescaled", targets["household_size_treatment"])
        self.assertFalse(targets["visit_interval_identified"])


class DCEPriceModelTests(unittest.TestCase):
    def test_clean_dce_prices_identify_heldout_price_effect(self):
        participants = {}
        rows = []
        for participant in range(30):
            participant_id = f"dce-{participant}"
            participants[participant_id] = {
                "choiceExperiment1_Results": [{"choiceMade": "left"}]
            }
            for choice_set in range(12):
                choice_id = f"{participant_id}-{choice_set}"
                cheap_is_first = choice_set % 2 == 0
                chosen_alternative = (
                    3 if choice_set == 0 else
                    (1 if cheap_is_first else 2) if choice_set != 1 else
                    (2 if cheap_is_first else 1)
                )
                for alternative in (1, 2, 3):
                    optout = alternative == 3
                    price = 0.0 if optout else (
                        1.0 if (alternative == 1) == cheap_is_first else 2.5
                    )
                    rows.append({
                        "respondent_id": participant_id,
                        "choice_set": str(choice_set + 1),
                        "choice_id": choice_id,
                        "alternative": str(alternative),
                        "chosen": str(int(alternative == chosen_alternative)),
                        "origin": str(int(alternative == 1 and not optout)),
                        "organic": "0",
                        "fat": "0" if optout else "1.5",
                        "price": str(price),
                        "price_inferred": "0",
                        "optout_asc": str(int(optout)),
                        "version": "2",
                        "exclude_dce": "FALSE",
                    })

        diagnostics = calibrate_dce_choice_model(
            {"participants": participants}, dce_rows=rows
        )

        self.assertTrue(diagnostics["model_converged"])
        self.assertTrue(diagnostics["price_coefficient_estimable"])
        self.assertTrue(diagnostics["beats_null_benchmark"])
        self.assertLess(diagnostics["price_coefficient"], 0)

    def test_identical_scene_placements_collapse_to_one_sku(self):
        rows = [product(product_id="scene-a"), product(product_id="scene-b")]
        canonical, name_to_id = canonicalize_products(rows)

        self.assertEqual(len(canonical), 1)
        self.assertEqual(name_to_id, {"Milk": "scene-a"})
        self.assertEqual(canonical[0]["source_ids"], ["scene-a", "scene-b"])

    def test_conflicting_duplicate_names_fail_loudly(self):
        rows = [product(product_id="a", price=2.0), product(product_id="b", price=3.0)]
        with self.assertRaisesRegex(ValueError, "Ambiguous product name"):
            canonicalize_products(rows)

    def test_same_product_is_not_misclassified_as_category_substitution(self):
        r1 = [
            {"productName": "Milk A", "category": "Milk", "price": 1, "quantity": 1},
            {"productName": "Milk B", "category": "Milk", "price": 1, "quantity": 1},
        ]
        r2 = [{"productName": "Milk A", "category": "Milk", "price": 1.25, "quantity": 1}]

        result = compute_price_elasticity(r1, r2, 2.0, 2.0)

        self.assertEqual(result["substitution_rate"], 0.0)

    def test_substitution_target_retains_numerator_and_both_denominators(self):
        r1 = [
            {"productName": "Milk A", "category": "Milk", "price": 1, "quantity": 1},
            {"productName": "Bread A", "category": "Bread", "price": 1, "quantity": 1},
        ]
        r2 = [
            {"productName": "Milk B", "category": "Milk", "price": 1.2, "quantity": 1},
        ]

        result = compute_price_elasticity(r1, r2, 2.0, 2.0)

        self.assertEqual(result["substitution_lines"], 1)
        self.assertEqual(result["phase2_choice_lines"], 1)
        self.assertEqual(result["baseline_choice_lines"], 2)
        self.assertEqual(result["substitution_rate"], 1.0)

    def test_sparse_replacement_events_do_not_enable_predictive_ranking(self):
        products = [
            product(product_id="a", name="Milk A", price=2.0),
            product(product_id="b", name="Milk B", price=1.5),
        ]
        observed = {
            **profile(product_id="a", name="Milk A"),
            "has_crisis_observation": True,
            "price_sensitivity": 0.5,
            "revealed_preference_margin": 0.1,
        }
        observed["baseline_basket"][0]["price"] = 2.0
        observed["crisis_basket"] = [{
            "product_id": "b", "product_name": "Milk B", "category": "Milk",
            "quantity": 1, "price": 1.5,
        }]

        audit = substitution_choice_diagnostics([observed], products)

        self.assertEqual(audit["n_unambiguous_events"], 1)
        self.assertEqual(audit["supported_ranking_categories"], [])
        self.assertEqual(
            audit["operational_fallback"],
            "dce_mnl_for_milk_then_validated_phase_transition_target_shares_"
            "else_seeded_uniform_affordable_same_category",
        )


class ModelInvariantTests(unittest.TestCase):
    def make_model(self, *, quantity=1, budget=20.0, start_month=1):
        cfg = {
            "products": [product()],
            "population": [profile(quantity=quantity, budget=budget)],
        }
        return SupermarketModel(
            config_data=cfg,
            base_consumers=1,
            start_month=start_month,
            fixed_seed=7,
        )

    def test_budget_is_a_hard_constraint(self):
        model = self.make_model(quantity=10, budget=3.0)
        sku = model.get_product_by_id("sku-1")
        sku.shelf_batches = [{"qty": 10, "age": 0}]
        consumer = ConsumerAgent("test-consumer", model, model.population_pool[0])

        spent, units = consumer._execute_purchase(sku, 10, 3.0)

        self.assertLessEqual(spent, 3.0)
        self.assertEqual((spent, units), (2.0, 1))
        self.assertTrue(consumer.budget_exhausted)
        self.assertEqual(sku.stock_shelf, 9)

    def test_product_uses_calibrated_storage_capacity(self):
        p = product()
        model = SupermarketModel(
            config_data={"products": [p], "population": [profile()]},
            base_consumers=1,
            fixed_seed=7,
        )
        sku = model.get_product_by_id("sku-1")
        calibrated = model.store_calibration["sku-1"]

        self.assertEqual(
            sku.max_storage_capacity, calibrated["max_storage_capacity"]
        )
        self.assertEqual(
            sku.stock_storage, calibrated["initial_stock_storage"]
        )

    def test_initial_shelf_inventory_uses_staggered_age_cohorts(self):
        model = self.make_model()
        sku = model.get_product_by_id("sku-1")
        calibrated_initial = model.store_calibration["sku-1"]["initial_stock_shelf"]

        self.assertEqual(sum(batch["qty"] for batch in sku.shelf_batches), calibrated_initial)
        self.assertGreater(len({batch["age"] for batch in sku.shelf_batches}), 1)
        self.assertTrue(all(0 <= batch["age"] < sku.max_shelf_life for batch in sku.shelf_batches))

    def test_snapshot_excludes_stock_expired_before_shopping(self):
        model = self.make_model()
        sku = model.get_product_by_id("sku-1")
        sku.shelf_batches = [{"qty": 4, "age": sku.max_shelf_life - 1}]
        sku.stock_storage = 0
        model.current_day = 1

        sku.step()

        self.assertEqual(sku.daily_waste, 4)
        self.assertEqual(sku.stock_shelf, 0)
        self.assertEqual(sku.snap_shelf, 0)

    def test_unobserved_product_does_not_receive_large_generic_demand(self):
        observed = product(product_id="milk-observed", name="Observed Milk")
        unobserved = {
            **product(product_id="cream-unobserved", name="Unobserved Cream"),
            "category": "Cream",
            "shelf_life_days": 20,
        }
        observed_profile = profile(
            product_id="milk-observed", name="Observed Milk", quantity=1
        )
        model = SupermarketModel(
            config_data={
                "products": [observed, unobserved],
                "population": [observed_profile],
            },
            base_consumers=200,
            fixed_seed=7,
        )
        calibration = model.store_calibration["cream-unobserved"]

        self.assertEqual(calibration["estimated_daily_demand"], 1.0)
        self.assertEqual(calibration["max_shelf_capacity"], 10)
        self.assertEqual(
            calibration["demand_basis"], "minimum_floor_no_product_evidence"
        )

    def test_crisis_budget_is_a_maximum_not_a_spending_target(self):
        p = {
            **profile(quantity=10, budget=20.0),
            "crisis_budget": 10.0,
            "budget_utilization_propensity": 0.5,
            "reference_price": 2.0,
        }
        model = SupermarketModel(
            config_data={"products": [product()], "population": [p]},
            base_consumers=1,
            is_crisis_mode=True,
            scenario_start_day=1,
            inflation_pct=25.0,
            fixed_seed=7,
        )

        model.step()

        self.assertEqual(model.last_daily_agents[0].amount_spent, 5.0)
        self.assertLess(
            model.last_daily_agents[0].amount_spent,
            model.last_daily_agents[0].crisis_budget,
        )

    def test_product_counters_are_not_reset_after_consumers_shop(self):
        model = self.make_model(quantity=1)
        model.step()

        purchased = sum(c.items_purchased for c in model.last_daily_agents)
        recorded = sum(p.daily_sales for p in model.products)
        self.assertGreater(purchased, 0)
        self.assertEqual(recorded, purchased)
        self.assertEqual(model.daily_records[-1]["Sales"], purchased)

    def test_complete_visit_never_exceeds_active_budget(self):
        model = self.make_model(quantity=10, budget=3.0)
        model.step()

        consumer = model.last_daily_agents[0]
        self.assertLessEqual(consumer.amount_spent, consumer.budget)
        self.assertEqual(consumer.items_purchased, 1)

    def test_daily_inventory_ledger_conserves_units(self):
        model = self.make_model(quantity=2, budget=20.0)

        for _ in range(12):
            opening = sum(p.stock_shelf + p.stock_storage for p in model.products)
            old_log_size = len(model.truck.log)
            model.step()
            new_logs = model.truck.log[old_log_size:]
            inbound = sum(
                row.get("Quantity", 0) + row.get("Refused", 0)
                for row in new_logs if row.get("Action") == "Delivery"
            )
            closing = sum(p.stock_shelf + p.stock_storage for p in model.products)
            record = model.daily_records[-1]
            self.assertEqual(
                opening + inbound,
                closing + record["Sales"] + record["Waste"],
            )

    def test_partial_stock_is_sold_instead_of_rejecting_whole_request(self):
        model = self.make_model(quantity=3)
        sku = model.get_product_by_id("sku-1")
        sku.shelf_batches = [{"qty": 1, "age": 0}]
        sku.stock_storage = 0

        model.step()

        consumer = model.last_daily_agents[0]
        self.assertEqual(consumer.items_wanted, 3)
        self.assertEqual(consumer.items_purchased, 1)
        self.assertEqual(consumer.items_unmet, 2)

    def test_start_month_is_respected_and_rolls_forward(self):
        model = self.make_model(start_month=12)
        model.step()
        self.assertEqual(model.current_month, 12)
        for _ in range(30):
            model.step()
        self.assertEqual(model.current_month, 1)

    def test_reference_prices_persist_in_the_household_profile(self):
        cfg = {
            "products": [product()],
            "population": [profile()],
        }
        model = SupermarketModel(
            config_data=cfg, base_consumers=1, fixed_seed=7,
            enable_prospect_theory=True,
        )
        sku = model.get_product_by_id("sku-1")
        sku.current_price = 3.0
        consumer = ConsumerAgent("first", model, model.population_pool[0])
        consumer._execute_purchase(sku, 1, 10.0)

        learned = model.population_pool[0]["_ref_prices"]["sku-1"]
        next_visit = ConsumerAgent("second", model, model.population_pool[0])
        self.assertEqual(next_visit._ref_prices["sku-1"], learned)
        self.assertNotEqual(learned, sku.base_price)

    def test_empirical_mode_keeps_observed_reference_price_fixed(self):
        model = self.make_model()
        sku = model.get_product_by_id("sku-1")
        sku.current_price = 3.0
        consumer = ConsumerAgent("first", model, model.population_pool[0])

        consumer._execute_purchase(sku, 1, 10.0)

        self.assertEqual(
            model.population_pool[0]["_ref_prices"]["sku-1"], sku.base_price
        )

    def test_pantries_deplete_even_when_household_does_not_shop(self):
        cfg = {
            "products": [product()],
            "population": [profile(), profile()],
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=3)
        second_household = model.population_pool[1]
        opening = second_household["_home_inv"]["sku-1"]

        model.step()

        visitors = {c.household_id for c in model.last_daily_agents}
        self.assertNotIn(second_household["_household_id"], visitors)
        self.assertAlmostEqual(
            second_household["_home_inv"]["sku-1"],
            opening - 1.0 / second_household["_expected_visit_interval"],
        )

    def test_non_visitor_shortfall_enters_population_access_metric(self):
        cfg = {
            "products": [product()],
            "population": [profile(), profile()],
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=3)
        non_visitor = model.population_pool[1]
        non_visitor["_home_inv"] = {"sku-1": 0.0}

        model.step()

        visitors = {c.household_id for c in model.last_daily_agents}
        self.assertNotIn(non_visitor["_household_id"], visitors)
        self.assertGreater(non_visitor["_daily_consumption_unmet"], 0.0)
        self.assertEqual(non_visitor["_access_stress_score"], 4)
        record = model.daily_records[-1]
        self.assertEqual(record["HouseholdsWithConsumptionShortfall"], 1)
        self.assertEqual(record["AccessStressHigh_Mid"], 0.5)
        self.assertEqual(record["FIESSevere_Mid"], record["AccessStressHigh_Mid"])

    def test_panic_alone_does_not_create_access_stress(self):
        model = self.make_model()
        model.global_panic_level = 1.0

        model.step()

        household = model.population_pool[0]
        consumer = model.last_daily_agents[0]
        self.assertEqual(household["_daily_consumption_unmet"], 0.0)
        self.assertEqual(consumer.access_stress_score, 0)
        self.assertEqual(model.daily_records[-1]["AccessStress_Mid"], 0.0)

    def test_tpb_adjustment_is_applied_once_to_price_margin(self):
        model = self.make_model()
        consumer = ConsumerAgent("threshold-consumer", model, model.population_pool[0])
        consumer.price_acceptance_margin = 0.40

        threshold = consumer._price_acceptance_threshold(intention=0.80)

        self.assertAlmostEqual(threshold, 0.40 + (0.80 - 0.50) * 0.12)

    def test_invalid_panic_coefficient_is_rejected(self):
        cfg = {
            "products": [product()],
            "population": [profile()],
        }
        with self.assertRaisesRegex(ValueError, "panic_growth_rate"):
            SupermarketModel(
                config_data=cfg,
                base_consumers=1,
                panic_growth_rate=1.01,
            )

    def test_inventory_policy_rejects_unit_mismatches(self):
        cfg = {
            "products": [product()],
            "population": [profile()],
        }
        with self.assertRaisesRegex(ValueError, "capacity fraction"):
            SupermarketModel(
                config_data=cfg,
                base_consumers=1,
                reorder_pt=30,
                target_stock=90,
            )

    def test_consumption_access_outputs_are_internally_consistent(self):
        model = self.make_model()
        model.population_pool[0]["_home_inv"] = {"sku-1": 0.5}

        model.step()

        record = model.daily_records[-1]
        self.assertAlmostEqual(record["ConsumptionFulfillmentRate"], 0.5)
        self.assertAlmostEqual(record["ConsumptionShortfall_Mid"], 0.5)
        self.assertEqual(record["AccessStress_Mid"], 3.0)
        self.assertEqual(record["AccessStressHigh_Mid"], 1.0)

    def test_household_visits_are_unique_and_repeat_over_time(self):
        cfg = {
            "products": [product()],
            "population": [profile(), profile()],
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=5)
        observed = []

        for _ in range(6):
            model.step()
            ids = [c.household_id for c in model.last_daily_agents]
            self.assertEqual(len(ids), len(set(ids)))
            observed.extend(ids)

        self.assertEqual(set(observed), {
            p["_household_id"] for p in model.population_pool
        })
        self.assertTrue(any(p["_visit_count"] > 1 for p in model.population_pool))

    def test_household_pantry_ledger_conserves_units(self):
        model = self.make_model(quantity=2, budget=20.0)

        for _ in range(5):
            opening = sum(
                sum(p["_home_inv"].values()) for p in model.population_pool
            )
            model.step()
            purchased = sum(c.items_purchased for c in model.last_daily_agents)
            closing = sum(
                sum(p["_home_inv"].values()) for p in model.population_pool
            )
            consumed = model.daily_records[-1]["HouseholdConsumption"]
            self.assertAlmostEqual(opening + purchased, closing + consumed, places=6)

    def test_substitute_replenishes_the_original_household_need(self):
        cfg = {
            "products": [
                product(product_id="wanted", name="Milk A"),
                product(product_id="sub", name="Milk B"),
            ],
            "population": [profile(product_id="wanted", name="Milk A")],
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=9)
        household = model.population_pool[0]
        household["_home_inv"] = {"wanted": 0.0}
        substitute = model.get_product_by_id("sub")
        visitor = ConsumerAgent("visit", model, household)

        _, units = visitor._execute_purchase(
            substitute, 1, 20.0, is_substitute=True, pantry_key="wanted"
        )

        self.assertEqual(units, 1)
        self.assertEqual(household["_home_inv"]["wanted"], 1.0)
        self.assertNotIn("sub", household["_home_inv"])

    def test_substitution_skips_unaffordable_high_compatibility_candidate(self):
        wanted = product(product_id="wanted", name="Milk A", price=2.0)
        expensive = product(
            product_id="expensive", name="Milk Organic", price=10.0, organic=True
        )
        affordable = product(
            product_id="affordable", name="Milk Basic", price=1.5, organic=False
        )
        p = {
            **profile(product_id="wanted", name="Milk A"),
            "sub_tolerance": 1.0,
            "organic_preference": 1.0,
            "price_sensitivity": 0.0,
        }
        cfg = {
            "products": [wanted, expensive, affordable],
            "population": [p],
            "stats": {"dce_choice_validation": {
                "status": "ok", "beats_majority_benchmark": True,
            }, "substitution_choice_validation": {
                "supported_ranking_categories": ["Milk"],
            }},
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=9)
        visitor = ConsumerAgent("visit", model, model.population_pool[0])

        substitute = visitor._find_best_substitute(
            "Milk", 1, "wanted",
            wanted_product=model.get_product_by_id("wanted"),
            remaining_budget=2.0,
        )

        self.assertEqual(substitute.prod_id, "affordable")
        self.assertGreater(visitor.substitution_price_rejections, 0)

    def test_substitute_price_acceptance_uses_wanted_sku_reference(self):
        wanted = product(product_id="wanted", name="Milk A", price=2.0)
        costly = product(product_id="costly", name="Milk B", price=3.0)
        cheap = product(product_id="cheap", name="Milk C", price=1.8)
        p = {
            **profile(product_id="wanted", name="Milk A"),
            "sub_tolerance": 1.0,
            "price_sensitivity": 1.0,
            "revealed_preference_margin": 0.10,
        }
        cfg = {
            "products": [wanted, costly, cheap],
            "population": [p],
            "stats": {"substitution_choice_validation": {
                "candidate_price_gate_supported": True,
                "supported_ranking_categories": ["Milk"],
            }},
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=4)
        visitor = ConsumerAgent("visit", model, model.population_pool[0])

        substitute = visitor._find_best_substitute(
            "Milk", 1, "wanted",
            wanted_product=model.get_product_by_id("wanted"),
            remaining_budget=5.0,
        )

        self.assertEqual(substitute.prod_id, "cheap")

    def test_substitution_never_crosses_catalogue_category(self):
        wanted = product(product_id="wanted", name="Milk A", price=2.0)
        cheese = product(product_id="cheese", name="Cheese", price=1.0)
        cheese["category"] = "Cheese"
        p = {
            **profile(product_id="wanted", name="Milk A"),
            "sub_tolerance": 1.0,
            "price_sensitivity": 0.0,
        }
        model = SupermarketModel(
            config_data={"products": [wanted, cheese], "population": [p]},
            base_consumers=1,
            fixed_seed=4,
        )
        visitor = ConsumerAgent("visit", model, model.population_pool[0])

        substitute = visitor._find_best_substitute(
            "Milk", 1, "wanted",
            wanted_product=model.get_product_by_id("wanted"),
            remaining_budget=5.0,
        )

        self.assertIsNone(substitute)

    def test_milk_dce_preferences_are_not_extrapolated_to_cheese(self):
        wanted = product(product_id="wanted", name="Cheese A", price=2.0)
        wanted["category"] = "Cheese"
        organic = product(
            product_id="organic", name="Cheese Organic", price=1.8, organic=True
        )
        organic["category"] = "Cheese"
        cheap = product(product_id="cheap", name="Cheese Basic", price=1.5)
        cheap["category"] = "Cheese"
        p = {
            **profile(product_id="wanted", name="Cheese A"),
            "sub_tolerance": 1.0,
            "organic_preference": 1.0,
            "price_sensitivity": 0.0,
        }
        p["baseline_basket"][0]["category"] = "Cheese"
        p["crisis_basket"] = p["baseline_basket"]
        cfg = {
            "products": [wanted, organic, cheap],
            "population": [p],
            "stats": {"dce_choice_validation": {
                "status": "ok",
                "beats_majority_benchmark": True,
                "applicable_categories": ["Milk"],
            }},
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=4)
        visitor = ConsumerAgent("visit", model, model.population_pool[0])

        substitute = visitor._find_best_substitute(
            "Cheese", 1, "wanted",
            wanted_product=model.get_product_by_id("wanted"),
            remaining_budget=5.0,
        )

        self.assertIn(substitute.prod_id, {"organic", "cheap"})
        self.assertEqual(
            visitor._nonprice_compatibility(model.get_product_by_id("organic")),
            visitor._nonprice_compatibility(model.get_product_by_id("cheap")),
        )

    def test_validated_dce_price_model_probabilistically_favors_cheaper_milk(self):
        wanted = product(product_id="wanted", name="Milk A", price=2.0)
        cheap = product(product_id="cheap", name="Milk Cheap", price=1.0)
        expensive = product(product_id="expensive", name="Milk Expensive", price=3.0)
        p = {
            **profile(product_id="wanted", name="Milk A"),
            "sub_tolerance": 1.0,
        }
        cfg = {
            "products": [wanted, cheap, expensive],
            "population": [p],
            "stats": {"dce_choice_validation": {
                "status": "ok",
                "beats_null_benchmark": True,
                "model_converged": True,
                "price_coefficient_estimable": True,
                "utility_scale_compatible_with_price": True,
                "price_coefficient": -5.0,
                "origin_coefficient": 0.0,
                "organic_coefficient": 0.0,
                "fat_linear_coefficient": 0.0,
                "fat_quadratic_coefficient": 0.0,
                "applicable_categories": ["Milk"],
            }},
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=11)
        visitor = ConsumerAgent("visit", model, model.population_pool[0])

        selected = [
            visitor._find_best_substitute(
                "Milk", 1, "wanted",
                wanted_product=model.get_product_by_id("wanted"),
                remaining_budget=10.0,
            ).prod_id
            for _ in range(100)
        ]

        self.assertGreater(selected.count("cheap"), 95)

    def test_validated_phase_transition_weights_allocate_nonmilk_substitutes(self):
        wanted = product(product_id="wanted", name="Yogurt A", price=2.0)
        common = product(product_id="common", name="Yogurt B", price=2.0)
        rare = product(product_id="rare", name="Yogurt C", price=2.0)
        for item in (wanted, common, rare):
            item["category"] = "Yogurt"
        p = {
            **profile(product_id="wanted", name="Yogurt A"),
            "sub_tolerance": 1.0,
        }
        p["baseline_basket"][0]["category"] = "Yogurt"
        p["crisis_basket"] = p["baseline_basket"]
        cfg = {
            "products": [wanted, common, rare],
            "population": [p],
            "stats": {"substitution_choice_validation": {
                "supported_transition_categories": ["Yogurt"],
                "empirical_transition_weights": {
                    "Yogurt": {"common": 100.0, "rare": 1.0},
                },
            }},
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=12)
        visitor = ConsumerAgent("visit", model, model.population_pool[0])

        selected = [
            visitor._find_best_substitute(
                "Yogurt", 1, "wanted",
                wanted_product=model.get_product_by_id("wanted"),
                remaining_budget=10.0,
            ).prod_id
            for _ in range(100)
        ]

        self.assertGreater(selected.count("common"), 95)

    def test_observed_baseline_choice_is_acceptable_at_reference_price(self):
        cfg = {
            "products": [product()],
            "population": [{
                **profile(),
                "price_sensitivity": 1.0,
                "finnish_preference": 0.0,
                "organic_preference": 0.0,
            }],
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=2)
        visitor = ConsumerAgent("visit", model, model.population_pool[0])
        sku = model.get_product_by_id("sku-1")

        self.assertTrue(visitor._accepts_price(sku))
        self.assertAlmostEqual(visitor._price_loss(sku), 0.0)

    def test_zero_panic_sensitivity_prevents_scarcity_contagion(self):
        model = self.make_model(quantity=3)
        model.panic_sensitivity = 0.0
        sku = model.get_product_by_id("sku-1")
        sku.shelf_batches = [{"qty": 1, "age": 0}]
        sku.stock_storage = 0

        model.step()

        self.assertGreater(model.panic_signals, 0)
        self.assertEqual(model.global_panic_level, 0.0)

    def test_fully_empty_shelf_emits_scarcity_signal(self):
        model = self.make_model(quantity=3)
        sku = model.get_product_by_id("sku-1")
        sku.shelf_batches = []
        sku.stock_storage = 0

        model.step()

        self.assertGreater(model.panic_signals, 0)
        self.assertGreater(model.daily_records[-1]["LostSales"], 0)

    def test_currency_rounding_does_not_reject_affordable_final_item(self):
        cfg = {
            "products": [product(price=0.10)],
            "population": [profile(quantity=3, budget=0.30)],
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=4)
        sku = model.get_product_by_id("sku-1")
        sku.shelf_batches = [{"qty": 3, "age": 0}]
        visitor = ConsumerAgent("visit", model, model.population_pool[0])

        spent, units = visitor._execute_purchase(sku, 3, 0.30)

        self.assertEqual((spent, units), (0.30, 3))
        self.assertFalse(visitor.budget_exhausted)

    def test_observed_crisis_basket_is_not_used_as_simulated_demand(self):
        p = profile(quantity=1, budget=20.0)
        p["crisis_basket"] = [{**p["baseline_basket"][0], "quantity": 5}]
        cfg = {"products": [product()], "population": [p]}
        model = SupermarketModel(
            config_data=cfg, base_consumers=1, fixed_seed=6,
            is_crisis_mode=True, scenario_start_day=1, inflation_pct=1.0,
        )

        model.step()

        self.assertTrue(model.is_scenario_active)
        self.assertEqual(model.last_daily_agents[0].items_base_wanted, 1)

    def test_crisis_flag_without_a_shock_matches_baseline(self):
        baseline = self.make_model()
        cfg = {
            "products": [product()],
            "population": [profile()],
        }
        no_shock = SupermarketModel(
            config_data=cfg, base_consumers=1, fixed_seed=7,
            is_crisis_mode=True, scenario_start_day=1,
        )

        for _ in range(5):
            baseline.step()
            no_shock.step()

        self.assertEqual(baseline.daily_records, no_shock.daily_records)

    def test_unidentified_behavioral_dynamics_are_off_by_default(self):
        model = self.make_model()

        self.assertFalse(model.panic_dynamics_enabled)
        self.assertFalse(model.tpb_enabled)
        self.assertFalse(model.prospect_theory_enabled)
        self.assertFalse(model.preference_learning_enabled)
        self.assertFalse(model.archetype_modifiers_enabled)
        self.assertFalse(model.policy_choice_effects_enabled)
        model.step()
        record = model.daily_records[-1]
        self.assertEqual(record["BehaviorEvidenceMode"], "empirical_only")
        self.assertEqual(record["PanicDynamicsEnabled"], 0)

    def test_empirical_price_rule_uses_transparent_relative_price_ratio(self):
        cfg = {
            "products": [product(price=2.0)],
            "population": [{
                **profile(),
                "price_sensitivity": 0.8,
                "finnish_preference": 0.0,
                "organic_preference": 0.0,
            }],
        }
        model = SupermarketModel(config_data=cfg, base_consumers=1, fixed_seed=2)
        visitor = ConsumerAgent("visit", model, model.population_pool[0])
        sku = model.get_product_by_id("sku-1")
        baseline_loss = visitor._price_loss(sku)
        sku.current_price = 3.0

        shocked_loss = visitor._price_loss(sku)

        self.assertAlmostEqual(baseline_loss, 0.0)
        self.assertAlmostEqual(shocked_loss, 0.4)

    def test_preference_learning_requires_explicit_opt_in(self):
        p = {
            **profile(),
            "archetype": "green_buyer",
            "organic_preference": 0.2,
        }
        cfg = {"products": [product(organic=True)], "population": [p]}
        empirical = SupermarketModel(
            config_data=cfg, base_consumers=1, fixed_seed=1,
        )
        empirical.step()
        self.assertEqual(empirical.population_pool[0]["organic_preference"], 0.2)

        exploratory = SupermarketModel(
            config_data=cfg, base_consumers=1, fixed_seed=1,
            enable_preference_learning=True,
            enable_archetype_modifiers=True,
        )
        exploratory.step()
        self.assertGreater(
            exploratory.population_pool[0]["organic_preference"], 0.2
        )

    def test_panic_response_requires_explicit_opt_in(self):
        cfg = {"products": [product()], "population": [profile()]}
        empirical = SupermarketModel(
            config_data=cfg, base_consumers=1, fixed_seed=1,
            is_crisis_mode=True, scenario_start_day=1,
            inflation_pct=25.0, panic_sens=1.0,
        )
        exploratory = SupermarketModel(
            config_data=cfg, base_consumers=1, fixed_seed=1,
            is_crisis_mode=True, scenario_start_day=1,
            inflation_pct=25.0, panic_sens=1.0,
            enable_panic_dynamics=True,
        )

        empirical.step()
        exploratory.step()

        self.assertEqual(empirical.global_panic_level, 0.0)
        self.assertGreater(exploratory.global_panic_level, 0.0)


class CalibrationTests(unittest.TestCase):
    def test_calibration_marks_observed_predictions_as_cross_fitted(self):
        profiles = []
        for i in range(20):
            p = profile()
            p.update({
                "source_id": f"p{i}",
                "has_crisis_observation": True,
                "observed_quantity_retention": 0.5 + (i % 3) * 0.1,
                "observed_substitution_rate": (i % 4) / 4,
                "observed_substitution_lines": i % 4,
                "observed_phase2_choice_lines": 4,
                "baseline_choice_lines": 5,
                "observed_price_shock": 0.25,
                "dce_price_sensitivity": (i % 5) / 5,
                "archetype": "price_champion",
            })
            profiles.append(p)

        calibrated, diagnostics = calibrate_behavioral_profiles(profiles)

        self.assertEqual(diagnostics["n_observed"], 20)
        self.assertEqual(diagnostics["n_validation"], 4)
        self.assertTrue(all(
            p["calibration_prediction_is_cross_fitted"] for p in calibrated
        ))
        self.assertTrue(all(0 <= p["price_sensitivity"] <= 1 for p in calibrated))
        self.assertEqual(
            diagnostics["substitution_action_method"],
            "fixed_holdout_repeated_10x_nested_participant_cv",
        )
        self.assertTrue(all(
            0 <= p["substitution_action_probability"] <= 1
            for p in calibrated
        ))
        self.assertTrue(all(
            p["sub_tolerance"] == p["substitution_action_probability"]
            for p in calibrated
        ))


if __name__ == "__main__":
    unittest.main()
