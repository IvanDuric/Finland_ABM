from greece_dairy import build_greece_dairy_synthetic_config
from model import PolicyConfig, SupermarketModel


def test_greece_config_is_deterministic_and_explicitly_synthetic():
    first = build_greece_dairy_synthetic_config()
    second = build_greece_dairy_synthetic_config()
    assert first == second
    assert first["stats"]["data_status"] == "fully_synthetic_no_empirical_data"
    assert first["stats"]["n_real"] == 0
    assert first["stats"]["n_synthetic_templates"] == 240
    assert first["stats"]["dce_choice_validation"]["n_participants"] == 0
    assert len(first["products"]) == 12


def test_greece_catalogue_has_dairy_specific_categories_and_shelf_lives():
    config = build_greece_dairy_synthetic_config()
    categories = {product["category"] for product in config["products"]}
    assert categories == {"Milk", "Yogurt", "Cheese", "Cream", "Butter"}
    assert all(product["shelf_life_days"] > 0 for product in config["products"])
    assert min(product["shelf_life_days"] for product in config["products"]) < 10
    assert max(product["shelf_life_days"] for product in config["products"]) >= 45


def test_essential_dairy_subsidy_targets_milk_and_yogurt_only():
    policy = PolicyConfig({
        "subsidy_active": True,
        "subsidy_target": "category",
        "subsidy_categories": ["Milk", "Yogurt"],
        "subsidy_rate": 0.20,
    })
    assert policy.apply_price_policy(2.0, 1.5, True, False, "Milk") == 1.6
    assert policy.apply_price_policy(2.0, 2.0, True, False, "Yogurt") == 1.6
    assert policy.apply_price_policy(2.0, 21.0, True, False, "Cheese") == 2.0


def test_greece_synthetic_model_runs_without_empirical_choice_claims():
    config = build_greece_dairy_synthetic_config(template_count=80, pool_size=160)
    model = SupermarketModel(
        config_data=config,
        base_consumers=80,
        start_month=7,
        reorder_pt=0.35,
        target_stock=0.85,
        lead_time=3,
        is_crisis_mode=True,
        scenario_start_day=3,
        crisis_duration=5,
        inflation_pct=15,
        disruption_days=4,
        panic_sens=0.0,
        hoarding_fac=1.0,
        fixed_seed=42,
        policy_cfg={},
    )
    for _ in range(10):
        model.step()
    assert len(model.products) == 12
    assert model.dce_price_choice_supported is False
    assert model.dce_nonprice_validation_passed is False
    assert model.daily_records[-1]["Sales"] >= 0
    assert model.daily_records[-1]["UnmetDemandUnits"] >= 0
