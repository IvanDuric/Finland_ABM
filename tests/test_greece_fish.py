from greece_fish import build_greece_fish_synthetic_config
from model import PolicyConfig, SupermarketModel


def test_greece_fish_config_is_deterministic_and_explicitly_synthetic():
    first = build_greece_fish_synthetic_config()
    second = build_greece_fish_synthetic_config()
    assert first == second
    assert first["stats"]["data_status"] == "fully_synthetic_no_empirical_data"
    assert first["stats"]["n_real"] == 0
    assert first["stats"]["n_synthetic_templates"] == 240
    assert first["stats"]["dce_choice_validation"]["n_participants"] == 0
    assert len(first["products"]) == 12


def test_fish_catalogue_has_distinct_perishability_groups():
    config = build_greece_fish_synthetic_config()
    categories = {product["category"] for product in config["products"]}
    assert categories == {
        "Fresh farmed fish", "Small pelagic fish", "Frozen fish", "Canned fish"
    }
    shelf_lives = {product["category"]: product["shelf_life_days"]
                   for product in config["products"]}
    assert shelf_lives["Small pelagic fish"] <= 5
    assert shelf_lives["Frozen fish"] >= 150
    assert shelf_lives["Canned fish"] == 730


def test_affordable_fish_subsidy_targets_small_pelagic_and_canned_only():
    policy = PolicyConfig({
        "subsidy_active": True,
        "subsidy_target": "category",
        "subsidy_categories": ["Small pelagic fish", "Canned fish"],
        "subsidy_rate": 0.20,
    })
    assert policy.apply_price_policy(4.0, 0.0, True, False, "Small pelagic fish") == 3.2
    assert policy.apply_price_policy(4.0, 0.0, True, False, "Canned fish") == 3.2
    assert policy.apply_price_policy(4.0, 0.0, True, False, "Fresh farmed fish") == 4.0


def test_greece_fish_model_runs_without_empirical_choice_claims():
    config = build_greece_fish_synthetic_config(template_count=80, pool_size=160)
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
        inflation_pct=20,
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
