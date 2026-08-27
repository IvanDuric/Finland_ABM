import json
import os
from pathlib import Path

import pytest

from model import PolicyConfig, SupermarketModel
from portugal_fruits import (
    _count_carrot_material,
    build_portugal_fruit_config,
)


EXPORT = Path(os.environ["PORTUGAL_FRUITS_RAW_EXPORT"]) if os.environ.get("PORTUGAL_FRUITS_RAW_EXPORT") else None
BUNDLED_CONFIG = Path(__file__).parents[1] / "data" / "portugal_fruits_preliminary_config.json"


def test_carrot_material_is_detected_recursively():
    value = {
        "q1_cenoura": {"text": "Com que frequência come cenouras?", "value": 3},
        "fruit": ["orange", {"prompt": "carrot shape"}],
    }
    assert _count_carrot_material(value) == 3


@pytest.fixture(scope="module")
def portugal_config():
    if EXPORT is not None and EXPORT.exists():
        return build_portugal_fruit_config(
            json.loads(EXPORT.read_text(encoding="utf-8")), pool_size=200
        )
    return json.loads(BUNDLED_CONFIG.read_text(encoding="utf-8"))


def test_bundled_deployment_config_is_present_and_deidentified():
    config = json.loads(BUNDLED_CONFIG.read_text(encoding="utf-8"))
    assert len(config["population"]) == 73
    assert len(config["products"]) == 18
    assert all(
        profile["source_id"].startswith("pt_preliminary_")
        and profile["empirical_source_id"] == profile["source_id"]
        for profile in config["population"]
    )
    forbidden_keys = {"email", "name", "participantid", "firebaseid", "sessionid"}
    assert not forbidden_keys.intersection(
        key.casefold() for profile in config["population"] for key in profile
    )


def test_preliminary_export_filters_halle_and_carrots(portugal_config):
    stats = portugal_config["stats"]
    assert stats["raw_sessions"] == 376
    assert stats["finished_sessions"] == 79
    assert stats["halle_finished_excluded"] == 6
    assert stats["n_real"] == 73
    assert stats["carrot_fields_ignored"] > 0
    assert all("cenoura" not in product["name"].casefold()
               for product in portugal_config["products"])


def test_orange_dce_uses_recorded_prices_and_beats_null(portugal_config):
    dce = portugal_config["stats"]["dce_choice_validation"]
    assert dce["status"] == "ok"
    assert dce["n_participants"] == 71
    assert dce["price_source"] == "recorded_orange_dce_prices_eur_per_kg"
    assert dce["price_coefficient"] < 0
    assert dce["beats_null_benchmark"] is True
    assert dce["applicable_categories"] == ["Orange"]


def test_fruit_model_runs_and_activates_generic_orange_choice(portugal_config):
    model = SupermarketModel(
        config_data=portugal_config,
        base_consumers=80,
        start_month=7,
        reorder_pt=0.35,
        target_stock=0.85,
        lead_time=3,
        is_crisis_mode=True,
        scenario_start_day=5,
        crisis_duration=10,
        inflation_pct=20,
        disruption_days=5,
        panic_sens=0.4,
        hoarding_fac=1.35,
        fixed_seed=42,
        policy_cfg={},
    )
    for _ in range(20):
        model.step()
    latest = model.daily_records[-1]
    assert len(model.products) == 18
    assert latest["Sales"] >= 0
    assert latest["LostSales"] >= 0
    assert latest["ChoicePriceScaleIdentified"] == 1
    assert latest["DCEAttributeRankingCategories"] == "orange"
    assert model.dce_generic_feature_coefficients["price"] < 0


def test_orange_subsidy_does_not_subsidize_other_fruit():
    policy = PolicyConfig({
        "subsidy_active": True,
        "subsidy_target": "category",
        "subsidy_categories": ["Orange"],
        "subsidy_rate": 0.15,
    })
    assert policy.apply_price_policy(2.0, 0.0, True, False, "Orange") == 1.7
    assert policy.apply_price_policy(2.0, 0.0, True, False, "Other fruit") == 2.0
