from io import BytesIO

import pandas as pd
from pypdf import PdfReader

import app


class _DummyModel:
    def step(self):
        return None


def _minimal_params():
    return {
        "days": 2,
        "month": 1,
        "base_con": 10,
        "reorder": 0.30,
        "target": 0.90,
        "lead": 3,
        "cri_start": 1,
        "cri_duration": 1,
        "inf": 25.0,
        "dis": 7,
        "panic": 0.50,
        "hoard": 1.5,
        "policy_cfg": {
            "subsidy_active": True,
            "subsidy_target": "domestic",
            "subsidy_rate": 0.15,
        },
        "purchase_limit": 3,
        "media_intensity": 0.6,
        "communication_type": "calming",
    }


def test_securefood_policy_run_builds_paired_no_policy_counterfactual(monkeypatch):
    calls = []

    def fake_make_model(params, is_crisis, seed, policy_cfg=None, **_kwargs):
        calls.append({
            "is_crisis": is_crisis,
            "policy_cfg": dict(policy_cfg or {}),
            "purchase_limit": params.get("purchase_limit"),
            "media_intensity": params.get("media_intensity"),
            "communication_type": params.get("communication_type"),
        })
        return _DummyModel()

    def fake_collect(_model, day, scenario):
        return {"Day": day, "Scenario": scenario}, []

    monkeypatch.setattr(app, "_make_model", fake_make_model)
    monkeypatch.setattr(app, "_collect_model_day", fake_collect)

    result = app._sf_run_simulation(_minimal_params())

    assert result is not None
    assert result["has_policy_counterfactual"] is True
    assert set(result["df"]["Scenario"]) == {
        "Baseline", "Crisis", "Crisis (No Policy)"
    }
    assert len(result["df"]) == 6

    baseline, policy_crisis, no_policy_crisis = calls
    assert baseline == {
        "is_crisis": False,
        "policy_cfg": {},
        "purchase_limit": None,
        "media_intensity": 0.0,
        "communication_type": "neutral",
    }
    assert policy_crisis["is_crisis"] is True
    assert policy_crisis["policy_cfg"]["subsidy_active"] is True
    assert policy_crisis["purchase_limit"] == 3
    assert policy_crisis["media_intensity"] == 0.6
    assert no_policy_crisis == {
        "is_crisis": True,
        "policy_cfg": {},
        "purchase_limit": None,
        "media_intensity": 0.0,
        "communication_type": "neutral",
    }


def test_securefood_no_policy_run_does_not_add_redundant_counterfactual(monkeypatch):
    calls = []

    def fake_make_model(params, is_crisis, seed, policy_cfg=None, **_kwargs):
        calls.append((is_crisis, dict(policy_cfg or {})))
        return _DummyModel()

    def fake_collect(_model, day, scenario):
        return {"Day": day, "Scenario": scenario}, []

    params = _minimal_params()
    params.update({
        "policy_cfg": {},
        "purchase_limit": None,
        "media_intensity": 0.0,
        "communication_type": "neutral",
    })
    monkeypatch.setattr(app, "_make_model", fake_make_model)
    monkeypatch.setattr(app, "_collect_model_day", fake_collect)

    result = app._sf_run_simulation(params)

    assert result is not None
    assert result["has_policy_counterfactual"] is False
    assert set(result["df"]["Scenario"]) == {"Baseline", "Crisis"}
    assert len(calls) == 2


def test_securefood_default_report_presets_are_no_policy_and_independent():
    sc_first, pm_first = app._sf_preset_report_params()
    sc_second, pm_second = app._sf_preset_report_params()

    assert sc_first["days"] == 90
    assert sc_first["policy_cfg"]["subsidy_active"] is False
    assert pm_first["days"] == 120
    assert pm_first["policy_cfg"]["subsidy_active"] is False
    assert pm_first["policy_cfg"]["subsidy_rate"] == 0.0
    assert pm_first["purchase_limit"] is None
    assert pm_first["communication_type"] == "neutral"
    assert pm_first["media_intensity"] == 0.0
    assert app._sf_has_active_policy(sc_first) is False
    assert app._sf_has_active_policy(pm_first) is False
    assert sc_first["exploratory_behaviour"] is True
    assert pm_first["exploratory_behaviour"] is True

    sc_first["policy_cfg"]["subsidy_active"] = True
    pm_first["policy_cfg"]["subsidy_active"] = True
    pm_first["policy_cfg"]["subsidy_rate"] = 0.99
    assert sc_second["policy_cfg"]["subsidy_active"] is False
    assert pm_second["policy_cfg"]["subsidy_active"] is False
    assert pm_second["policy_cfg"]["subsidy_rate"] == 0.0


def test_securefood_without_policy_disables_every_supported_policy_lever():
    params = _minimal_params()
    params["policy_cfg"].update({
        "fat_tax_active": True,
        "labelling_active": True,
    })

    stripped = app._sf_without_policy(params)

    assert app._sf_has_active_policy(params) is True
    assert app._sf_has_active_policy(stripped) is False
    assert stripped["purchase_limit"] is None
    assert stripped["media_intensity"] == 0.0
    assert stripped["communication_type"] == "neutral"
    assert stripped["policy_cfg"]["fat_tax_active"] is False
    assert stripped["policy_cfg"]["subsidy_active"] is False
    assert stripped["policy_cfg"]["labelling_active"] is False
    assert params["policy_cfg"]["subsidy_active"] is True


def test_securefood_parameter_signature_detects_changed_analysis_inputs():
    params = _minimal_params()
    same_values = _minimal_params()
    reordered = dict(reversed(list(params.items())))

    assert app._sf_param_signature(params) == app._sf_param_signature(same_values)
    assert app._sf_param_signature(params) == app._sf_param_signature(reordered)

    same_values["inf"] = 55.0
    assert app._sf_param_signature(params) != app._sf_param_signature(same_values)

    same_values = _minimal_params()
    same_values["policy_cfg"]["subsidy_rate"] = 0.30
    assert app._sf_param_signature(params) != app._sf_param_signature(same_values)


def test_securefood_default_report_excludes_optional_policy_chapter(monkeypatch):
    def fake_make_model(_params, is_crisis, seed, policy_cfg=None, **_kwargs):
        model = _DummyModel()
        model.is_crisis = is_crisis
        model.has_policy = bool(policy_cfg and policy_cfg.get("subsidy_active"))
        return model

    def fake_collect(model, day, scenario):
        crisis = 1.0 if model.is_crisis else 0.0
        policy = 1.0 if model.has_policy else 0.0
        aggregate = {
            "Day": day, "Scenario": scenario,
            "Revenue": 100.0 - 10.0 * crisis + policy,
            "NominalRevenue": 100.0 - 5.0 * crisis + policy,
            "LostSales": 2.0 * crisis, "PanicLevel": 0.2 * crisis,
            "Waste": 1.0, "AvgPrice": 2.0 + 0.5 * crisis,
            "StockpilePressure": 1.0 + 0.1 * crisis,
            "FoodStressedPct": 0.1 + 0.1 * crisis - 0.01 * policy,
            "BudgetExh_Low": 0.1 + 0.1 * crisis,
            "BudgetExh_High": 0.02 + 0.02 * crisis,
            "GiniAccess": 0.1 + 0.05 * crisis - 0.01 * policy,
            "ImportDepPct": 30.0 + 5.0 * crisis - policy,
            "Fulfillment_Low": 0.95 - 0.15 * crisis + 0.02 * policy,
            "Fulfillment_Mid": 0.97 - 0.10 * crisis + 0.01 * policy,
            "Fulfillment_High": 0.99 - 0.05 * crisis,
            "FulfillmentRate": 0.97 - 0.10 * crisis,
            "FIESSevere_Low": 0.05 + 0.10 * crisis - 0.01 * policy,
            "FIESSevere_Mid": 0.03 + 0.05 * crisis,
            "FIESSevere_High": 0.01 + 0.02 * crisis,
            "DomesticSales": 70.0 - 5.0 * crisis + policy,
            "ImportSales": 30.0 + 5.0 * crisis - policy,
        }
        product = [{
            "Day": day, "Scenario": scenario, "Category": "Milk", "Shelf": 20.0,
        }]
        return aggregate, product

    monkeypatch.setattr(app, "_make_model", fake_make_model)
    monkeypatch.setattr(app, "_collect_model_day", fake_collect)
    app._generate_sf_report_artifacts.clear()

    sc_params = _minimal_params()
    sc_params.update({"policy_cfg": {}, "purchase_limit": None, "media_intensity": 0.0})
    pm_params = _minimal_params()

    default_artifacts = app._generate_sf_report_artifacts(
        sc_params=sc_params,
        pm_params=pm_params,
        report_mode="Test default",
        include_policy_analysis=False,
    )
    default_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(default_artifacts["pdf"])).pages
    )
    default_csv = pd.read_csv(BytesIO(default_artifacts["aggregate_csv"]))
    assert "Food Security & Equity Analysis" in default_text
    assert "Policy Effectiveness & Recommendations" not in default_text
    assert not default_csv["PolicyAnalysisIncluded"].astype(bool).any()
    assert "Crisis (Selected Policy)" not in set(default_csv["Scenario"])

    app._generate_sf_report_artifacts.clear()
    policy_artifacts = app._generate_sf_report_artifacts(
        sc_params=sc_params,
        pm_params=pm_params,
        report_mode="Test policy",
        include_policy_analysis=True,
    )
    policy_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(policy_artifacts["pdf"])).pages
    )
    policy_csv = pd.read_csv(BytesIO(policy_artifacts["aggregate_csv"]))
    assert "Policy Effectiveness & Recommendations" in policy_text
    assert policy_csv["PolicyAnalysisIncluded"].astype(bool).all()
    assert "Crisis (Selected Policy)" in set(policy_csv["Scenario"])
