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


def test_securefood_report_presets_are_explicit_and_independent():
    sc_first, pm_first = app._sf_preset_report_params()
    sc_second, pm_second = app._sf_preset_report_params()

    assert sc_first["days"] == 90
    assert sc_first["policy_cfg"]["subsidy_active"] is False
    assert pm_first["days"] == 120
    assert pm_first["policy_cfg"]["subsidy_rate"] == 0.15
    assert pm_first["purchase_limit"] == 3
    assert pm_first["communication_type"] == "calming"
    assert pm_first["media_intensity"] == 0.6
    assert sc_first["exploratory_behaviour"] is True
    assert pm_first["exploratory_behaviour"] is True

    sc_first["policy_cfg"]["subsidy_active"] = True
    pm_first["policy_cfg"]["subsidy_rate"] = 0.99
    assert sc_second["policy_cfg"]["subsidy_active"] is False
    assert pm_second["policy_cfg"]["subsidy_rate"] == 0.15
