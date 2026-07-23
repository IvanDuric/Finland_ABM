"""
GROCERYsim Data Processor v2.0
================================
Converts a Firebase JSON export + Unity product catalogue into an
enriched population pool for the Mesa ABM.

Pipeline
--------
1. Parse Firebase JSON → demographics, round1/round2 baskets, DCE choices,
   questionnaire ratings
2. Compute DCE preference scores (origin, organic, fat, cheaper-bundle choice proxy)
3. Audit questionnaire reliability and exploratory K-Means stability
4. Extract observed price elasticity from round1 vs round2 basket comparison
5. Build enriched "real" profiles
6. Retain observed profiles; each model seed resamples complete profiles
7. Return mesa_config dict  {products, population, stats}

DCE attribute encoding
----------------------
  Code = [Origin][Type][Fat]
    Origin : F = Finnish (Suomi), I = Imported
    Type   : C = Conventional, O = Organic
    Fat    : 0 = 0 % fat, 15 = 1.5 % fat, 38 = 3.8 % fat
  Examples: FC0, FO15, IC38, IO0 …

Questionnaire (Finnish, 21 items, 1–5 Likert)
----------------------------------------------
  Price         : Q2 (idx 1), Q6 (idx 5), Q19 (idx 18)
  Health        : Q5 (idx 4), Q12 (idx 11), Q14 (idx 13), Q15 (idx 14)
  Environment   : Q18 (idx 17), Q20 (idx 19), Q21 (idx 20)
  Animal welfare: Q7 (idx 6), Q9 (idx 8)
  Sensory/Habit : Q1 (idx 0), Q3 (idx 2), Q8 (idx 7), Q10 (idx 9),
                  Q11 (idx 10), Q13 (idx 12), Q17 (idx 16)
"""

import json
import csv
import math
import warnings
import copy

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, log_loss, silhouette_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Approximate reference prices (EUR/L, Finnish market 2024) used for
# DCE price-sensitivity scoring only — NOT used by the simulation engine.
DCE_PRICES = {
    "FC0": 1.25, "FC15": 1.45, "FC38": 1.85,
    "FO0": 1.65, "FO15": 1.90, "FO38": 2.35,
    "IC0": 1.05, "IC15": 1.20, "IC38": 1.55,
    "IO0": 1.40, "IO15": 1.65, "IO38": 2.05,
}

DCE_ATTRIBUTES = {
    "FC0":  {"origin": "Finnish",   "organic": False, "fat": 0.0},
    "FC15": {"origin": "Finnish",   "organic": False, "fat": 1.5},
    "FC38": {"origin": "Finnish",   "organic": False, "fat": 3.8},
    "FO0":  {"origin": "Finnish",   "organic": True,  "fat": 0.0},
    "FO15": {"origin": "Finnish",   "organic": True,  "fat": 1.5},
    "FO38": {"origin": "Finnish",   "organic": True,  "fat": 3.8},
    "IC0":  {"origin": "Imported",  "organic": False, "fat": 0.0},
    "IC15": {"origin": "Imported",  "organic": False, "fat": 1.5},
    "IC38": {"origin": "Imported",  "organic": False, "fat": 3.8},
    "IO0":  {"origin": "Imported",  "organic": True,  "fat": 0.0},
    "IO15": {"origin": "Imported",  "organic": True,  "fat": 1.5},
    "IO38": {"origin": "Imported",  "organic": True,  "fat": 3.8},
}

# Questionnaire factor groupings (0-indexed positions in the 21-item list)
QUESTIONNAIRE_FACTORS = {
    "price":          [1, 5, 18],
    "health":         [4, 11, 13, 14],
    "environment":    [17, 19, 20],
    "animal_welfare": [6, 8],
    "sensory_habit":  [0, 2, 7, 9, 10, 12, 16],
}

ARCHETYPE_LABELS = ["price_champion", "green_buyer", "health_optimizer", "habitual_buyer"]

# Income group midpoints (EUR/month) for demographic analytics
INCOME_MIDPOINTS = {
    "0-999":    500,  "1000-1999": 1500, "2000-2999": 2500,
    "3000-3999": 3500, "4000-4999": 4500, "5000-5999": 5500,
    "6000+":    7000,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Demographics extraction
# ---------------------------------------------------------------------------

def extract_demographics(user_data: dict) -> dict:
    demo = {
        "age":            int(user_data.get("age", 35)),
        "gender":         user_data.get("gender", "Unknown"),
        "income_group":   "Unknown",
        "income_midpoint": 2500,
        "employment":     "Unknown",
        "household_size": 2,
        "children":       0,
        "buys_milk":      True,
        "consumes_milk":  True,
    }
    for info in user_data.get("additionalDropdownInfo", []):
        label = info.get("label", "")
        value = info.get("value", "")
        if label == "Net monthly household income":
            demo["income_group"]   = value
            demo["income_midpoint"] = INCOME_MIDPOINTS.get(value, 2500)
        elif label == "Employment":
            demo["employment"] = value
        elif label == "Number of household members":
            try:
                demo["household_size"] = max(1, int(value))
            except (ValueError, TypeError):
                pass
        elif label == "Number of children":
            try:
                demo["children"] = max(0, int(value))
            except (ValueError, TypeError):
                pass
        elif label == "Buying milk":
            demo["buys_milk"] = str(value).lower() in ("kyllä", "yes", "1", "true")
        elif label == "Consuming milk":
            demo["consumes_milk"] = str(value).lower() in ("kyllä", "yes", "1", "true")
    return demo


# ---------------------------------------------------------------------------
# 2. DCE preference scoring
# ---------------------------------------------------------------------------

def parse_dce_choices(choices: list) -> dict:
    """
    From up to 18 pairwise choices, compute four continuous scores (0–1):
      finnish_preference   – tendency to prefer Finnish over Imported
      organic_preference   – tendency to prefer Organic over Conventional
      preferred_fat        – mean fat content (%) of chosen alternatives
      dce_cheaper_bundle_choice_rate – descriptive cheaper-coded bundle share;
                           not an identified price coefficient because displayed
                           prices are absent from the export
    """
    finnish_chosen = 0;  finnish_shown  = 0
    organic_chosen = 0;  organic_shown  = 0
    fat_chosen = []
    cheaper_chosen = 0;  price_decisions = 0

    for c in choices:
        choice = c.get("choiceMade", "none").lower()
        if choice == "none":
            continue
        left_code  = c.get("leftItemShown",  "")
        right_code = c.get("rightItemShown", "")
        if left_code not in DCE_ATTRIBUTES or right_code not in DCE_ATTRIBUTES:
            continue

        chosen_code  = left_code  if choice == "left"  else right_code
        chosen_attrs = DCE_ATTRIBUTES[chosen_code]
        left_attrs   = DCE_ATTRIBUTES[left_code]
        right_attrs  = DCE_ATTRIBUTES[right_code]

        # Origin preference (only informative when origins differ)
        if left_attrs["origin"] != right_attrs["origin"]:
            finnish_shown += 1
            if chosen_attrs["origin"] == "Finnish":
                finnish_chosen += 1

        # Organic preference (only informative when types differ)
        if left_attrs["organic"] != right_attrs["organic"]:
            organic_shown += 1
            if chosen_attrs["organic"]:
                organic_chosen += 1

        # Fat content of chosen
        fat_chosen.append(chosen_attrs["fat"])

        # Price sensitivity: did they pick the cheaper option?
        # Use actual prices from the enriched dataset if available; fall back to lookup table
        lp = float(c.get("leftItem",  {}).get("price") or DCE_PRICES.get(left_code,  1.5))
        rp = float(c.get("rightItem", {}).get("price") or DCE_PRICES.get(right_code, 1.5))
        if abs(lp - rp) > 0.05:
            price_decisions += 1
            chosen_price = lp if choice == "left" else rp
            other_price  = rp if choice == "left" else lp
            if chosen_price < other_price:
                cheaper_chosen += 1

    return {
        "finnish_preference":    (finnish_chosen / finnish_shown) if finnish_shown > 0 else 0.5,
        "organic_preference":    (organic_chosen / organic_shown) if organic_shown > 0 else 0.2,
        "preferred_fat":         float(np.mean(fat_chosen)) if fat_chosen else 1.5,
        "dce_cheaper_bundle_choice_rate": (
            cheaper_chosen / price_decisions if price_decisions > 0 else 0.5
        ),
        # Backward-compatible alias. Do not interpret as an identified price effect.
        "dce_price_sensitivity": (
            cheaper_chosen / price_decisions if price_decisions > 0 else 0.5
        ),
    }


def _clean_dce_preference_map(dce_rows: list[dict] | None) -> dict[str, dict]:
    """Derive participant descriptors from the cleaned long-format DCE rows."""
    if not dce_rows:
        return {}
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in dce_rows:
        participant_id = str(row.get("respondent_id", "")).strip()
        choice_id = str(row.get("choice_id", "")).strip()
        if participant_id and choice_id:
            grouped.setdefault((participant_id, choice_id), []).append(row)

    accum: dict[str, dict[str, list | int]] = {}
    for (participant_id, _), rows in grouped.items():
        if len(rows) != 3:
            continue
        product_rows = [r for r in rows if int(float(r.get("optout_asc", 0))) == 0]
        chosen = [r for r in rows if int(float(r.get("chosen", 0))) == 1]
        if len(product_rows) != 2 or len(chosen) != 1:
            continue
        state = accum.setdefault(participant_id, {
            "finnish": [], "organic": [], "fat": [], "cheaper": [],
            "optout": [],
        })
        chosen_row = chosen[0]
        is_optout = int(float(chosen_row.get("optout_asc", 0))) == 1
        state["optout"].append(float(is_optout))
        if is_optout:
            continue
        if len({int(float(r.get("origin", 0))) for r in product_rows}) > 1:
            state["finnish"].append(float(chosen_row.get("origin", 0)))
        if len({int(float(r.get("organic", 0))) for r in product_rows}) > 1:
            state["organic"].append(float(chosen_row.get("organic", 0)))
        state["fat"].append(float(chosen_row.get("fat", 1.5)))
        prices = [float(r.get("price", 0)) for r in product_rows]
        if abs(prices[0] - prices[1]) > 1e-9:
            state["cheaper"].append(float(
                float(chosen_row.get("price", 0)) == min(prices)
            ))

    result = {}
    for participant_id, state in accum.items():
        result[participant_id] = {
            "finnish_preference": float(np.mean(state["finnish"]))
            if state["finnish"] else 0.5,
            "organic_preference": float(np.mean(state["organic"]))
            if state["organic"] else 0.2,
            "preferred_fat": float(np.mean(state["fat"]))
            if state["fat"] else 1.5,
            "dce_cheaper_bundle_choice_rate": float(np.mean(state["cheaper"]))
            if state["cheaper"] else 0.5,
            "dce_price_sensitivity": float(np.mean(state["cheaper"]))
            if state["cheaper"] else 0.5,
            "dce_optout_rate": float(np.mean(state["optout"]))
            if state["optout"] else 0.0,
            "dce_preferences_use_recorded_prices": True,
        }
    return result


def calibrate_dce_choice_model(
    firebase_data: dict, random_state: int = 42,
    dce_rows: list[dict] | None = None,
) -> dict:
    """Validate pooled DCE attribute effects on held-out participants.

    When the cleaned long-format DCE file is supplied, a conditional multinomial
    logit is fitted to the two milk alternatives plus opt-out. Choice sets with an
    inferred price are excluded from estimation. The fallback Firebase-only path
    retains the older non-price diagnostic because displayed prices are absent
    from that export itself.
    """
    participants = firebase_data.get("participants", firebase_data)
    participant_ids = sorted([
        pid for pid, data in participants.items()
        if data.get("choiceExperiment1_Results")
    ])
    if len(participant_ids) < 10:
        return {"status": "insufficient_data", "n_participants": len(participant_ids)}

    if dce_rows:
        linked_ids = set(participant_ids)
        grouped: dict[str, list[dict]] = {}
        excluded_inferred = 0
        for row in dce_rows:
            participant_id = str(row.get("respondent_id", "")).strip()
            choice_id = str(row.get("choice_id", "")).strip()
            if participant_id in linked_ids and choice_id:
                grouped.setdefault(choice_id, []).append(row)

        choice_sets = []
        for choice_id, rows in grouped.items():
            if len(rows) != 3:
                continue
            if any(str(r.get("exclude_dce", "FALSE")).strip().upper() == "TRUE"
                   for r in rows):
                continue
            if any(int(float(r.get("price_inferred", 0))) == 1 for r in rows):
                excluded_inferred += 1
                continue
            chosen = [int(float(r.get("chosen", 0))) for r in rows]
            if sum(chosen) != 1:
                continue
            rows = sorted(rows, key=lambda r: int(float(r.get("alternative", 0))))
            X = []
            for row in rows:
                optout = int(float(row.get("optout_asc", 0))) == 1
                if optout:
                    version_two = float(str(row.get("version", "")) == "2")
                    X.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, version_two])
                else:
                    fat_centered = float(row.get("fat", 1.5)) - 1.5
                    X.append([
                        float(row.get("price", 0)),
                        float(row.get("origin", 0)),
                        float(row.get("organic", 0)),
                        fat_centered,
                        fat_centered ** 2,
                        0.0,
                        0.0,
                    ])
            choice_sets.append({
                "participant_id": str(rows[0].get("respondent_id", "")),
                "X": np.asarray(X, dtype=float),
                "chosen": int(np.argmax(chosen)),
                "version": str(rows[0].get("version", "")),
            })

        available_ids = sorted({row["participant_id"] for row in choice_sets})
        if len(available_ids) < 10:
            return {
                "status": "insufficient_data",
                "n_participants": len(available_ids),
                "price_source": "cleaned_dce_csv",
            }
        train_ids, validation_ids = train_test_split(
            available_ids, test_size=0.20, random_state=random_state
        )
        train_id_set = set(train_ids)
        validation_id_set = set(validation_ids)
        training = [r for r in choice_sets if r["participant_id"] in train_id_set]
        validation = [r for r in choice_sets if r["participant_id"] in validation_id_set]

        def objective(beta: np.ndarray) -> float:
            loss = 0.0
            for choice_set in training:
                utility = choice_set["X"] @ beta
                utility -= np.max(utility)
                log_denom = math.log(float(np.sum(np.exp(utility))))
                loss += log_denom - float(utility[choice_set["chosen"]])
            # Weak ridge regularisation prevents unstable coefficients in sparse
            # attribute combinations while leaving price on its observed EUR scale.
            return loss / max(1, len(training)) + 0.005 * float(np.sum(beta ** 2))

        fitted = minimize(
            objective, np.zeros(7, dtype=float), method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7},
        )
        beta = np.asarray(fitted.x, dtype=float)

        validation_loss = 0.0
        correct = 0
        training_optout_by_version = {
            version: float(np.mean([
                row["chosen"] == 2 for row in training
                if row["version"] == version
            ]))
            for version in {row["version"] for row in training}
        }
        overall_optout_share = float(np.mean([
            row["chosen"] == 2 for row in training
        ]))
        null_loss = 0.0
        null_correct = 0
        for choice_set in validation:
            utility = choice_set["X"] @ beta
            utility -= np.max(utility)
            probability = np.exp(utility) / np.sum(np.exp(utility))
            chosen_index = choice_set["chosen"]
            optout_share = training_optout_by_version.get(
                choice_set["version"], overall_optout_share
            )
            null_probabilities = np.asarray([
                (1.0 - optout_share) / 2.0,
                (1.0 - optout_share) / 2.0,
                optout_share,
            ])
            validation_loss -= math.log(max(float(probability[chosen_index]), 1e-12))
            correct += int(int(np.argmax(probability)) == chosen_index)
            null_loss -= math.log(max(float(null_probabilities[chosen_index]), 1e-12))
            null_correct += int(int(np.argmax(null_probabilities)) == chosen_index)
        n_validation = max(1, len(validation))
        validation_log_loss = validation_loss / n_validation
        null_log_loss = null_loss / n_validation
        accuracy = correct / n_validation
        null_accuracy = null_correct / n_validation
        beats_null = bool(validation_log_loss < null_log_loss)
        names = [
            "price_coefficient", "origin_coefficient", "organic_coefficient",
            "fat_linear_coefficient", "fat_quadratic_coefficient",
            "optout_coefficient", "optout_version2_coefficient",
        ]
        result = {
            "status": "ok",
            "n_participants": len(available_ids),
            "n_train_choices": len(training),
            "n_validation_choices": len(validation),
            "n_inferred_price_choice_sets_excluded": excluded_inferred,
            "validation_accuracy": round(float(accuracy), 4),
            "null_model_accuracy": round(float(null_accuracy), 4),
            "majority_accuracy": round(float(null_accuracy), 4),
            "validation_log_loss": round(float(validation_log_loss), 4),
            "null_model_log_loss": round(float(null_log_loss), 4),
            "beats_null_benchmark": beats_null,
            "beats_majority_benchmark": beats_null,
            "model_converged": bool(fitted.success),
            "model_message": str(fitted.message),
            **{name: round(float(value), 6) for name, value in zip(names, beta)},
            "operational_use": "pooled_milk_candidate_multinomial_probabilities",
            "participant_ranking_method": "pooled_conditional_multinomial_logit",
            "attribute_scope": "milk_products_only",
            "applicable_categories": ["Milk"],
            "utility_scale_compatible_with_price": True,
            "price_coefficient_estimable": True,
            "price_source": "cleaned_dce_csv_recorded_prices",
            "price_unit": "EUR_per_displayed_unit",
            "estimation_sample": "training_participants_only",
            "caution": (
                "Pooled preference model; participant-specific random coefficients "
                "and willingness-to-pay heterogeneity are not identified."
            ),
        }
        return result

    train_ids, validation_ids = train_test_split(
        participant_ids, test_size=0.20, random_state=random_state
    )

    def _rows(ids: list[str], include_lookup_price: bool = False):
        X, y = [], []
        for participant_id in ids:
            for choice in participants[participant_id].get(
                "choiceExperiment1_Results", []
            ):
                left = choice.get("leftItemShown", "")
                right = choice.get("rightItemShown", "")
                made = str(choice.get("choiceMade", "none")).lower()
                if (left not in DCE_ATTRIBUTES or right not in DCE_ATTRIBUTES
                        or made not in {"left", "right"}):
                    continue
                la, ra = DCE_ATTRIBUTES[left], DCE_ATTRIBUTES[right]
                row = []
                if include_lookup_price:
                    row.append(DCE_PRICES[left] - DCE_PRICES[right])
                row.extend([
                    float(la["origin"] == "Finnish") - float(ra["origin"] == "Finnish"),
                    float(la["organic"]) - float(ra["organic"]),
                    la["fat"] - ra["fat"],
                    la["fat"] ** 2 - ra["fat"] ** 2,
                ])
                X.append(row)
                y.append(int(made == "left"))
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

    X_train, y_train = _rows(train_ids)
    X_validation, y_validation = _rows(validation_ids)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_validation)[:, 1]
    accuracy = float(accuracy_score(y_validation, probabilities >= 0.5))
    majority_accuracy = float(max(np.mean(y_validation), 1.0 - np.mean(y_validation)))
    coefficients = model.coef_[0]

    X_lookup, _ = _rows(participant_ids, include_lookup_price=True)
    return {
        "status": "ok",
        "n_participants": len(participant_ids),
        "n_train_choices": len(y_train),
        "n_validation_choices": len(y_validation),
        "validation_accuracy": round(accuracy, 4),
        "majority_accuracy": round(majority_accuracy, 4),
        "validation_log_loss": round(float(log_loss(y_validation, probabilities)), 4),
        "origin_coefficient": round(float(coefficients[0]), 4),
        "organic_coefficient": round(float(coefficients[1]), 4),
        "fat_linear_coefficient": round(float(coefficients[2]), 4),
        "fat_quadratic_coefficient": round(float(coefficients[3]), 4),
        "beats_majority_benchmark": bool(accuracy > majority_accuracy),
        "operational_use": "milk_attribute_validation; substitution_use_requires_separate_replacement_gate",
        "participant_ranking_method": "equal_weight_descriptive_attribute_compatibility",
        "attribute_scope": "milk_products_only",
        "applicable_categories": ["Milk"],
        "utility_scale_compatible_with_price": False,
        "price_coefficient_estimable": False,
        "price_source": "actual_displayed_prices_absent",
        "lookup_design_condition_number": round(float(np.linalg.cond(X_lookup)), 2),
    }


# ---------------------------------------------------------------------------
# 3. Questionnaire factor scoring
# ---------------------------------------------------------------------------

def parse_questionnaire(ratings: list) -> dict:
    """
    Parse the 21-item Likert questionnaire into five factor scores (0–1).
    Missing items are excluded from each construct mean. A neutral score is used
    only when fewer than half of a construct's items were observed, and that
    fallback is explicitly recorded rather than silently imputing item answers.
    """
    scores = _questionnaire_vector(ratings)

    factor_scores = {}
    for factor, indices in QUESTIONNAIRE_FACTORS.items():
        vals = [scores[i] for i in indices if np.isfinite(scores[i])]
        required = int(math.ceil(len(indices) / 2))
        # Normalise from [1, 5] → [0, 1]
        factor_scores[f"q_{factor}"] = (
            float((np.mean(vals) - 1.0) / 4.0) if len(vals) >= required else 0.5
        )
        factor_scores[f"q_{factor}_observed_items"] = len(vals)
        factor_scores[f"q_{factor}_fallback"] = bool(len(vals) < required)

    return factor_scores


def _questionnaire_vector(ratings: list) -> np.ndarray:
    """Return the 21 positional Likert responses with missing values as NaN."""
    scores = np.full(21, np.nan, dtype=float)
    for index, rating in enumerate(ratings[:21]):
        try:
            value = float(rating["value"])
            if 1.0 <= value <= 5.0:
                scores[index] = value
        except (KeyError, ValueError, TypeError):
            continue
    return scores


def _cronbach_alpha(matrix: np.ndarray) -> float | None:
    """Raw Cronbach alpha for complete rows of one declared construct."""
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None
    item_variances = np.var(matrix, axis=0, ddof=1)
    total_variance = float(np.var(np.sum(matrix, axis=1), ddof=1))
    if not np.isfinite(total_variance) or total_variance <= 1e-12:
        return None
    k = matrix.shape[1]
    return float(k / (k - 1) * (1.0 - float(np.sum(item_variances)) / total_variance))


def questionnaire_reliability(firebase_data: dict) -> dict:
    """Audit declared questionnaire constructs without inventing missing answers."""
    participants = firebase_data.get("participants", firebase_data)
    vectors = [
        _questionnaire_vector(data.get("questionnaireRatings", []))
        for data in participants.values()
        if isinstance(data, dict) and data.get("questionnaireRatings")
    ]
    if not vectors:
        return {"status": "insufficient_data", "n_participants": 0, "constructs": []}
    responses = np.vstack(vectors)
    constructs = []
    for factor, indices in QUESTIONNAIRE_FACTORS.items():
        block = responses[:, indices]
        complete = block[np.isfinite(block).all(axis=1)]
        alpha = _cronbach_alpha(complete)
        if len(complete) < 20 or alpha is None:
            quality = "insufficient"
        elif alpha >= 0.70:
            quality = "acceptable"
        elif alpha >= 0.60:
            quality = "exploratory"
        else:
            quality = "weak"
        constructs.append({
            "construct": factor,
            "n_items": len(indices),
            "n_complete": int(len(complete)),
            "missing_cell_rate": round(float(np.mean(~np.isfinite(block))), 4),
            "cronbach_alpha": round(alpha, 4) if alpha is not None else None,
            "quality": quality,
        })
    return {
        "status": "ok",
        "n_participants": len(responses),
        "constructs": constructs,
        "all_constructs_acceptable": all(
            row["quality"] == "acceptable" for row in constructs
        ),
        "method": "raw_cronbach_alpha_complete_rows",
        "caution": "Declared positional item groups; no reverse-key metadata or confirmatory factor model available.",
    }


# ---------------------------------------------------------------------------
# 4. Price elasticity from round1 vs round2 basket comparison
# ---------------------------------------------------------------------------

def compute_price_elasticity(
    round1_basket: list,
    round2_basket: list,
    round1_total: float,
    round2_max: float,
) -> dict:
    """
    Compare observed behaviour in the normal (round1) and crisis (round2)
    shopping sessions to derive individual-level price elasticity proxies.

    Returns
    -------
    budget_utilization   – fraction of crisis budget actually spent (0–1)
    spending_reduction   – proportional drop in spending round1 → round2
    substitution_rate    – fraction of round2 items that differ from round1
                           choices in the same category
    price_sensitivity    – composite behavioural sensitivity score (0–1)
    """
    if not round2_basket:
        return {
            "budget_utilization": 1.0,
            "spending_reduction": 0.0,
            "substitution_rate":  0.0,
            "price_sensitivity":  0.5,
            "quantity_retention": 1.0,
            "round2_actual_spend": 0.0,
            "substitution_lines": 0,
            "phase2_choice_lines": 0,
            "baseline_choice_lines": len(round1_basket),
        }

    def item_name(item: dict) -> str:
        return str(item.get("product_name", item.get("productName", "")))

    def item_category(item: dict) -> str:
        return str(item.get("category", ""))

    round2_actual = sum(
        float(i.get("price", 0)) * int(i.get("quantity", 1))
        for i in round2_basket
    )

    budget_util       = min(1.0, round2_actual / round2_max) if round2_max > 0 else 1.0
    spending_reduction = max(0.0, 1.0 - round2_actual / round1_total) if round1_total > 0 else 0.0

    # Substitution: a phase-two product not bought in phase one, in a category
    # that was represented in phase one.
    r1_names = {item_name(i) for i in round1_basket}
    r1_categories = {item_category(i) for i in round1_basket}
    subs = sum(
        1 for item in round2_basket
        if item_category(item) in r1_categories
        and item_name(item) not in r1_names
    )
    substitution_rate = subs / len(round2_basket) if round2_basket else 0.0

    round1_units = sum(max(1, int(i.get("quantity", 1))) for i in round1_basket)
    round2_units = sum(max(1, int(i.get("quantity", 1))) for i in round2_basket)
    quantity_retention = (
        min(1.5, round2_units / round1_units) if round1_units > 0 else 1.0
    )

    # Composite: weighted blend of the three signals
    composite = min(
        1.0,
        0.40 * spending_reduction +
        0.30 * (1.0 - budget_util) +
        0.30 * substitution_rate,
    )

    return {
        "budget_utilization": float(budget_util),
        "spending_reduction": float(spending_reduction),
        "substitution_rate":  float(substitution_rate),
        "price_sensitivity":  float(composite),
        "quantity_retention": float(quantity_retention),
        "round2_actual_spend": float(round2_actual),
        "substitution_lines": int(subs),
        "phase2_choice_lines": len(round2_basket),
        "baseline_choice_lines": len(round1_basket),
    }


# ---------------------------------------------------------------------------
# 5. Clean basket items against the product catalogue
# ---------------------------------------------------------------------------

def canonicalize_products(products: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Collapse repeated catalogue placements into one scientific SKU per name.

    The Unity export contains many rows that differ only in their scene-object
    ``id``.  Survey baskets identify products by name, so treating those rows as
    separate SKUs creates unreachable inventory and duplicate expiry events.
    Duplicate names are accepted only when every product attribute except ``id``
    is identical; conflicting definitions fail loudly instead of being resolved
    arbitrarily.
    """
    canonical: list[dict] = []
    name_to_product: dict[str, dict] = {}
    source_ids: dict[str, list[str]] = {}

    for raw in products:
        product = dict(raw)
        name = str(product.get("name", "")).strip()
        if not name:
            raise ValueError("Product catalogue contains a product without a name.")
        product["name"] = name
        product_id = str(product.get("id", "")).strip()
        if not product_id:
            raise ValueError(f"Product {name!r} has no stable id.")

        comparable = {k: v for k, v in product.items() if k != "id"}
        if name in name_to_product:
            existing = name_to_product[name]
            existing_comparable = {k: v for k, v in existing.items()
                                   if k not in {"id", "source_ids"}}
            if comparable != existing_comparable:
                raise ValueError(
                    f"Ambiguous product name {name!r}: duplicate catalogue rows "
                    "have different attributes. Survey baskets must be mapped to "
                    "an explicit SKU id before simulation."
                )
            source_ids[name].append(product_id)
            continue

        name_to_product[name] = product
        source_ids[name] = [product_id]
        canonical.append(product)

    for product in canonical:
        product["source_ids"] = source_ids[product["name"]]

    return canonical, {p["name"]: str(p["id"]) for p in canonical}


def _clean_basket(raw_basket: list, product_name_to_id: dict[str, str] | set) -> list:
    """
    Return only items whose productName is present in the Unity catalogue.
    Items absent from the catalogue are not simulated.
    """
    # Backwards compatibility for callers that only provide a set of names.
    if isinstance(product_name_to_id, set):
        product_name_to_id = {name: name for name in product_name_to_id}

    clean_by_product: dict[str, dict] = {}
    for item in raw_basket:
        name = item.get("productName", "")
        if name not in product_name_to_id:
            continue
        product_id = product_name_to_id[name]
        quantity = max(1, int(item.get("quantity", 1)))
        price = float(item.get("price", 1.0))
        cleaned = {
            "product_id":     product_id,
            "product_name":   name,
            "quantity":       quantity,
            "price":          price,
            "category":       item.get("category", "Unknown"),
            "fat_content":    float(item.get("fatContent", 0.0)),
            "is_bio":         bool(item.get("isBio", False)),
            "is_plant_based": bool(item.get("isPlantBased", False)),
        }
        if product_id not in clean_by_product:
            cleaned["_line_value"] = price * quantity
            clean_by_product[product_id] = cleaned
            continue
        existing = clean_by_product[product_id]
        for key in (
            "product_name", "category", "fat_content", "is_bio",
            "is_plant_based",
        ):
            if existing[key] != cleaned[key]:
                raise ValueError(
                    f"Conflicting duplicate basket rows for product {name!r}: "
                    f"field {key!r} differs within one shopping occasion."
                )
        existing["quantity"] += quantity
        existing["_line_value"] += price * quantity
        existing["price"] = existing["_line_value"] / existing["quantity"]

    clean = []
    for item in clean_by_product.values():
        item.pop("_line_value", None)
        clean.append(item)
    return clean


# ---------------------------------------------------------------------------
# 6. Build enriched real profiles
# ---------------------------------------------------------------------------

def build_real_profiles(
    firebase_data: dict,
    product_name_to_id: dict[str, str] | set,
    dce_rows: list[dict] | None = None,
) -> tuple[list, dict]:
    """
    Returns (profiles_list, summary_stats_dict).
    Each profile is a dict fully describing one real participant.

    Handles two Firebase export formats:
      - Flat   : {participantId: {age, round1_Basket, …}, …}
      - Nested : {"participants": {participantId: {…}, …}}
    """
    # Unwrap nested format produced by Firebase "Export JSON" button
    if "participants" in firebase_data and isinstance(firebase_data["participants"], dict):
        firebase_data = firebase_data["participants"]

    profiles = []
    n_skipped         = 0
    n_products_dropped = 0
    clean_dce_preferences = _clean_dce_preference_map(dce_rows)

    for user_id, user_data in firebase_data.items():
        demo      = extract_demographics(user_data)
        r1_raw    = user_data.get("round1_Basket", [])
        r2_raw    = user_data.get("round2_Basket", [])
        r1_total  = float(user_data.get("round1_TotalSpent", 0.0))
        r2_max    = float(user_data.get("round2_MaxBudget", r1_total))

        clean_r1 = _clean_basket(r1_raw, product_name_to_id)
        clean_r2 = _clean_basket(r2_raw, product_name_to_id)
        has_crisis_observation = bool(clean_r2)
        r1_prices = {i["product_name"]: i["price"] for i in clean_r1}
        r2_prices = {i["product_name"]: i["price"] for i in clean_r2}
        repeated_price_ratios = [
            r2_prices[name] / r1_prices[name]
            for name in r1_prices.keys() & r2_prices.keys()
            if r1_prices[name] > 0
        ]
        observed_price_shock = (
            float(np.median(repeated_price_ratios) - 1.0)
            if repeated_price_ratios else None
        )

        # Track how many basket items were dropped
        n_products_dropped += (len(r1_raw) - len(clean_r1)) + (len(r2_raw) - len(clean_r2))

        if not clean_r1:
            n_skipped += 1
            continue

        # If round2 basket is entirely out-of-catalogue, fall back to round1
        if not clean_r2:
            clean_r2 = clean_r1

        dce_prefs = clean_dce_preferences.get(
            str(user_id),
            parse_dce_choices(user_data.get("choiceExperiment1_Results", [])),
        )
        q_scores   = parse_questionnaire(user_data.get("questionnaireRatings", []))
        # Behavioural targets use the same canonical, catalogue-linked basket
        # universe as the ABM. Raw out-of-catalogue and repeated scene rows must
        # not enter the target while being impossible for the model to reproduce.
        elasticity = compute_price_elasticity(clean_r1, clean_r2, r1_total, r2_max)

        profile = {
            "source_id":           user_id,
            "is_real":             True,
            **demo,
            "baseline_basket":     clean_r1,
            "crisis_basket":       clean_r2,
            "budget":              float(r1_total),
            "crisis_budget":       float(r2_max),
            "has_crisis_observation": has_crisis_observation,
            # DCE-derived preferences
            "finnish_preference":  float(dce_prefs["finnish_preference"]),
            "organic_preference":  float(dce_prefs["organic_preference"]),
            "preferred_fat":       float(dce_prefs["preferred_fat"]),
            # Pre-crisis DCE signal. Cross-fitted phase-transition calibration
            # below replaces these provisional behavioural parameters without
            # using a participant's own phase-two outcome as their predictor.
            "dce_cheaper_bundle_choice_rate": float(
                dce_prefs["dce_cheaper_bundle_choice_rate"]
            ),
            "dce_price_sensitivity": float(
                dce_prefs["dce_cheaper_bundle_choice_rate"]
            ),
            "dce_optout_rate": float(dce_prefs.get("dce_optout_rate", 0.0)),
            "dce_preferences_use_recorded_prices": bool(
                dce_prefs.get("dce_preferences_use_recorded_prices", False)
            ),
            "price_sensitivity": float(
                dce_prefs["dce_cheaper_bundle_choice_rate"]
            ),
            "substitution_rate":   0.5,
            # Phase-two outcomes are validation targets, never simulation inputs.
            "observed_spending_reduction": float(elasticity["spending_reduction"]),
            "observed_budget_utilization": float(elasticity["budget_utilization"]),
            "observed_substitution_rate": float(elasticity["substitution_rate"]),
            "observed_substitution_lines": int(elasticity["substitution_lines"]),
            "observed_phase2_choice_lines": int(elasticity["phase2_choice_lines"]),
            "baseline_choice_lines": int(elasticity["baseline_choice_lines"]),
            "observed_quantity_retention": float(elasticity["quantity_retention"]),
            "observed_round2_spend": float(elasticity["round2_actual_spend"]),
            "observed_price_shock": observed_price_shock,
            # Questionnaire factor scores (q_price, q_health, q_environment, …)
            **q_scores,
            # Reference price = mean price of round1 purchases (for utility calibration)
            "reference_price":     float(np.mean([i["price"] for i in clean_r1])) if clean_r1 else 1.5,
            "archetype":           None,  # assigned by assign_archetypes()
            "cluster_id":          -1,
        }
        profiles.append(profile)

    stats = {
        "n_real":             len(profiles),
        "n_skipped":          n_skipped,
        "n_products_dropped": n_products_dropped,
    }
    return profiles, stats


# ---------------------------------------------------------------------------
# 7. Archetype assignment via K-Means
# ---------------------------------------------------------------------------

ARCHETYPE_FEATURE_KEYS = [
    "q_price", "q_health", "q_environment", "q_animal_welfare", "q_sensory_habit",
    "organic_preference", "finnish_preference", "preferred_fat",
]


def _archetype_feature_matrix(profiles: list[dict]) -> np.ndarray:
    """Use observed constructs/attributes only; exclude the pseudo-price DCE score."""
    return np.asarray([
        [float(profile.get(key, 0.5 if key != "preferred_fat" else 1.5))
         for key in ARCHETYPE_FEATURE_KEYS]
        for profile in profiles
    ], dtype=float)


def archetype_stability_diagnostics(
    profiles: list[dict],
    selected_k: int = 4,
    random_state: int = 42,
    n_bootstrap: int = 100,
) -> dict:
    """Compare k solutions and quantify assignment stability under resampling.

    The thresholds are conservative reporting gates, not universal truths. A
    categorical archetype is operational only when the requested k is the best
    silhouette solution, has adequate separation and bootstrap agreement, and
    contains no tiny cluster.
    """
    n = len(profiles)
    if n < 10:
        return {
            "status": "insufficient_data", "n_profiles": n,
            "selected_k": selected_k, "archetypes_supported": False,
            "candidates": [],
        }
    raw = _archetype_feature_matrix(profiles)
    X = StandardScaler().fit_transform(raw)
    rng = np.random.default_rng(random_state)
    candidates = []
    max_k = min(6, n - 1)
    for k in range(2, max_k + 1):
        base = KMeans(n_clusters=k, random_state=random_state, n_init=25)
        base_labels = base.fit_predict(X)
        sizes = np.bincount(base_labels, minlength=k)
        silhouette = float(silhouette_score(X, base_labels))
        agreements = []
        for bootstrap_id in range(n_bootstrap):
            draw = rng.integers(0, n, size=n)
            if len(np.unique(draw)) < k:
                continue
            boot = KMeans(
                n_clusters=k,
                random_state=random_state + bootstrap_id + 1,
                n_init=10,
            ).fit(X[draw])
            agreements.append(adjusted_rand_score(base_labels, boot.predict(X)))
        candidates.append({
            "k": k,
            "silhouette": round(silhouette, 4),
            "bootstrap_ari_median": round(float(np.median(agreements)), 4) if agreements else None,
            "bootstrap_ari_p10": round(float(np.quantile(agreements, 0.10)), 4) if agreements else None,
            "minimum_cluster_size": int(np.min(sizes)),
            "minimum_cluster_share": round(float(np.min(sizes) / n), 4),
        })
    recommended = max(candidates, key=lambda row: row["silhouette"])
    selected = next((row for row in candidates if row["k"] == selected_k), None)
    supported = bool(
        selected is not None
        and selected_k == recommended["k"]
        and selected["silhouette"] >= 0.25
        and (selected["bootstrap_ari_median"] or -1.0) >= 0.75
        and selected["minimum_cluster_size"] >= max(5, int(math.ceil(0.05 * n)))
        and selected_k <= len(ARCHETYPE_LABELS)
    )
    return {
        "status": "ok",
        "n_profiles": n,
        "selected_k": selected_k,
        "recommended_k": int(recommended["k"]),
        "archetypes_supported": supported,
        "candidates": candidates,
        "feature_keys": ARCHETYPE_FEATURE_KEYS,
        "gate": {
            "selected_is_recommended": bool(selected_k == recommended["k"]),
            "silhouette_at_least_0_25": bool(selected and selected["silhouette"] >= 0.25),
            "median_ari_at_least_0_75": bool(selected and (selected["bootstrap_ari_median"] or -1.0) >= 0.75),
            "minimum_cluster_size": max(5, int(math.ceil(0.05 * n))),
        },
    }


def assign_archetypes(
    profiles: list,
    n_clusters: int = 4,
    random_state: int = 42,
    operational: bool = True,
) -> list:
    """
    Cluster real profiles into n_clusters archetypes using
    questionnaire factors + behavioural scores.

    Clusters are labelled by inspecting which dimension dominates the centroid:
      price_champion  – driven by low price
      green_buyer     – driven by environment / animal welfare / organic
      health_optimizer – driven by health / nutrition
      habitual_buyer  – driven by familiarity / sensory habit
    """
    if len(profiles) < n_clusters:
        for p in profiles:
            p["exploratory_archetype"] = ARCHETYPE_LABELS[0]
            p["archetype"] = ARCHETYPE_LABELS[0] if operational else "continuous_profile"
            p["archetype_operational"] = bool(operational)
            p["cluster_id"] = 0
        return profiles

    X_raw = _archetype_feature_matrix(profiles)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    labels = km.fit_predict(X)

    # Map each cluster to a label based on which composite score is highest
    centroids_raw = scaler.inverse_transform(km.cluster_centers_)
    label_map = {}
    for cid in range(n_clusters):
        c = centroids_raw[cid]
        # Composite scores per archetype dimension
        # indices: price(0), health(1), env(2), animal(3), sensory(4), organic(5), finnish(6), fat(7)
        price_score  = c[0]
        green_score  = c[2] + c[3] + c[5]
        health_score = c[1]
        habit_score  = c[4]
        dominant_idx = int(np.argmax([price_score, green_score, health_score, habit_score]))
        # Guard against duplicate assignments (just cycle through labels)
        candidate = ARCHETYPE_LABELS[dominant_idx]
        used = list(label_map.values())
        if candidate in used:
            for fallback in ARCHETYPE_LABELS:
                if fallback not in used:
                    candidate = fallback
                    break
        label_map[cid] = candidate

    for i, p in enumerate(profiles):
        exploratory_label = label_map[int(labels[i])]
        p["exploratory_archetype"] = exploratory_label
        p["archetype"] = exploratory_label if operational else "continuous_profile"
        p["archetype_operational"] = bool(operational)
        p["cluster_id"] = int(labels[i])

    return profiles


# ---------------------------------------------------------------------------
# 8. Cross-fitted behavioural calibration
# ---------------------------------------------------------------------------

def _calibration_features(profile: dict) -> list[float]:
    """Pre-crisis predictors only; phase-two outcomes are deliberately excluded."""
    basket = profile.get("baseline_basket", [])
    units = sum(float(i.get("quantity", 1)) for i in basket)
    spend = sum(
        float(i.get("price", 0)) * float(i.get("quantity", 1)) for i in basket
    )
    organic_units = sum(
        float(i.get("quantity", 1)) for i in basket if i.get("is_bio", False)
    )
    plant_units = sum(
        float(i.get("quantity", 1))
        for i in basket if i.get("is_plant_based", False)
    )
    archetype = profile.get("archetype", "")
    return [
        float(profile.get(
            "dce_cheaper_bundle_choice_rate",
            profile.get("dce_price_sensitivity", 0.5),
        )),
        float(profile.get("finnish_preference", 0.5)),
        float(profile.get("organic_preference", 0.2)),
        float(profile.get("preferred_fat", 1.5)),
        float(profile.get("q_price", 0.5)),
        float(profile.get("q_health", 0.5)),
        float(profile.get("q_environment", 0.5)),
        float(profile.get("q_animal_welfare", 0.5)),
        float(profile.get("q_sensory_habit", 0.5)),
        float(profile.get("age", 35)),
        float(profile.get("income_midpoint", 2500)),
        float(profile.get("household_size", 2)),
        float(profile.get("children", 0)),
        units,
        spend,
        organic_units / max(1.0, units),
        plant_units / max(1.0, units),
        *[float(archetype == label) for label in ARCHETYPE_LABELS],
    ]


def _nested_substitution_action_probabilities(
    profiles: list[dict],
    observed: list[int],
    X_all: np.ndarray,
    train_pos: np.ndarray,
    random_state: int,
    n_repeats: int = 10,
) -> tuple[np.ndarray, dict]:
    """Translate observed replacement rates into ABM line-action probabilities.

    The survey rate uses phase-two purchased rows as its denominator, whereas the
    ABM asks whether to seek a replacement once for every baseline basket row.
    A direct rate-to-probability assignment therefore changes the estimand.  This
    routine estimates the translation using only the fixed calibration cohort.

    Repeated outer folds supply honest predictions for calibration participants.
    Within each outer training fold, a second set of folds estimates the scale
    between predicted phase-two replacement rates and baseline opportunities.
    The untouched fixed validation cohort is predicted only after model selection
    and scale estimation are complete.
    """
    X_observed = X_all[observed]
    y_rate = np.asarray([
        np.clip(profiles[i].get("observed_substitution_rate", 0.0), 0.0, 1.0)
        for i in observed
    ], dtype=float)

    baseline_lines = np.asarray([
        max(1, int(p.get("baseline_choice_lines", len(p.get("baseline_basket", [])))))
        for p in (profiles[i] for i in observed)
    ], dtype=float)
    phase2_lines = np.asarray([
        max(0, int(p.get(
            "observed_phase2_choice_lines", len(p.get("crisis_basket", []))
        )))
        for p in (profiles[i] for i in observed)
    ], dtype=float)
    observed_counts = np.asarray([
        max(0.0, float(p.get(
            "observed_substitution_lines",
            round(float(p.get("observed_substitution_rate", 0.0)) * phase2_lines[pos]),
        )))
        for pos, p in enumerate(profiles[i] for i in observed)
    ], dtype=float)

    train_pos = np.asarray(train_pos, dtype=int)
    X_train = X_observed[train_pos]
    y_train = y_rate[train_pos]
    base_train = baseline_lines[train_pos]
    count_train = observed_counts[train_pos]
    n_train = len(train_pos)
    action_sum = np.zeros(n_train, dtype=float)
    naive_sum = np.zeros(n_train, dtype=float)
    rate_sum = np.zeros(n_train, dtype=float)
    prediction_count = np.zeros(n_train, dtype=int)

    def _model() -> object:
        return make_pipeline(StandardScaler(), Ridge(alpha=5.0))

    for repeat in range(n_repeats):
        outer = KFold(
            n_splits=min(5, n_train), shuffle=True,
            random_state=random_state + repeat,
        )
        for outer_train, outer_test in outer.split(X_train):
            # Inner OOF rate predictions make the denominator scale independent
            # of the outer-test outcomes and reduce in-sample optimism.
            inner_rate = np.zeros(len(outer_train), dtype=float)
            inner = KFold(
                n_splits=min(4, len(outer_train)), shuffle=True,
                random_state=random_state + 100 + repeat,
            )
            for inner_train, inner_test in inner.split(outer_train):
                fitted = _model()
                fitted.fit(
                    X_train[outer_train[inner_train]],
                    y_train[outer_train[inner_train]],
                )
                inner_rate[inner_test] = np.clip(fitted.predict(
                    X_train[outer_train[inner_test]]
                ), 0.0, 1.0)

            denominator = float(np.sum(
                inner_rate * base_train[outer_train]
            ))
            scale = (
                float(np.sum(count_train[outer_train])) / denominator
                if denominator > 1e-12 else 0.0
            )
            scale = float(np.clip(scale, 0.0, 2.0))

            fitted = _model()
            fitted.fit(X_train[outer_train], y_train[outer_train])
            heldout_rate = np.clip(
                fitted.predict(X_train[outer_test]), 0.0, 1.0
            )
            heldout_action = np.clip(heldout_rate * scale, 0.0, 1.0)
            naive_action = float(np.clip(
                np.sum(count_train[outer_train]) /
                max(1.0, np.sum(base_train[outer_train])),
                0.0, 1.0,
            ))
            action_sum[outer_test] += heldout_action
            naive_sum[outer_test] += naive_action
            rate_sum[outer_test] += heldout_rate
            prediction_count[outer_test] += 1

    divisor = np.maximum(1, prediction_count)
    crossfit_action = action_sum / divisor
    crossfit_naive = naive_sum / divisor
    crossfit_rate = rate_sum / divisor
    normalized_error = np.maximum(1.0, base_train)
    action_mae = float(np.mean(
        np.abs(count_train - crossfit_action * base_train) / normalized_error
    ))
    naive_mae = float(np.mean(
        np.abs(count_train - crossfit_naive * base_train) / normalized_error
    ))
    retained = bool(action_mae < naive_mae)

    # Final translation for profiles outside the calibration cohort. No fixed
    # validation outcome participates in either this scale or the fitted model.
    full_denominator = float(np.sum(crossfit_rate * base_train))
    full_scale = (
        float(np.sum(count_train)) / full_denominator
        if full_denominator > 1e-12 else 0.0
    )
    full_scale = float(np.clip(full_scale, 0.0, 2.0))
    final_model = _model()
    final_model.fit(X_train, y_train)
    final_rate = np.clip(final_model.predict(X_all), 0.0, 1.0)
    action_probability = np.clip(final_rate * full_scale, 0.0, 1.0)
    global_naive = float(np.clip(
        np.sum(count_train) / max(1.0, np.sum(base_train)), 0.0, 1.0
    ))
    if retained:
        for local_pos, observed_pos in enumerate(train_pos):
            action_probability[observed[observed_pos]] = crossfit_action[local_pos]
    else:
        action_probability[:] = global_naive
        for local_pos, observed_pos in enumerate(train_pos):
            action_probability[observed[observed_pos]] = crossfit_naive[local_pos]

    diagnostics = {
        "substitution_action_method": "fixed_holdout_repeated_10x_nested_participant_cv",
        "substitution_action_model_retained": retained,
        "substitution_action_mae": round(action_mae, 4),
        "substitution_action_naive_mae": round(naive_mae, 4),
        "substitution_action_skill": round(
            1.0 - action_mae / naive_mae, 4
        ) if naive_mae > 1e-12 else 0.0,
        "substitution_action_scale": round(full_scale, 4),
        "substitution_action_naive_probability": round(global_naive, 4),
        "substitution_action_training_observed_count": int(np.sum(count_train)),
        "substitution_action_training_baseline_lines": int(np.sum(base_train)),
    }
    return action_probability, diagnostics


def calibrate_behavioral_profiles(
    profiles: list[dict], random_state: int = 42,
) -> tuple[list[dict], dict]:
    """Estimate phase-two response traits without participant-level leakage.

    Five-fold cross-fitting gives every observed participant a prediction from a
    model that was not trained on that participant. A separate fixed 80/20 split
    supplies honest diagnostic errors against a naive training-mean benchmark.
    """
    observed = [
        i for i, p in enumerate(profiles) if p.get("has_crisis_observation", False)
    ]
    if len(observed) < 10:
        return profiles, {
            "status": "insufficient_data", "n_observed": len(observed)
        }

    X_all = np.asarray([_calibration_features(p) for p in profiles], dtype=float)
    targets = {
        "price_response": np.asarray([
            np.clip(1.0 - profiles[i]["observed_quantity_retention"], 0.0, 1.0)
            for i in observed
        ]),
        "substitution": np.asarray([
            np.clip(profiles[i]["observed_substitution_rate"], 0.0, 1.0)
            for i in observed
        ]),
        "hoarding": np.asarray([
            np.clip(profiles[i]["observed_quantity_retention"] - 1.0, 0.0, 1.0)
            for i in observed
        ]),
        "budget_utilization": np.asarray([
            np.clip(profiles[i].get("observed_budget_utilization", 1.0), 0.0, 1.0)
            for i in observed
        ]),
    }
    X_observed = X_all[observed]
    train_pos, validation_pos = train_test_split(
        np.arange(len(observed)), test_size=0.20, random_state=random_state
    )

    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, float | int | str] = {
        "status": "ok",
        "n_observed": len(observed),
        "n_train": len(train_pos),
        "n_validation": len(validation_pos),
        "method": "ridge_5fold_crossfit",
    }

    for name, y in targets.items():
        holdout_model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        holdout_model.fit(X_observed[train_pos], y[train_pos])
        holdout_pred = np.clip(
            holdout_model.predict(X_observed[validation_pos]), 0.0, 1.0
        )
        holdout_y = y[validation_pos]
        naive = float(np.mean(y[train_pos]))
        mae = float(np.mean(np.abs(holdout_y - holdout_pred)))
        naive_mae = float(np.mean(np.abs(holdout_y - naive)))
        diagnostics[f"{name}_mae"] = round(mae, 4)
        diagnostics[f"{name}_naive_mae"] = round(naive_mae, 4)
        diagnostics[f"{name}_skill"] = round(
            1.0 - mae / naive_mae, 4
        ) if naive_mae > 1e-12 else 0.0
        use_predictive_model = mae < naive_mae
        diagnostics[f"{name}_model_retained"] = bool(use_predictive_model)

        crossfit = np.zeros(len(observed), dtype=float)
        folds = KFold(
            n_splits=min(5, len(observed)), shuffle=True,
            random_state=random_state,
        )
        for fold_train, fold_test in folds.split(X_observed):
            if use_predictive_model:
                fold_model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
                fold_model.fit(X_observed[fold_train], y[fold_train])
                crossfit[fold_test] = np.clip(
                    fold_model.predict(X_observed[fold_test]), 0.0, 1.0
                )
            else:
                crossfit[fold_test] = float(np.mean(y[fold_train]))

        if use_predictive_model:
            full_model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
            full_model.fit(X_observed, y)
            all_predictions = np.clip(full_model.predict(X_all), 0.0, 1.0)
        else:
            all_predictions = np.full(len(profiles), float(np.mean(y)))
        all_predictions[observed] = crossfit
        predictions[name] = all_predictions

    action_probabilities, action_diagnostics = (
        _nested_substitution_action_probabilities(
            profiles, observed, X_all, train_pos, random_state
        )
    )
    diagnostics.update(action_diagnostics)

    for i, profile in enumerate(profiles):
        profile["price_response_intensity"] = float(predictions["price_response"][i])
        profile["price_sensitivity"] = float(predictions["price_response"][i])
        profile["substitution_rate"] = float(predictions["substitution"][i])
        profile["substitution_action_probability"] = float(
            action_probabilities[i]
        )
        profile["sub_tolerance"] = float(action_probabilities[i])
        profile["hoarding_propensity"] = float(predictions["hoarding"][i])
        profile["budget_utilization_propensity"] = float(
            predictions["budget_utilization"][i]
        )
        profile["calibration_prediction_is_cross_fitted"] = bool(i in observed)

    observed_shocks = [
        p["observed_price_shock"] for p in profiles
        if p.get("observed_price_shock") is not None
    ]
    median_shock = float(np.median(observed_shocks)) if observed_shocks else 0.25

    def _predict_retention(profile: dict, margin: float) -> float:
        sensitivity = float(profile.get("price_sensitivity", 0.5))
        # Match the empirical-only ABM rule exactly. At the reference price the
        # disutility is ``sensitivity``; after a proportional price shock it is
        # ``sensitivity * (1 + shock)``. The selected-SKU reservation margin is
        # therefore overcome when sensitivity * shock exceeds the margin.
        # TPB and Prospect Theory are optional extensions and must not enter the
        # default calibration target.
        shock_loss = sensitivity * median_shock
        if margin < shock_loss:
            return 0.0

        budget = round(float(profile.get("crisis_budget", profile.get("budget", 0))), 2)
        wanted_units = bought_units = 0
        for item in profile.get("baseline_basket", []):
            qty = max(1, int(item.get("quantity", 1)))
            unit_price = round(float(item.get("price", 0)) * (1.0 + median_shock), 2)
            wanted_units += qty
            if unit_price <= 0:
                continue
            affordable = int(math.floor((budget + 1e-9) / unit_price))
            bought = min(qty, affordable)
            bought_units += bought
            budget = round(budget - bought * unit_price, 2)
        return bought_units / max(1, wanted_units)

    train_profile_indices = [observed[pos] for pos in train_pos]
    validation_profile_indices = [observed[pos] for pos in validation_pos]
    for profile in profiles:
        profile["phase2_calibration_role"] = "not_observed"
    for index in train_profile_indices:
        profiles[index]["phase2_calibration_role"] = "training"
    for index in validation_profile_indices:
        profiles[index]["phase2_calibration_role"] = "validation"
    train_target = float(np.mean([
        profiles[i]["observed_quantity_retention"] for i in train_profile_indices
    ]))
    margin_grid = np.linspace(0.0, 0.25, 101)
    margin_errors = []
    for margin in margin_grid:
        predicted_mean = float(np.mean([
            _predict_retention(profiles[i], float(margin))
            for i in train_profile_indices
        ]))
        margin_errors.append(abs(predicted_mean - train_target))
    calibrated_margin = float(margin_grid[int(np.argmin(margin_errors))])
    for profile in profiles:
        profile["revealed_preference_margin"] = calibrated_margin

    validation_retention_pred = np.asarray([
        _predict_retention(profiles[i], calibrated_margin)
        for i in validation_profile_indices
    ])
    validation_retention_obs = np.asarray([
        profiles[i]["observed_quantity_retention"]
        for i in validation_profile_indices
    ])
    diagnostics["revealed_preference_margin"] = round(calibrated_margin, 4)
    diagnostics["retention_decision_rule"] = "empirical_relative_price_ratio"
    diagnostics["retention_validation_mae"] = round(float(np.mean(
        np.abs(validation_retention_obs - validation_retention_pred)
    )), 4)
    retention_naive_mae = float(np.mean(np.abs(
        validation_retention_obs - train_target
    )))
    diagnostics["retention_validation_naive_mae"] = round(
        retention_naive_mae, 4
    )
    diagnostics["retention_validation_skill"] = round(
        1.0 - diagnostics["retention_validation_mae"] / retention_naive_mae,
        4,
    ) if retention_naive_mae > 1e-12 else 0.0
    diagnostics["retention_validation_observed_mean"] = round(
        float(np.mean(validation_retention_obs)), 4
    )
    diagnostics["retention_validation_predicted_mean"] = round(
        float(np.mean(validation_retention_pred)), 4
    )

    diagnostics["passes_price_naive_benchmark"] = bool(
        diagnostics["price_response_skill"] > 0
    )
    diagnostics["passes_substitution_naive_benchmark"] = bool(
        diagnostics["substitution_skill"] > 0
    )
    diagnostics["observed_median_price_shock"] = round(
        median_shock, 4
    ) if observed_shocks else None
    diagnostics["observed_mean_quantity_retention"] = round(float(np.mean([
        p["observed_quantity_retention"] for p in profiles
        if p.get("has_crisis_observation", False)
    ])), 4)
    return profiles, diagnostics


# ---------------------------------------------------------------------------
# 9. Observed replacement-choice audit
# ---------------------------------------------------------------------------

def substitution_choice_diagnostics(
    profiles: list[dict], products: list[dict],
) -> dict:
    """Test candidate-screening and ranking claims against observed replacements.

    A replacement event is deliberately narrow: within one catalogue category,
    phase one contains exactly one SKU absent from phase two and phase two contains
    exactly one SKU absent from phase one. Quantity changes, additions without a
    removal, and ambiguous many-to-many changes are not labelled as choices.

    The audit is diagnostic rather than a fitted choice model. It asks whether the
    phase-two retention price threshold transfers to replacement SKUs and whether
    the current participant-compatibility ordering beats a leave-one-event-out
    category-popularity benchmark. Conservative gates prevent sparse descriptive
    patterns from becoming deterministic ABM rules.
    """
    product_map = {
        str(product.get("id", "")): product
        for product in products if str(product.get("id", ""))
    }
    category_products: dict[str, list[dict]] = {}
    for product in products:
        category = str(product.get("category", "Unknown")).strip()
        category_products.setdefault(category, []).append(product)

    events: list[dict] = []
    transition_events: list[dict] = []
    for profile in profiles:
        if not profile.get("has_crisis_observation", False):
            continue
        phase_one: dict[str, dict[str, dict]] = {}
        phase_two: dict[str, dict[str, dict]] = {}
        for item in profile.get("baseline_basket", []):
            category = str(item.get("category", "Unknown")).strip()
            phase_one.setdefault(category, {})[str(item.get("product_id", ""))] = item
        for item in profile.get("crisis_basket", []):
            category = str(item.get("category", "Unknown")).strip()
            phase_two.setdefault(category, {})[str(item.get("product_id", ""))] = item
        for category in phase_one.keys() & phase_two.keys():
            phase_one_qty = {
                sku: max(0.0, float(item.get("quantity", 0)))
                for sku, item in phase_one[category].items()
            }
            phase_two_qty = {
                sku: max(0.0, float(item.get("quantity", 0)))
                for sku, item in phase_two[category].items()
            }
            removed_qty = {
                sku: max(0.0, quantity - phase_two_qty.get(sku, 0.0))
                for sku, quantity in phase_one_qty.items()
                if quantity > phase_two_qty.get(sku, 0.0)
            }
            added_qty = {
                sku: max(0.0, quantity - phase_one_qty.get(sku, 0.0))
                for sku, quantity in phase_two_qty.items()
                if quantity > phase_one_qty.get(sku, 0.0)
            }
            removed_total = sum(removed_qty.values())
            added_total = sum(added_qty.values())
            replacement_mass = min(removed_total, added_total)
            if replacement_mass > 0 and added_total > 0:
                for target_id, quantity in added_qty.items():
                    if target_id in product_map:
                        transition_events.append({
                            "category": category,
                            "target_id": target_id,
                            "source_ids": sorted(removed_qty),
                            "weight": replacement_mass * quantity / added_total,
                            "role": profile.get("phase2_calibration_role", "unknown"),
                        })
            removed = set(phase_one[category]) - set(phase_two[category])
            added = set(phase_two[category]) - set(phase_one[category])
            if len(removed) != 1 or len(added) != 1:
                continue
            source_id = next(iter(removed))
            target_id = next(iter(added))
            if source_id not in product_map or target_id not in product_map:
                continue
            events.append({
                "category": category,
                "source_id": source_id,
                "target_id": target_id,
                "source_item": phase_one[category][source_id],
                "target_item": phase_two[category][target_id],
                "profile": profile,
            })

    if not events:
        return {
            "status": "insufficient_data",
            "n_unambiguous_events": 0,
            "candidate_price_gate_supported": False,
            "supported_ranking_categories": [],
            "operational_fallback": "seeded_uniform_affordable_same_category",
        }

    target_counts: dict[str, dict[str, int]] = {}
    for event in events:
        counts = target_counts.setdefault(event["category"], {})
        target = event["target_id"]
        counts[target] = counts.get(target, 0) + 1

    rows = []
    all_price_covered = 0
    for category in sorted({event["category"] for event in events}):
        category_events = [event for event in events if event["category"] == category]
        top1 = top3 = price_covered = 0
        reciprocal_ranks: list[float] = []
        loo_hits = 0
        candidate_counts: list[int] = []

        for event in category_events:
            profile = event["profile"]
            source_price = max(float(event["source_item"].get("price", 0)), 0.01)
            sensitivity = float(profile.get("price_sensitivity", 0.5))
            margin = float(profile.get("revealed_preference_margin", 0.0))
            candidates = [
                product for product in category_products.get(category, [])
                if str(product.get("id", "")) != event["source_id"]
            ]
            candidate_counts.append(len(candidates))
            # Use the price actually recorded in phase two for the chosen target,
            # not its catalogue baseline price. This reproduces the rule under the
            # experimental price shock rather than silently validating it at normal
            # prices.
            target_price = float(event["target_item"].get("price", 0))
            target_eligible = bool(
                target_price > 0
                and sensitivity * (target_price / source_price - 1.0)
                <= margin + 1e-12
            )
            price_covered += int(target_eligible)
            all_price_covered += int(target_eligible)

            def compatibility(product: dict) -> float:
                if category.strip().casefold() != "milk":
                    return 0.5
                finnish = float(profile.get("finnish_preference", 0.5))
                organic = float(profile.get("organic_preference", 0.5))
                preferred_fat = float(profile.get("preferred_fat", 1.5))
                origin_fit = finnish if product.get("origin") == "Suomi" else 1.0 - finnish
                organic_fit = (
                    organic if product.get("is_bio", False) else 1.0 - organic
                )
                fat_fit = math.exp(
                    -abs(float(product.get("fat_content", 0)) - preferred_fat) / 2.0
                )
                return (origin_fit + organic_fit + fat_fit) / 3.0

            # Ranking is assessed over the catalogue category independently of
            # the price screen. This prevents a failed screen from mechanically
            # depressing (or improving) the attribute-ranking diagnostic.
            ranked = sorted(
                candidates,
                key=lambda product: (
                    -compatibility(product),
                    float(product.get("price", 0)),
                    str(product.get("id", "")),
                ),
            )
            ranked_ids = [str(product.get("id", "")) for product in ranked]
            if event["target_id"] in ranked_ids:
                rank = ranked_ids.index(event["target_id"]) + 1
                top1 += int(rank == 1)
                top3 += int(rank <= 3)
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            loo_counts = dict(target_counts[category])
            loo_counts[event["target_id"]] -= 1
            if loo_counts[event["target_id"]] <= 0:
                del loo_counts[event["target_id"]]
            if loo_counts:
                popular = min(
                    loo_counts,
                    key=lambda product_id: (-loo_counts[product_id], product_id),
                )
                loo_hits += int(popular == event["target_id"])

        n_events = len(category_events)
        coverage = price_covered / n_events
        top1_rate = top1 / n_events
        popularity_rate = loo_hits / n_events
        ranking_supported = bool(
            n_events >= 30
            and top1_rate >= 0.25
            and top1_rate >= popularity_rate + 0.05
        )
        rows.append({
            "category": category,
            "n_events": n_events,
            "catalogue_skus": len(category_products.get(category, [])),
            "mean_candidates": round(float(np.mean(candidate_counts)), 1),
            "price_gate_target_coverage": round(coverage, 4),
            "ranking_top1_accuracy": round(top1_rate, 4),
            "ranking_top3_accuracy": round(top3 / n_events, 4),
            "ranking_mean_reciprocal_rank": round(
                float(np.mean(reciprocal_ranks)), 4
            ),
            "loo_category_popularity_accuracy": round(popularity_rate, 4),
            "ranking_supported": ranking_supported,
        })

    overall_coverage = all_price_covered / len(events)
    supported_categories = [
        row["category"] for row in rows if row["ranking_supported"]
    ]

    # The two shopping stages also identify category-level destination shares
    # without pretending that ambiguous many-to-many basket changes reveal an
    # exact source-target pairing. Fit target shares on the calibration cohort
    # and test them on the fixed phase-two validation participants.
    transition_rows = []
    transition_weights: dict[str, dict[str, float]] = {}
    supported_transition_categories: list[str] = []
    for category in sorted({event["category"] for event in transition_events}):
        training_events = [
            event for event in transition_events
            if event["category"] == category and event["role"] == "training"
        ]
        validation_events = [
            event for event in transition_events
            if event["category"] == category and event["role"] == "validation"
        ]
        counts: dict[str, float] = {}
        for event in training_events:
            counts[event["target_id"]] = (
                counts.get(event["target_id"], 0.0) + float(event["weight"])
            )
        training_mass = float(sum(event["weight"] for event in training_events))
        validation_mass = float(sum(event["weight"] for event in validation_events))
        model_loss = uniform_loss = 0.0
        top1_weight = 0.0
        for event in validation_events:
            candidates = [
                str(product.get("id", ""))
                for product in category_products.get(category, [])
                if str(product.get("id", "")) not in event["source_ids"]
            ]
            if event["target_id"] not in candidates or not candidates:
                continue
            alpha = 0.5
            denominator = sum(counts.get(candidate, 0.0) + alpha for candidate in candidates)
            probability = (
                counts.get(event["target_id"], 0.0) + alpha
            ) / max(denominator, 1e-12)
            uniform_probability = 1.0 / len(candidates)
            weight = float(event["weight"])
            model_loss -= weight * math.log(max(probability, 1e-12))
            uniform_loss -= weight * math.log(uniform_probability)
            most_likely = min(
                candidates,
                key=lambda sku: (-(counts.get(sku, 0.0) + alpha), sku),
            )
            top1_weight += weight * float(most_likely == event["target_id"])
        mean_model_loss = model_loss / validation_mass if validation_mass else None
        mean_uniform_loss = uniform_loss / validation_mass if validation_mass else None
        supported = bool(
            training_mass >= 10.0
            and validation_mass >= 3.0
            and mean_model_loss is not None
            and mean_uniform_loss is not None
            and mean_model_loss + 0.02 < mean_uniform_loss
        )
        if supported:
            supported_transition_categories.append(category)
            transition_weights[category] = {
                sku: round(float(weight), 6) for sku, weight in counts.items()
            }
        transition_rows.append({
            "category": category,
            "training_replacement_mass": round(training_mass, 3),
            "validation_replacement_mass": round(validation_mass, 3),
            "validation_log_loss": round(mean_model_loss, 4)
            if mean_model_loss is not None else None,
            "uniform_log_loss": round(mean_uniform_loss, 4)
            if mean_uniform_loss is not None else None,
            "validation_top1_share": round(top1_weight / validation_mass, 4)
            if validation_mass else None,
            "transition_model_supported": supported,
        })
    # Coverage of observed chosen targets measures sensitivity only. It cannot
    # establish specificity because the export does not record which rejected
    # alternatives were actually displayed/considered. Require a substantial
    # event base and very high sensitivity before transferring a retention rule
    # to candidate rejection; the present dataset deliberately fails this gate.
    price_gate_supported = bool(
        len(events) >= 100 and overall_coverage >= 0.90
    )
    return {
        "status": "ok",
        "n_unambiguous_events": len(events),
        "event_definition": "one_removed_and_one_added_sku_within_category",
        "validation_method": "event_level_leave_one_out_popularity_benchmark",
        "categories": rows,
        "candidate_price_gate_target_coverage": round(overall_coverage, 4),
        "candidate_price_gate_minimum_events": 100,
        "candidate_price_gate_minimum_coverage": 0.90,
        "candidate_price_gate_supported": price_gate_supported,
        "candidate_price_gate_limitation": (
            "Chosen-target coverage measures sensitivity only; rejected candidate "
            "sets are absent, so specificity is not identifiable."
        ),
        "ranking_gate": (
            "n>=30, top1>=0.25, and top1 exceeds "
            "leave-one-out category popularity by >=0.05"
        ),
        "supported_ranking_categories": supported_categories,
        "phase_transition_target_models": transition_rows,
        "supported_transition_categories": supported_transition_categories,
        "empirical_transition_weights": transition_weights,
        "transition_gate": (
            "training replacement mass>=10, validation replacement mass>=3, "
            "and held-out log loss improves on uniform by >=0.02"
        ),
        "operational_fallback": (
            "dce_mnl_for_milk_then_validated_phase_transition_target_shares_"
            "else_seeded_uniform_affordable_same_category"
        ),
        "caution": (
            "Events are sparse observational reconstructions from two shopping "
            "occasions, not randomized candidate choice sets. Same-category "
            "interchangeability remains a structural assumption."
        ),
    }


def summarize_baseline_observations(
    profiles: list[dict], products: list[dict],
) -> dict:
    """Summarise catalogue-linked phase-one shopping occasions for ABM checks.

    These are internal reproduction targets from the same data that initialise
    agents. They can reveal temporal scaling and stock-system distortions, but
    cannot provide independent external validation.
    """
    product_map = {
        str(product.get("id", "")): product for product in products
    }
    basket_units: list[float] = []
    basket_values: list[float] = []
    category_units: dict[str, float] = {}
    organic_units = domestic_units = total_units = 0.0
    for profile in profiles:
        units = value = 0.0
        for item in profile.get("baseline_basket", []):
            quantity = max(0.0, float(item.get("quantity", 0)))
            price = max(0.0, float(item.get("price", 0)))
            units += quantity
            value += quantity * price
            total_units += quantity
            category = str(item.get("category", "Unknown")).strip() or "Unknown"
            category_units[category] = category_units.get(category, 0.0) + quantity
            product = product_map.get(str(item.get("product_id", "")), {})
            organic_units += quantity * bool(
                item.get("is_bio", product.get("is_bio", False))
            )
            domestic_units += quantity * bool(product.get("origin") == "Suomi")
        basket_units.append(units)
        basket_values.append(value)

    n = len(basket_units)
    return {
        "status": "ok" if n else "insufficient_data",
        "n_shopping_occasions": n,
        "mean_linked_basket_units": round(float(np.mean(basket_units)), 4) if n else None,
        "mean_linked_basket_value": round(float(np.mean(basket_values)), 4) if n else None,
        "organic_unit_share": round(organic_units / total_units, 4) if total_units else None,
        "domestic_unit_share": round(domestic_units / total_units, 4) if total_units else None,
        "category_unit_shares": {
            category: round(units / total_units, 4)
            for category, units in sorted(category_units.items())
        } if total_units else {},
        "evidence_tier": "internal_reproduction_target",
        "occasion_unit": "observed_household_shopping_session",
        "household_size_treatment": (
            "not_rescaled; the recorded basket is already the household-level outcome"
        ),
        "visit_interval_identified": False,
        "temporal_conversion": (
            "Analyst-selected population and store traffic determine revisit interval; "
            "GROCERYsim does not observe inter-visit time."
        ),
    }


def summarize_phase2_holdout_targets(
    profiles: list[dict], calibration_stats: dict,
) -> dict:
    """Return one-occasion targets for the untouched phase-two holdout cohort."""
    holdout = [
        profile for profile in profiles
        if profile.get("phase2_calibration_role") == "validation"
    ]
    if not holdout:
        return {"status": "insufficient_data", "n_holdout": 0}
    training = [
        profile for profile in profiles
        if profile.get("phase2_calibration_role") == "training"
    ]

    metrics = {
        "quantity_retention": "observed_quantity_retention",
        "spending_reduction": "observed_spending_reduction",
        "budget_utilization": "observed_budget_utilization",
        "substitution_rate": "observed_substitution_rate",
    }
    targets = {}
    for metric, field in metrics.items():
        values = np.asarray([float(profile[field]) for profile in holdout], dtype=float)
        targets[metric] = {
            "mean": round(float(np.mean(values)), 4),
            "training_mean": round(float(np.mean([
                float(profile[field]) for profile in training
            ])), 4) if training else round(float(np.mean(values)), 4),
            "sd": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
            "n": len(values),
            # Declared internal diagnostic tolerance. It is intentionally broad
            # and must never be described as a preregistered confirmatory bound.
            "absolute_tolerance": 0.10,
            "individual_model_retained": bool(
                calibration_stats.get(
                    "substitution_action_model_retained"
                    if metric == "substitution_rate"
                    else "budget_utilization_model_retained"
                    if metric in {"spending_reduction", "budget_utilization"}
                    else "passes_price_naive_benchmark",
                    False,
                )
            ),
        }
    training_prices: dict[str, list[float]] = {}
    for profile in training:
        for item in profile.get("crisis_basket", []):
            product_id = str(item.get("product_id", ""))
            price = float(item.get("price", 0))
            if product_id and price > 0:
                training_prices.setdefault(product_id, []).append(price)
    price_overrides = {
        product_id: round(float(np.median(prices)), 4)
        for product_id, prices in training_prices.items()
    }
    return {
        "status": "ok",
        "n_holdout": len(holdout),
        "cohort": "fixed_20_percent_phase2_validation_split",
        "random_state": 42,
        "price_shock": calibration_stats.get("observed_median_price_shock"),
        "training_phase2_price_overrides": price_overrides,
        "n_skus_with_training_phase2_price": len(price_overrides),
        "metrics": targets,
        "evidence_tier": "calibration_holdout",
        "occasion_unit": "single_controlled_shopping_visit",
        "acceptance_rule": "model mean within absolute 0.10 of holdout mean",
        "caution": (
            "Internal holdout reproduction only. Training-cohort median phase-two "
            "SKU prices are used where available and the median shock fills uncovered "
            "SKUs; this cannot validate multi-day crisis dynamics."
        ),
    }


# ---------------------------------------------------------------------------
# 8. Complete-profile participant resampling
# ---------------------------------------------------------------------------

def _resampled_profile(template: dict, draw_index: int) -> dict:
    """Deep-copy one complete participant profile without fabricating attributes."""
    new = copy.deepcopy(template)
    empirical_id = str(template.get("empirical_source_id", template.get("source_id", "participant")))
    new["empirical_source_id"] = empirical_id
    new["source_id"] = f"{empirical_id}::draw_{draw_index}"
    new["is_real"] = False
    new["is_participant_resample"] = True
    new["resample_draw_index"] = int(draw_index)
    return new


def bootstrap_population(
    real_profiles: list,
    target_size: int,
    jitter_seed: int = 42,
) -> list:
    """
    Build a population pool of exactly `target_size` agents.

    Draw complete participant profiles with replacement. The legacy ``jitter_seed``
    name is retained for API compatibility, but no demographic, belief, preference,
    basket, quantity, or price value is perturbed.
    """
    if not real_profiles:
        return []
    if target_size < 1:
        raise ValueError("target_size must be positive")
    rng = np.random.default_rng(jitter_seed)
    draws = rng.integers(0, len(real_profiles), size=int(target_size))
    return [_resampled_profile(real_profiles[int(index)], draw_index)
            for draw_index, index in enumerate(draws)]


# ---------------------------------------------------------------------------
# 9. Public pipeline entry points
# ---------------------------------------------------------------------------

def run_pipeline_from_data(
    firebase_dict: dict,
    products_dict: dict,
    pool_size: int = 2000,
    n_archetypes: int = 4,
    dce_rows: list[dict] | None = None,
) -> dict:
    """
    Main entry point for the Streamlit app.
    Accepts pre-loaded dicts; returns mesa_config dict.
    """
    dce_choice_stats = calibrate_dce_choice_model(
        firebase_dict, dce_rows=dce_rows
    )
    questionnaire_stats = questionnaire_reliability(firebase_dict)
    canonical_products, product_name_to_id = canonicalize_products(
        products_dict.get("products", [])
    )

    real_profiles, parse_stats = build_real_profiles(
        firebase_dict, product_name_to_id, dce_rows=dce_rows
    )

    if not real_profiles:
        raise ValueError(
            "No valid profiles could be built — check that basket product names "
            "match those in master_products.json."
        )

    archetype_stats = archetype_stability_diagnostics(
        real_profiles, selected_k=n_archetypes,
    )
    real_profiles = assign_archetypes(
        real_profiles,
        n_clusters=n_archetypes,
        operational=archetype_stats.get("archetypes_supported", False),
    )
    real_profiles, calibration_stats = calibrate_behavioral_profiles(real_profiles)
    substitution_stats = substitution_choice_diagnostics(
        real_profiles, canonical_products,
    )
    baseline_targets = summarize_baseline_observations(
        real_profiles, canonical_products,
    )
    phase2_targets = summarize_phase2_holdout_targets(
        real_profiles, calibration_stats,
    )

    archetype_counts: dict[str, int] = {}
    exploratory_counts: dict[str, int] = {}
    for p in real_profiles:
        archetype = p.get("archetype", "continuous_profile")
        exploratory = p.get("exploratory_archetype", archetype)
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        exploratory_counts[exploratory] = exploratory_counts.get(exploratory, 0) + 1

    config = {
        "products":   canonical_products,
        # Keep only the observed empirical units in configuration. Every model
        # instance draws its household pool from these complete profiles using
        # its own fixed seed, propagating participant-sampling uncertainty.
        "population": real_profiles,
        "population_target_size": int(pool_size),
        "stats": {
            **parse_stats,
            "population_pipeline_version": 9,
            "pool_size":       int(pool_size),
            "empirical_sampling_units": len(real_profiles),
            "population_method": "seeded_complete_profile_resampling_with_replacement",
            "synthetic_attribute_jitter": False,
            "catalogue_rows_raw": len(products_dict.get("products", [])),
            "catalogue_skus":     len(canonical_products),
            "catalogue_duplicates_collapsed": (
                len(products_dict.get("products", [])) - len(canonical_products)
            ),
            "archetypes_real": archetype_counts,
            "exploratory_archetypes_real": exploratory_counts,
            "archetype_stability": archetype_stats,
            "questionnaire_reliability": questionnaire_stats,
            "behavioral_calibration": calibration_stats,
            "dce_choice_validation": dce_choice_stats,
            "substitution_choice_validation": substitution_stats,
            "baseline_reproduction_targets": baseline_targets,
            "phase2_reproduction_targets": phase2_targets,
        },
    }
    return config


def run_pipeline(
    firebase_path: str,
    products_path: str,
    output_path: str = "mesa_config.json",
    pool_size: int = 2000,
    n_archetypes: int = 4,
    dce_path: str | None = None,
) -> dict:
    """
    Convenience wrapper for command-line use.
    Reads from files, writes output JSON, and returns the config dict.
    """
    firebase_dict = load_json(firebase_path)
    products_dict = load_json(products_path)
    dce_rows = None
    if dce_path:
        with open(dce_path, "r", encoding="utf-8-sig", newline="") as handle:
            dce_rows = list(csv.DictReader(handle))

    config = run_pipeline_from_data(
        firebase_dict, products_dict, pool_size, n_archetypes,
        dce_rows=dce_rows,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    s = config["stats"]
    print(f"✅  Real profiles parsed     : {s['n_real']}")
    print(f"⚠️   Profiles skipped (empty) : {s['n_skipped']}")
    print(f"ℹ️   Basket items dropped      : {s['n_products_dropped']}")
    print(f"🧬  Synthetic agents added    : {s['pool_size'] - s['n_real']}")
    print(f"👥  Total pool size           : {s['pool_size']}")
    print(f"🏷️   Archetype distribution    : {s['archetypes_real']}")
    print(f"💾  Saved → {output_path}")

    return config


if __name__ == "__main__":
    run_pipeline(
        firebase_path  = "Finland_20.1.2026v1.json",
        products_path  = "master_products.json",
        output_path    = "mesa_config.json",
        pool_size      = 2000,
        n_archetypes   = 4,
        dce_path       = ".streamlit/dce_data_clean.csv",
    )
