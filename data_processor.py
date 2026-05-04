"""
GROCERYsim Data Processor v2.0
================================
Converts a Firebase JSON export + Unity product catalogue into an
enriched population pool for the Mesa ABM.

Pipeline
--------
1. Parse Firebase JSON → demographics, round1/round2 baskets, DCE choices,
   questionnaire ratings
2. Compute DCE preference scores  (origin, organic, fat, price sensitivity)
3. Compute questionnaire factor scores → assign behavioral archetype via K-Means
4. Extract observed price elasticity from round1 vs round2 basket comparison
5. Build enriched "real" profiles
6. Stratified bootstrap to target pool size (preserves archetype distribution)
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
import math
import random
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
      dce_price_sensitivity – tendency to choose the cheaper of two options
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
        lp = DCE_PRICES.get(left_code, 1.5)
        rp = DCE_PRICES.get(right_code, 1.5)
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
        "dce_price_sensitivity": (cheaper_chosen / price_decisions) if price_decisions > 0 else 0.5,
    }


# ---------------------------------------------------------------------------
# 3. Questionnaire factor scoring
# ---------------------------------------------------------------------------

def parse_questionnaire(ratings: list) -> dict:
    """
    Parse the 21-item Likert questionnaire into five factor scores (0–1).
    Gracefully handles missing items by substituting neutral value 3.
    """
    scores = []
    for r in ratings:
        try:
            scores.append(float(r["value"]))
        except (KeyError, ValueError, TypeError):
            scores.append(3.0)
    # Pad to 21 if shorter
    while len(scores) < 21:
        scores.append(3.0)

    factor_scores = {}
    for factor, indices in QUESTIONNAIRE_FACTORS.items():
        vals = [scores[i] for i in indices if i < len(scores)]
        # Normalise from [1, 5] → [0, 1]
        factor_scores[f"q_{factor}"] = float((np.mean(vals) - 1.0) / 4.0)

    return factor_scores


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
        }

    round2_actual = sum(
        float(i.get("price", 0)) * int(i.get("quantity", 1))
        for i in round2_basket
    )

    budget_util       = min(1.0, round2_actual / round2_max) if round2_max > 0 else 1.0
    spending_reduction = max(0.0, 1.0 - round2_actual / round1_total) if round1_total > 0 else 0.0

    # Substitution: items in round2 that differ from round1 within same category
    r1_cat_to_name = {i.get("category", ""): i.get("productName", "") for i in round1_basket}
    subs = sum(
        1 for item in round2_basket
        if item.get("category") in r1_cat_to_name
        and r1_cat_to_name[item["category"]] != item.get("productName", "")
    )
    substitution_rate = subs / len(round2_basket) if round2_basket else 0.0

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
    }


# ---------------------------------------------------------------------------
# 5. Clean basket items against the product catalogue
# ---------------------------------------------------------------------------

def _clean_basket(raw_basket: list, valid_names: set) -> list:
    """
    Return only items whose productName is present in the Unity catalogue.
    Items absent from the catalogue are not simulated.
    """
    clean = []
    for item in raw_basket:
        name = item.get("productName", "")
        if name not in valid_names:
            continue
        clean.append({
            "product_name":   name,
            "quantity":       max(1, int(item.get("quantity", 1))),
            "price":          float(item.get("price", 1.0)),
            "category":       item.get("category", "Unknown"),
            "fat_content":    float(item.get("fatContent", 0.0)),
            "is_bio":         bool(item.get("isBio", False)),
            "is_plant_based": bool(item.get("isPlantBased", False)),
        })
    return clean


# ---------------------------------------------------------------------------
# 6. Build enriched real profiles
# ---------------------------------------------------------------------------

def build_real_profiles(
    firebase_data: dict,
    valid_product_names: set,
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

    for user_id, user_data in firebase_data.items():
        demo      = extract_demographics(user_data)
        r1_raw    = user_data.get("round1_Basket", [])
        r2_raw    = user_data.get("round2_Basket", [])
        r1_total  = float(user_data.get("round1_TotalSpent", 0.0))
        r2_max    = float(user_data.get("round2_MaxBudget", r1_total))

        clean_r1 = _clean_basket(r1_raw, valid_product_names)
        clean_r2 = _clean_basket(r2_raw, valid_product_names)

        # Track how many basket items were dropped
        n_products_dropped += (len(r1_raw) - len(clean_r1)) + (len(r2_raw) - len(clean_r2))

        if not clean_r1:
            n_skipped += 1
            continue

        # If round2 basket is entirely out-of-catalogue, fall back to round1
        if not clean_r2:
            clean_r2 = clean_r1

        dce_prefs  = parse_dce_choices(user_data.get("choiceExperiment1_Results", []))
        q_scores   = parse_questionnaire(user_data.get("questionnaireRatings", []))
        elasticity = compute_price_elasticity(r1_raw, r2_raw, r1_total, r2_max)

        # Blend DCE and observed price sensitivity
        price_sensitivity = 0.5 * dce_prefs["dce_price_sensitivity"] + 0.5 * elasticity["price_sensitivity"]

        profile = {
            "source_id":           user_id,
            "is_real":             True,
            **demo,
            "baseline_basket":     clean_r1,
            "crisis_basket":       clean_r2,
            "budget":              float(r1_total),
            "crisis_budget":       float(r2_max),
            # DCE-derived preferences
            "finnish_preference":  float(dce_prefs["finnish_preference"]),
            "organic_preference":  float(dce_prefs["organic_preference"]),
            "preferred_fat":       float(dce_prefs["preferred_fat"]),
            # Blended price sensitivity
            "price_sensitivity":   float(min(1.0, price_sensitivity)),
            "substitution_rate":   float(elasticity["substitution_rate"]),
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

def assign_archetypes(profiles: list, n_clusters: int = 4, random_state: int = 42) -> list:
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
            p["archetype"] = ARCHETYPE_LABELS[0]
        return profiles

    # Feature matrix: questionnaire factors + behavioural prefs
    feature_keys = [
        "q_price", "q_health", "q_environment", "q_animal_welfare", "q_sensory_habit",
        "price_sensitivity", "organic_preference", "finnish_preference",
    ]
    X_raw = np.array([
        [p.get(k, 0.5) for k in feature_keys]
        for p in profiles
    ])

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
        # indices: price(0), health(1), env(2), animal(3), sensory(4), price_sens(5), organic(6), finnish(7)
        price_score  = c[0] + c[5]           # questionnaire price + behavioural price sensitivity
        green_score  = c[2] + c[3] + c[6]    # env + animal_welfare + organic preference
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
        p["archetype"]  = label_map[int(labels[i])]
        p["cluster_id"] = int(labels[i])

    return profiles


# ---------------------------------------------------------------------------
# 8. Stratified bootstrap
# ---------------------------------------------------------------------------

def _jitter_profile(template: dict, agent_idx: int, rng: np.random.Generator) -> dict:
    """
    Create a synthetic agent by applying controlled noise to a real profile.
    Continuous scores are jittered with a small Gaussian; demographics are
    perturbed within realistic integer ranges.
    """
    new = template.copy()
    new["source_id"] = f"synthetic_{agent_idx}"
    new["is_real"]   = False

    # Demographic perturbation
    new["age"]            = int(max(18, template["age"] + rng.integers(-5, 6)))
    new["household_size"] = int(max(1, template["household_size"] + rng.integers(-1, 2)))
    new["children"]       = int(max(0, template["children"] + rng.integers(-1, 2)))

    def _clip_jitter(val: float, sigma: float = 0.10) -> float:
        return float(np.clip(val + rng.normal(0, sigma), 0.0, 1.0))

    new["price_sensitivity"]  = _clip_jitter(template["price_sensitivity"],  0.10)
    new["finnish_preference"] = _clip_jitter(template["finnish_preference"],  0.12)
    new["organic_preference"] = _clip_jitter(template["organic_preference"],  0.12)
    new["preferred_fat"]      = float(np.clip(
        template["preferred_fat"] + rng.normal(0, 0.5), 0.0, 3.8
    ))

    # Jitter questionnaire factors
    for key in new:
        if key.startswith("q_"):
            new[key] = _clip_jitter(template.get(key, 0.5), 0.08)

    # Jitter basket prices and occasionally vary quantity ±1
    def _jitter_basket(basket: list) -> list:
        jittered = []
        for item in basket:
            ni = item.copy()
            ni["price"] = float(item["price"] * rng.uniform(0.90, 1.10))
            if item["quantity"] > 1 and rng.random() < 0.15:
                ni["quantity"] = max(1, item["quantity"] + int(rng.choice([-1, 1])))
            jittered.append(ni)
        return jittered

    new["baseline_basket"] = _jitter_basket(template["baseline_basket"])
    new["crisis_basket"]   = _jitter_basket(template["crisis_basket"])

    # Recalculate reference price from jittered baseline basket
    if new["baseline_basket"]:
        new["reference_price"] = float(
            np.mean([i["price"] for i in new["baseline_basket"]])
        )

    return new


def bootstrap_population(
    real_profiles: list,
    target_size: int,
    jitter_seed: int = 42,
) -> list:
    """
    Build a population pool of exactly `target_size` agents.

    If target_size <= len(real_profiles) : return a random sample of real profiles.
    If target_size  > len(real_profiles) : include all real + generate synthetics,
      drawing templates proportionally from each archetype bucket (stratified).
    """
    rng = np.random.default_rng(jitter_seed)
    random.seed(jitter_seed)

    if not real_profiles:
        return []

    if target_size <= len(real_profiles):
        return random.sample(real_profiles, target_size)

    population = list(real_profiles)  # all real profiles go in first
    n_synthetic = target_size - len(real_profiles)

    # Build archetype pools for stratified sampling
    archetype_pool: dict[str, list] = {}
    for p in real_profiles:
        a = p.get("archetype", ARCHETYPE_LABELS[0])
        archetype_pool.setdefault(a, []).append(p)

    archetypes = list(archetype_pool.keys())
    weights    = [len(archetype_pool[a]) / len(real_profiles) for a in archetypes]

    for i in range(n_synthetic):
        archetype = random.choices(archetypes, weights=weights, k=1)[0]
        template  = random.choice(archetype_pool[archetype])
        synthetic = _jitter_profile(template, len(real_profiles) + i, rng)
        population.append(synthetic)

    random.shuffle(population)
    return population


# ---------------------------------------------------------------------------
# 9. Public pipeline entry points
# ---------------------------------------------------------------------------

def run_pipeline_from_data(
    firebase_dict: dict,
    products_dict: dict,
    pool_size: int = 2000,
    n_archetypes: int = 4,
) -> dict:
    """
    Main entry point for the Streamlit app.
    Accepts pre-loaded dicts; returns mesa_config dict.
    """
    valid_product_names = {p["name"] for p in products_dict.get("products", [])}

    real_profiles, parse_stats = build_real_profiles(firebase_dict, valid_product_names)

    if not real_profiles:
        raise ValueError(
            "No valid profiles could be built — check that basket product names "
            "match those in master_products.json."
        )

    real_profiles = assign_archetypes(real_profiles, n_clusters=n_archetypes)

    population = bootstrap_population(real_profiles, pool_size)

    archetype_counts = {a: 0 for a in ARCHETYPE_LABELS}
    for p in real_profiles:
        archetype_counts[p.get("archetype", ARCHETYPE_LABELS[0])] += 1

    config = {
        "products":   products_dict.get("products", []),
        "population": population,
        "stats": {
            **parse_stats,
            "pool_size":       len(population),
            "archetypes_real": archetype_counts,
        },
    }
    return config


def run_pipeline(
    firebase_path: str,
    products_path: str,
    output_path: str = "mesa_config.json",
    pool_size: int = 2000,
    n_archetypes: int = 4,
) -> dict:
    """
    Convenience wrapper for command-line use.
    Reads from files, writes output JSON, and returns the config dict.
    """
    firebase_dict = load_json(firebase_path)
    products_dict = load_json(products_path)

    config = run_pipeline_from_data(
        firebase_dict, products_dict, pool_size, n_archetypes
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
    )
