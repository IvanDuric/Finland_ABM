"""
GROCERYsim ABM Engine v2.0
============================
Mesa-based agent-based model for grocery store supply chain and consumer
behaviour simulation.

Agent types
-----------
ProductAgent     – represents one SKU on the shelf + in storage.
                   Manages shelf replenishment from storage, ageing, near-expiry
                   discounting, expiry-based removal, and reorder signalling.
SupplyTruck      – singleton logistics agent.  Queues and delivers orders,
                   enforces lead-time and supply disruption periods.
ConsumerAgent    – one shopping visit by a persistent household profile. Makes
                   evidence-gated price-acceptance and substitution decisions;
                   pantry stock persists between visits. Unidentified learning,
                   panic, TPB, and Prospect Theory dynamics are opt-in extensions.

Model
-----
SupermarketModel – orchestrates the simulation.  Accepts the in-memory config
                   dict produced by data_processor.run_pipeline_from_data(),
                   schedules repeat household visits, advances home consumption,
                   and records per-day aggregate and per-product metrics.
"""

import json
import math
import random
from collections import defaultdict

import numpy as np
from mesa import Agent, Model
from mesa.time import BaseScheduler

from behavioral_mechanisms import (
    effective_hoarding_multiplier,
    normalized_tpb_weights,
    tpb_intention,
)


# ---------------------------------------------------------------------------
# Optional archetype behavioural modifiers
# ---------------------------------------------------------------------------

# Exploratory learning rate; not estimated from the current dataset.
LEARNING_RATE = 0.015

ARCHETYPE_MODIFIERS = {
    # (substitution_tolerance, panic_hoarding_multiplier, price_tolerance_extra)
    "price_champion":   (0.85, 1.2, -0.05),   # readily substitutes; mild hoarding
    "green_buyer":      (0.35, 1.0, +0.20),   # resistant to substitution to non-organic
    "health_optimizer": (0.55, 1.1, +0.10),
    "habitual_buyer":   (0.25, 1.4, +0.05),   # brand-loyal; higher panic hoarding
}


# ---------------------------------------------------------------------------
# CO2 emission factors (kg CO2-equivalent per unit sold or wasted)
# ---------------------------------------------------------------------------

# Emission intensity by (is_finnish: bool, is_organic: bool)
# Finnish = shorter supply chain; organic = lower input intensity
CO2_PRODUCTION: dict[tuple, float] = {
    (True,  True):  0.8,    # Finnish organic
    (True,  False): 1.2,    # Finnish conventional
    (False, True):  1.5,    # Imported organic (longer transport)
    (False, False): 2.2,    # Imported conventional (production + transport)
}
CO2_WASTE_FACTOR = 3.5      # kg CO2-eq per wasted unit (methane from decomposition)


def is_low_income_access_stressed(consumer, fulfillment_threshold: float = 0.80) -> bool:
    """Return whether a low-income shopper faces affordability or access stress.

    Budget exhaustion alone is not sufficient: during a severe stockout a shopper
    may be unable to find enough products to spend their budget. The diagnostic
    therefore also flags a materially unfulfilled requested basket.
    """
    if float(consumer.income_midpoint) >= 2000.0:
        return False
    wanted = int(consumer.items_wanted)
    access_shortfall = (
        wanted > 0
        and float(consumer.items_purchased) / wanted < float(fulfillment_threshold)
    )
    return bool(consumer.budget_exhausted or access_shortfall)


# ---------------------------------------------------------------------------
# Policy Configuration
# ---------------------------------------------------------------------------

class PolicyConfig:
    """
    Holds all active policy levers for a simulation run.

    Instantiate from a plain dict (e.g. from the Streamlit sidebar) so that the
    model never needs to know about Streamlit session_state directly.
    """

    def __init__(self, cfg: dict = None):
        if cfg is None:
            cfg = {}

        # --- Fat / sugar tax ---
        self.fat_tax_active    = bool(cfg.get("fat_tax_active",    False))
        self.fat_tax_threshold = float(cfg.get("fat_tax_threshold", 3.5))   # % fat
        self.fat_tax_rate      = float(cfg.get("fat_tax_rate",      0.20))  # 20 % surcharge

        # --- Domestic / organic subsidy ---
        self.subsidy_active = bool(cfg.get("subsidy_active", False))
        # target: "domestic", "organic", or "both"
        self.subsidy_target = str(cfg.get("subsidy_target", "domestic"))
        self.subsidy_rate   = float(cfg.get("subsidy_rate",  0.15))         # 15 % discount
        self.subsidy_categories = {
            str(category).strip().casefold()
            for category in (cfg.get("subsidy_categories", []) or [])
        }

        # --- Domestic supply shock (animal disease, drought, …) ---
        self.domestic_shock_active   = bool(cfg.get("domestic_shock_active",   False))
        self.domestic_shock_day      = int(cfg.get("domestic_shock_day",       30))
        self.domestic_shock_duration = int(cfg.get("domestic_shock_duration",  30))
        # 0–1: fraction of Finnish deliveries blocked during the shock
        self.domestic_shock_severity = float(cfg.get("domestic_shock_severity", 0.70))

        # --- Nutritional labelling / consumer information ---
        self.labelling_active        = bool(cfg.get("labelling_active",        False))
        self.labelling_day           = int(cfg.get("labelling_day",            1))
        # Additive boost to agent preference scores once labelling kicks in
        self.labelling_health_boost  = float(cfg.get("labelling_health_boost",  0.15))
        self.labelling_organic_boost = float(cfg.get("labelling_organic_boost", 0.10))

    # Convenience helpers (called from agents each step)

    def is_shock_active(self, current_day: int) -> bool:
        if not self.domestic_shock_active:
            return False
        return self.domestic_shock_day <= current_day < (
            self.domestic_shock_day + self.domestic_shock_duration
        )

    def is_labelling_active(self, current_day: int) -> bool:
        return self.labelling_active and current_day >= self.labelling_day

    def apply_price_policy(
        self, base_price: float, fat_content: float,
        is_finnish: bool, is_organic: bool, category: str = "",
    ) -> float:
        """Return the post-policy shelf price for one product unit."""
        price = base_price

        # Fat tax surcharge
        if self.fat_tax_active and fat_content >= self.fat_tax_threshold:
            price *= (1.0 + self.fat_tax_rate)

        # Domestic / organic subsidy discount
        if self.subsidy_active:
            applies = (
                (self.subsidy_target == "domestic" and is_finnish) or
                (self.subsidy_target == "organic"  and is_organic) or
                (self.subsidy_target == "both"     and (is_finnish or is_organic))
                or (
                    self.subsidy_target == "category"
                    and str(category).strip().casefold() in self.subsidy_categories
                )
            )
            if applies:
                price *= (1.0 - self.subsidy_rate)

        return round(price, 4)


# ---------------------------------------------------------------------------
# 1. Product Agent
# ---------------------------------------------------------------------------

class ProductAgent(Agent):
    """
    Represents one SKU.

    Stock is modelled as a FIFO list of batches on the shelf, each with an
    age counter.  Storage stock is kept as a simple scalar.

    Near-expiry logic
    -----------------
    • age >= (max_shelf_life - NEAR_EXPIRY_DAYS) → price drops 50 %
    • age >= max_shelf_life                       → batch is removed and counted
                                                    as waste

    Replenishment logic
    -------------------
    Shelf < 30 % capacity  → move stock from storage (intra-day)
    Storage < reorder_point → truck places order; delivery arrives in lead_time days
    """

    NEAR_EXPIRY_DAYS = 2  # days before expiry when discount kicks in

    def __init__(self, unique_id, model, product_data: dict, ai_capacity: int = None):
        super().__init__(unique_id, model)

        self.prod_id      = product_data.get("id", unique_id)
        self.name         = product_data["name"]
        self.category     = product_data.get("category", "Unknown")
        self.base_price   = float(product_data.get("price", 1.0))
        self.current_price = self.base_price

        self.is_bio         = bool(product_data.get("is_bio", False))
        self.fat_content    = float(product_data.get("fat_content", 0.0))
        # Normalise origin → canonical "Suomi" for any Finnish-equivalent label.
        # Unity exports "Finnish", Firebase may use "FI", survey uses "Suomi".
        _raw_origin = product_data.get("origin", "Unknown")
        _FINNISH = {"suomi", "finnish", "finland", "fi", "kotimainen"}
        self.origin = "Suomi" if str(_raw_origin).strip().lower() in _FINNISH else _raw_origin
        self.is_domestic = bool(
            product_data.get("is_domestic", self.origin == "Suomi")
        )
        self.is_plant_based = bool(product_data.get("is_plant_based", False))
        # Optional case-study attributes used by a validated pooled choice model
        # (for example origin, size and appearance in the Portugal orange DCE).
        # Finland products omit this field and retain the existing milk utility.
        self.choice_attributes = {
            str(key): float(value)
            for key, value in (product_data.get("choice_attributes", {}) or {}).items()
        }

        # Shelf life is needed when constructing the initial FIFO age profile.
        self.max_shelf_life = int(product_data.get("shelf_life_days", 14))

        # --- Shelf (FIFO batches) ---
        initial_shelf = int(product_data.get("initial_stock_shelf", 10))
        self.max_shelf_capacity = int(product_data.get("max_shelf_capacity", 20))
        # A running shop does not open with every unit delivered simultaneously.
        # Spread opening stock across deterministic age cohorts so identical
        # shelf-life products do not all expire on the same artificial day.
        cohort_count = min(max(1, initial_shelf), max(1, self.max_shelf_life))
        cohort_base, cohort_remainder = divmod(initial_shelf, cohort_count)
        self.shelf_batches = []
        for cohort in range(cohort_count):
            quantity = cohort_base + (1 if cohort < cohort_remainder else 0)
            age = int(cohort * self.max_shelf_life / cohort_count)
            if quantity > 0:
                self.shelf_batches.append({"qty": quantity, "age": age})

        # --- Storage (scalar) ---
        default_storage            = int(product_data.get("initial_stock_storage", 20))
        self.max_storage_capacity  = int(
            ai_capacity
            if ai_capacity is not None
            else product_data.get("max_storage_capacity", 50)
        )
        self.stock_storage = int(
            ai_capacity if ai_capacity is not None else default_storage
        )
        self.stock_storage = min(
            self.stock_storage, self.max_storage_capacity
        )

        # Daily counters (reset each step)
        self.daily_sales            = 0
        self.daily_revenue          = 0.0   # nominal (current_price × units) — for store cash-flow
        self.daily_base_revenue     = 0.0   # constant-price (base_price × units) — for demand charts
        self.daily_waste            = 0
        self.daily_lost_sales       = 0
        self.daily_substitutions    = 0
        self.daily_refused_delivery = 0
        self.daily_near_expiry_sold = 0

        # Policy / environmental counters (reset each step)
        self.daily_co2_sales     = 0.0   # kg CO2-eq from units sold
        self.daily_co2_waste     = 0.0   # kg CO2-eq from units wasted
        self.daily_domestic_sales = 0    # units sold that are Finnish-origin
        self.daily_import_sales   = 0    # units sold that are imported

        # Snapshot for logging (taken once per step before consumer action)
        self.snap_shelf   = 0
        self.snap_storage = 0
        self.snap_pending = 0

        # Set during the product phase and read during the later consumer phase.
        self._has_near_expiry = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stock_shelf(self) -> int:
        return sum(b["qty"] for b in self.shelf_batches)

    @property
    def effective_price(self) -> float:
        """Current selling price, considering discount for near-expiry batches."""
        return self.current_price

    def get_oldest_batch_age(self) -> int:
        if not self.shelf_batches:
            return 0
        return max(b["age"] for b in self.shelf_batches)

    def days_until_expiry(self) -> int:
        """Minimum days until any shelf batch expires."""
        if not self.shelf_batches:
            return self.max_shelf_life
        oldest = max(b["age"] for b in self.shelf_batches)
        return max(0, self.max_shelf_life - oldest)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self):
        """
        Executed once per simulation day BEFORE consumers shop.
        Order of operations:
          1. Reset daily counters
          2. Refill shelf from storage if below 50 %
          3. Apply inflation and policy prices
          4. Age shelf batches and remove expired units
          5. Snapshot the post-expiry stock available to shoppers
        """
        # 1. Reset counters
        self.daily_sales            = 0
        self.daily_revenue          = 0.0   # nominal — reset each step
        self.daily_base_revenue     = 0.0   # constant-price — reset each step
        self.daily_waste            = 0
        self.daily_lost_sales       = 0
        self.daily_substitutions    = 0
        self.daily_refused_delivery = 0
        self.daily_near_expiry_sold = 0
        self.daily_co2_sales        = 0.0
        self.daily_co2_waste        = 0.0
        self.daily_domestic_sales   = 0
        self.daily_import_sales     = 0

        # 2. Refill shelf from storage
        # Threshold raised from 30% → 50%: stock-room staff refill shelves
        # proactively at half-empty rather than waiting until nearly empty.
        # The old 30% threshold caused artificial stockouts on the shop floor
        # while storage remained full — stock was available but unreachable.
        current_shelf   = self.stock_shelf
        shelf_threshold = self.max_shelf_capacity * 0.50
        if current_shelf < shelf_threshold:
            needed   = self.max_shelf_capacity - current_shelf
            to_move  = min(needed, self.stock_storage)
            if to_move > 0:
                self.shelf_batches.append({"qty": to_move, "age": 0})
                self.stock_storage -= to_move

        # 3. Price update — crisis inflation first, then policy modifiers
        if (
            self.model.is_scenario_active
            and self.prod_id in self.model.scenario_price_overrides
        ):
            inflated = round(
                self.model.scenario_price_overrides[self.prod_id], 4
            )
        elif self.model.is_scenario_active:
            inflated = round(
                self.base_price * (1.0 + self.model.inflation_percent / 100.0), 4
            )
        else:
            inflated = self.base_price

        # Apply fat tax / subsidy on top of (possibly inflated) base price
        policy: PolicyConfig = self.model.policy_config
        is_finnish = self.is_domestic
        self.current_price = policy.apply_price_policy(
            inflated, self.fat_content, is_finnish, self.is_bio, self.category
        )

        # 4. Age batches, apply near-expiry discount, remove expired
        for b in self.shelf_batches:
            b["age"] += 1

        valid_batches = []
        has_near_expiry = False
        for b in self.shelf_batches:
            days_left = self.max_shelf_life - b["age"]
            if days_left <= 0:
                # Expired — remove and count as waste
                self.daily_waste    += b["qty"]
                self.daily_co2_waste += b["qty"] * CO2_WASTE_FACTOR
                self.model.food_waste_log.record(
                    day      = self.model.current_day,
                    product_id = self.prod_id,
                    product  = self.name,
                    category = self.category,
                    quantity = b["qty"],
                    reason   = "Expiry",
                )
            else:
                if days_left <= self.NEAR_EXPIRY_DAYS:
                    has_near_expiry = True
                valid_batches.append(b)
        self.shelf_batches = valid_batches

        # Store whether we have near-expiry stock (used during purchase)
        self._has_near_expiry = has_near_expiry

        # 5. Snapshot the state consumers can actually access. Previously this
        # happened before expiry removal, so charts showed inventory that was no
        # longer available during the shopping phase.
        self.snap_shelf = self.stock_shelf
        self.snap_storage = self.stock_storage
        self.snap_pending = self.model.truck.get_pending_stock(self.prod_id)


# ---------------------------------------------------------------------------
# 2. Food Waste Log
# ---------------------------------------------------------------------------

class FoodWasteLog:
    """Accumulates food waste events for the entire simulation run."""

    def __init__(self):
        self.records: list[dict] = []

    def record(
        self, day: int, product: str, category: str, quantity: int, reason: str,
        product_id: str | None = None,
    ):
        self.records.append({
            "Day": day, "ProductID": product_id, "Product": product,
            "Category": category,
            "Quantity": quantity, "Reason": reason,
        })

    def total_waste(self) -> int:
        return sum(r["Quantity"] for r in self.records)

    def by_product(self) -> dict:
        totals: dict[str, int] = defaultdict(int)
        for r in self.records:
            totals[r["Product"]] += r["Quantity"]
        return dict(totals)

    def by_category(self) -> dict:
        totals: dict[str, int] = defaultdict(int)
        for r in self.records:
            totals[r["Category"]] += r["Quantity"]
        return dict(totals)


# ---------------------------------------------------------------------------
# 3. Supply Truck Agent
# ---------------------------------------------------------------------------

class SupplyTruck(Agent):
    """
    Manages the order queue and delivery schedule.

    Reorder trigger : storage < reorder_point  (e.g. 30 % of max_storage)
    Order quantity  : fill storage to target_stock_level  (e.g. 90 % of max)
    Lead time       : configurable (default 2 days)
    Disruption      : during disruption_days after scenario start, deliveries
                      are blocked but orders are still queued.
    """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.delivery_queue: dict[int, dict[str, int]] = {}
        self.log: list[dict] = []

    def get_pending_stock(self, product_id: str) -> int:
        return sum(
            manifest.get(product_id, 0)
            for manifest in self.delivery_queue.values()
        )

    def step(self):
        today = self.model.current_day

        # --- Disruption check (scenario-level and policy-level) ---
        trucks_allowed = True
        if self.model.is_scenario_active and self.model.supply_disruption_days > 0:
            days_into_scenario = today - self.model.scenario_start_day
            if 0 <= days_into_scenario < self.model.supply_disruption_days:
                trucks_allowed = False

        policy: PolicyConfig = self.model.policy_config
        shock_active = policy.is_shock_active(today)

        # --- Deliver pending orders that have arrived ---
        if trucks_allowed:
            arrived = sorted(d for d in self.delivery_queue if d <= today)
            for d_date in arrived:
                for product_id, qty in self.delivery_queue[d_date].items():
                    product = self.model.get_product_by_id(product_id)
                    if product:
                        # Policy domestic supply shock — randomly block a fraction
                        # of Finnish-origin deliveries proportional to severity
                        if shock_active and product.is_domestic:
                            blocked_frac = policy.domestic_shock_severity
                            qty = max(0, int(qty * (1.0 - blocked_frac)))
                            if qty == 0:
                                self.log.append({
                                    "Day": today, "Product": product.name,
                                    "ProductID": product.prod_id,
                                    "Action": "Blocked", "Quantity": 0,
                                    "Refused": 0, "Note": "Domestic shock",
                                })
                                continue

                        space     = product.max_storage_capacity - product.stock_storage
                        accepted  = max(0, min(qty, space))
                        refused   = qty - accepted
                        product.stock_storage += accepted
                        if refused > 0:
                            product.daily_refused_delivery += refused
                            # Refused delivery → waste (perishable cannot be returned)
                            product.daily_waste            += refused
                            product.daily_co2_waste        += refused * CO2_WASTE_FACTOR
                            self.model.food_waste_log.record(
                                day      = today,
                                product_id = product.prod_id,
                                product  = product.name,
                                category = product.category,
                                quantity = refused,
                                reason   = "Refused Delivery",
                            )
                        self.log.append({
                            "Day": today, "Product": product.name,
                            "ProductID": product.prod_id,
                            "Action": "Delivery", "Quantity": accepted,
                            "Refused": refused,
                            "Note": "Storage Full" if refused > 0 else "OK",
                        })
                del self.delivery_queue[d_date]

        # --- Place new orders ---
        arrival_day = today + self.model.lead_time_days
        todays_order: dict[str, int] = {}

        for agent in self.model.products:

            # Check total supply pipeline: storage already on hand + stock in transit.
            # Old logic blocked ALL new orders if even 1 unit was pending — this caused
            # under-ordering during demand surges (e.g. hoarding events) because the
            # small pending delivery would become inadequate by the time it arrived.
            pending       = self.get_pending_stock(agent.prod_id)
            total_supply  = agent.stock_storage + pending
            trigger       = agent.max_storage_capacity * self.model.reorder_point

            if total_supply >= trigger:
                continue   # enough supply in pipeline, no order needed

            # Order enough to fill storage to target level, net of what's already coming
            target_qty = int(agent.max_storage_capacity * self.model.target_stock_level)
            order_qty  = max(0, target_qty - total_supply)
            if order_qty > 0:
                todays_order[agent.prod_id] = order_qty
                self.log.append({
                    "Day": today, "Product": agent.name,
                    "ProductID": agent.prod_id,
                    "Action": "Order", "Quantity": order_qty,
                    "Explanation": (
                        f"TotalSupply {total_supply} (storage {agent.stock_storage} "
                        f"+ pending {pending}) < trigger {int(trigger)}"
                    ),
                })

        if todays_order:
            if arrival_day not in self.delivery_queue:
                self.delivery_queue[arrival_day] = {}
            for product_id, qty in todays_order.items():
                self.delivery_queue[arrival_day][product_id] = (
                    self.delivery_queue[arrival_day].get(product_id, 0) + qty
                )


# ---------------------------------------------------------------------------
# 4. Consumer Agent
# ---------------------------------------------------------------------------

class ConsumerAgent(Agent):
    """
    Represents a single shopping visit by a persistent household profile.

    Created at the start of each simulation day and removed at day end.

    Decision process
    ----------------
    For each item in the active basket (baseline or crisis):
      1. Locate the product.
      2. Evaluate utility(product).  If below threshold, record lost sale and
         attempt to find a substitute.
      3. If product is in stock, purchase it (FIFO from oldest batch first,
         near-expiry stock gets 50 % discount).
      4. If out of stock, record lost sale and attempt substitute.

    Choice architecture
    -------------------
    Requested-SKU acceptance uses the separately calibrated incremental price-loss
    margin. Substitute candidates must be in stock, affordable, and in the same
    category. A replacement-event audit gates transfer of the retention price screen
    and deterministic attribute ordering. Failed gates use a seeded uniform draw
    among feasible candidates outside domains with validated choice evidence.
    A phase-transition visit samples the cross-fitted substitution propensity once per
    basket line and distinguishes maximum available budget from reservation spending.
    """

    def __init__(self, unique_id, model, profile: dict):
        super().__init__(unique_id, model)
        self.profile = profile
        self.household_id = str(profile.get("_household_id", unique_id))
        self.visit_number = int(profile.get("_visit_count", 0)) + 1
        self.expected_visit_interval = float(
            profile.get("_expected_visit_interval", 1.0)
        )

        # Preference scores (0–1 each)
        self.price_sensitivity  = float(profile.get("price_sensitivity",  0.5))
        self.finnish_preference = float(profile.get("finnish_preference",  0.5))
        self.organic_preference = float(profile.get("organic_preference",  0.2))
        self.preferred_fat      = float(profile.get("preferred_fat",       1.5))
        self.reference_price    = float(profile.get("reference_price",     1.5))
        self.substitution_rate  = float(profile.get("substitution_rate",   0.5))
        self.archetype          = profile.get("archetype", "habitual_buyer")

        # Cluster labels do not validate the modifier magnitudes below. Keep
        # them neutral unless an exploratory run explicitly enables them.
        mods = (
            ARCHETYPE_MODIFIERS.get(self.archetype, (0.5, 1.0, 0.0))
            if model.archetype_modifiers_enabled
            else (0.5, 1.0, 0.0)
        )
        # If behavioural learning has already updated sub_tolerance in the profile,
        # use that value; otherwise fall back to the archetype default.
        self.sub_tolerance        = float(profile.get("sub_tolerance", mods[0]))
        self.hoarding_multiplier  = mods[1]   # hoarding boost during panic
        self.price_tolerance_extra = mods[2]  # bonus tolerance on top of base

        # Baskets
        self.baseline_basket = profile.get("baseline_basket", [])
        self.crisis_basket   = profile.get("crisis_basket",   self.baseline_basket)
        self.budget          = round(float(profile.get("budget", 50.0)), 2)
        self.crisis_budget   = round(
            float(profile.get("crisis_budget", self.budget)), 2
        )
        self.budget_utilization_propensity = min(
            1.0,
            max(0.0, float(profile.get("budget_utilization_propensity", 1.0))),
        )

        # Income proxy — used for affordability / food-stress calculations.
        # Stored as the midpoint of the reported income bracket (€ / month).
        self.income_midpoint = float(profile.get("income_midpoint", 2500.0))

        # Panic state (updated by model)
        self.panic_level = 0.0

        # ---- Policy / welfare tracking (reset each step) ----
        self.items_wanted     = 0      # items in active basket
        self.items_base_wanted = 0     # observed basket quantity before stockpiling
        self.items_allowed     = 0     # demand after any purchase-limit policy
        self.items_purchased  = 0      # items actually bought
        self.items_substituted = 0     # purchased units supplied by another SKU
        self.choice_lines_purchased = 0
        self.choice_lines_substituted = 0
        self.budget_exhausted = False  # ran out of budget before finishing basket
        self.amount_spent     = 0.0    # nominal expenditure during this visit
        self.total_fat_bought = 0.0    # sum(fat_content × qty) for nutrition scoring

        # ---- Behavioural learning state — persisted in profile dict ----
        # Visit objects are short-lived, so longitudinal learning belongs to the
        # persistent household profile.
        self._organic_streak: int         = profile.setdefault("_organic_streak", 0)
        self._fat_history:    list[float] = profile.setdefault("_fat_history", [])

        # ── Prospect Theory (Kahneman & Tversky 1979) ──────────────────────────
        # Loss aversion λ=2.25 and curvature α=0.88 from Tversky & Kahneman (1992)
        self.loss_aversion   = float(profile.get("loss_aversion", 2.25))
        self.kt_alpha        = 0.88
        # Per-product reference prices seeded from base (pre-crisis) prices.
        # Because visit objects are recreated, the dict must be pre-populated;
        # otherwise every consumer falls back to a single scalar reference_price
        # (mean of their whole basket) which breaks per-product Prospect Theory.
        # Using base_price (not current_price) ensures the reference is always the
        # pre-inflation price, so every crisis day feels correctly expensive.
        self._ref_prices: dict = profile.setdefault("_ref_prices", {})
        for product_id, pa in model.product_map.items():
            self._ref_prices.setdefault(product_id, pa.base_price)

        # Phase-two calibration identifies retain/drop behaviour for observed
        # grocery needs across categories. It remains separate from the controlled
        # milk DCE, whose pooled price-and-attribute utility is used only to allocate
        # a replacement after the phase-transition rule says replacement occurs.
        self.price_acceptance_margin = float(
            profile.get("revealed_preference_margin", 0.05)
        )

        # ── Theory of Planned Behaviour (Ajzen 1991) ────────────────────────────
        # Weights from Armitage & Conner (2001) meta-analysis
        self.attitude         = max(0.3, 1.0 - self.price_sensitivity * 0.4)
        self.subjective_norm  = 0.0   # updated each step from store-crowding signal
        self.pbc              = 1.0   # perceived behavioural control
        # The published transferred values sum to 1.14. Treating them as raw
        # additive weights and clipping at one distorted high-intention agents.
        # A normalized convex combination preserves their relative importance.
        (self._tpb_att_w, self._tpb_norm_w, self._tpb_pbc_w) = (
            normalized_tpb_weights()
        )

        # ── Temporal Discounting / Stockpiling (O'Donoghue & Rabin 1999) ───────
        # Heuristic inspired by quasi-hyperbolic discounting: present-biased
        # agents stockpile more when panic rises. Beta is not estimated here.
        self.beta           = max(0.5, 0.90 - self.price_sensitivity * 0.15)

        # Home inventory persists in the household profile while each
        # ConsumerAgent represents only one store visit.
        self._home_inv: dict = profile.setdefault("_home_inv", {})

        # Allow model-level override of stockpile_days (from sidebar slider)
        if getattr(model, "stockpile_days_override", None) is not None:
            base_days = float(model.stockpile_days_override)
        else:
            base_days = float(profile.get("stockpile_days", 1.0))
        self.stockpile_days  = base_days   # rises with panic during step()

        # ── FIES (FAO Food Insecurity Experience Scale, simplified 4-item) ─────
        self.access_stress_score = int(profile.get("_access_stress_score", 0))
        # Deprecated alias retained for historical exports and UI code. This
        # objective model diagnostic is not the survey-based FAO FIES.
        self.food_insecurity_score = self.access_stress_score
        self.items_unmet           = 0   # items wanted but not obtained
        self._bought_organic       = False
        self._panic_signal_sent    = False
        self.substitution_attempts = 0
        self.substitution_candidates_considered = 0
        self.substitution_price_rejections = 0

    # ------------------------------------------------------------------
    # Behavioural theory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kt_value(delta: float, loss_aversion: float = 2.25, alpha: float = 0.88) -> float:
        """
        Kahneman-Tversky (1979) value function.
        delta > 0 = gain relative to reference (price fell / stock better than expected)
        delta < 0 = loss  (price rose / stockout surprise)
        Asymmetry: losses weighted ~2.25× more than equivalent gains (loss aversion).
        """
        if delta >= 0:
            return delta ** alpha
        return -(loss_aversion * ((-delta) ** alpha))

    def _tpb_intention(self) -> float:
        """
        Behavioural intention from Theory of Planned Behaviour (Ajzen 1991).
        Uses the relative Armitage & Conner (2001) transferred weights after
        normalization to sum to one:
          I = 0.430·A + 0.228·SN + 0.342·PBC
        Returns a value in [0, 1] representing purchase motivation strength.
        """
        return tpb_intention(
            self.attitude,
            self.subjective_norm,
            self.pbc,
            (self._tpb_att_w, self._tpb_norm_w, self._tpb_pbc_w),
        )

    def _price_acceptance_threshold(self, intention: float = 0.50) -> float:
        """Return the maximum accepted incremental price loss.

        TPB relief is an exploratory extension. In empirical-only mode intention
        is exactly 0.50, so the held-out calibrated margin is unchanged.
        """
        return max(
            0.0,
            self.price_acceptance_margin
            + (intention - 0.50) * 0.12
            + self.price_tolerance_extra,
        )

    @staticmethod
    def _effective_price(product: ProductAgent) -> float:
        return product.current_price * (0.5 if product._has_near_expiry else 1.0)

    def _price_loss(
        self, product: ProductAgent, reference_price: float | None = None,
    ) -> float:
        """Incremental price loss relative to an explicit comparison price."""
        effective_price = self._effective_price(product)
        ref_price = float(
            reference_price
            if reference_price is not None
            else self._ref_prices.get(product.prod_id, product.base_price)
        )
        ref_price = max(ref_price, 0.01)
        if self.model.prospect_theory_enabled:
            price_delta = (ref_price - effective_price) / ref_price
            kt_value = self._kt_value(
                price_delta, self.loss_aversion, self.kt_alpha
            )
            relative_multiplier = max(0.01, 1.0 - kt_value * 0.6)
            return self.price_sensitivity * (relative_multiplier - 1.0)
        return self.price_sensitivity * (effective_price / ref_price - 1.0)

    def _accepts_price(
        self,
        product: ProductAgent,
        intention: float = 0.50,
        reference_price: float | None = None,
    ) -> bool:
        return self._price_loss(product, reference_price) <= (
            self._price_acceptance_threshold(intention) + 1e-12
        )

    def _dce_candidate_utility(self, product: ProductAgent) -> float:
        """Pooled DCE utility for candidates on the recorded-price scale."""
        generic = self.model.dce_generic_feature_coefficients
        if generic:
            utility = generic.get("price", 0.0) * self._effective_price(product)
            utility += sum(
                coefficient * float(product.choice_attributes.get(attribute, 0.0))
                for attribute, coefficient in generic.items()
                if attribute not in {"price", "optout"}
            )
            return utility
        fat_centered = float(product.fat_content) - 1.5
        coefficients = self.model.dce_choice_coefficients
        return (
            coefficients.get("price", 0.0) * self._effective_price(product)
            + coefficients.get("origin", 0.0) * float(product.origin == "Suomi")
            + coefficients.get("organic", 0.0) * float(product.is_bio)
            + coefficients.get("fat_linear", 0.0) * fat_centered
            + coefficients.get("fat_quadratic", 0.0) * fat_centered ** 2
        )

    def _nonprice_compatibility(self, product: ProductAgent) -> float:
        """Transparent 0-1 descriptive fit used only to rank substitutes.

        The three components come directly from participant choice shares or the
        chosen-fat mean. Equal aggregation is an explicit allocation heuristic;
        it is never compared with price or interpreted as cardinal utility/WTP.
        """
        if (
            not self.model.dce_nonprice_validation_passed
            or str(product.category).strip().casefold()
            not in self.model.dce_applicable_categories
        ):
            return 0.5
        finnish_fit = (
            self.finnish_preference
            if product.origin == "Suomi"
            else 1.0 - self.finnish_preference
        )
        organic_fit = (
            self.organic_preference
            if product.is_bio
            else 1.0 - self.organic_preference
        )
        fat_fit = math.exp(-abs(product.fat_content - self.preferred_fat) / 2.0)
        origin_weight = organic_weight = fat_weight = 1.0
        policy: PolicyConfig = self.model.policy_config
        if (
            self.model.policy_choice_effects_enabled
            and policy.is_labelling_active(self.model.current_day)
        ):
            organic_weight += policy.labelling_organic_boost
            fat_weight += policy.labelling_health_boost
        return (
            origin_weight * finnish_fit
            + organic_weight * organic_fit
            + fat_weight * fat_fit
        ) / (origin_weight + organic_weight + fat_weight)

    def _baseline_product_utility(self, product: ProductAgent) -> float:
        """Deprecated compatibility alias retained for downstream consumers."""
        return self._nonprice_compatibility(product)

    # ------------------------------------------------------------------
    # Utility computation
    # ------------------------------------------------------------------

    def _compute_utility(self, product: ProductAgent) -> float:
        """
        Return the descriptive non-price compatibility score used only by legacy
        or explicitly validated ranking paths. The pooled milk DCE has its own
        price-and-attribute utility on the displayed-EUR scale.
        """
        return self._nonprice_compatibility(product)

    # ------------------------------------------------------------------
    # Substitution search
    # ------------------------------------------------------------------

    def _find_best_substitute(
        self,
        category: str,
        wanted_qty: int,
        exclude_product_id: str,
        wanted_product: ProductAgent | None = None,
        remaining_budget: float | None = None,
        intention: float = 0.50,
    ) -> ProductAgent | None:
        """
        Return an affordable in-stock product in the same catalogue category.

        The phase-two retain/drop price threshold is applied to replacement
        candidates only when the replacement-event audit supports that transfer.
        Likewise, non-price compatibility is a deterministic ordering only in
        categories that clear the replacement-ranking validation gate. Otherwise
        a seeded uniform draw makes the unidentified allocation rule explicit and
        propagates it through Monte Carlo runs.
        """
        self.substitution_attempts += 1
        # Respect archetype substitution tolerance
        if self.model._day_rng.random() > self.sub_tolerance:
            return None   # agent refuses to substitute

        normalized_category = str(category).strip().casefold()
        candidates = [
            a for a in self.model.products
            if str(a.category).strip().casefold() == normalized_category
            and a.prod_id != exclude_product_id
            and a.stock_shelf > 0
        ]
        self.substitution_candidates_considered += len(candidates)
        if not candidates:
            return None

        reference_price = (
            wanted_product.base_price
            if wanted_product is not None
            else max(0.01, self.reference_price)
        )
        eligible = []
        for candidate in candidates:
            effective_price = self._effective_price(candidate)
            if remaining_budget is not None and effective_price > remaining_budget + 1e-9:
                self.substitution_price_rejections += 1
                continue
            if (
                self.model.substitution_price_gate_supported
                and not self._accepts_price(
                    candidate,
                    intention=intention,
                    reference_price=reference_price,
                )
            ):
                self.substitution_price_rejections += 1
                continue
            eligible.append(candidate)
        if not eligible:
            return None

        if (
            self.model.dce_price_choice_supported
            and normalized_category in self.model.dce_applicable_categories
        ):
            utilities = [self._dce_candidate_utility(candidate) for candidate in eligible]
            maximum = max(utilities)
            weights = [math.exp(value - maximum) for value in utilities]
            return self.model._day_rng.choices(eligible, weights=weights, k=1)[0]
        if normalized_category in self.model.substitution_transition_categories:
            empirical_weights = self.model.substitution_transition_weights.get(
                normalized_category, {}
            )
            weights = [
                max(0.0, float(empirical_weights.get(candidate.prod_id, 0.0)))
                for candidate in eligible
            ]
            if sum(weights) > 0:
                return self.model._day_rng.choices(
                    eligible, weights=weights, k=1
                )[0]
        if normalized_category in self.model.substitution_ranking_categories:
            return min(
                eligible,
                key=lambda candidate: (
                    -self._nonprice_compatibility(candidate),
                    self._effective_price(candidate),
                    candidate.prod_id,
                ),
            )
        return self.model._day_rng.choice(eligible)

    # ------------------------------------------------------------------
    # Purchase execution (FIFO with near-expiry discount)
    # ------------------------------------------------------------------

    def _execute_purchase(
        self,
        product: ProductAgent,
        wanted_qty: int,
        remaining_budget: float,
        is_substitute: bool = False,
        pantry_key: str | None = None,
    ) -> tuple[float, int]:
        """
        Deduct stock from shelf batches (oldest first) and record revenue.
        Returns ``(money_spent, units_purchased)``.
        """
        # ── Nudge / Choice Architecture (Thaler & Sunstein 2008) ─────────────
        # Enforce per-product purchase limit if a rationing policy is active.
        # This is a soft paternalistic intervention; agents with higher panic
        # would otherwise hoard more than socially optimal.
        if self.model.purchase_limit is not None:
            wanted_qty = min(wanted_qty, self.model.purchase_limit)

        qty_left  = wanted_qty
        cost_paid = 0.0

        for batch in product.shelf_batches:
            if qty_left <= 0:
                break
            days_left  = product.max_shelf_life - batch["age"]
            unit_price = product.current_price
            if days_left <= ProductAgent.NEAR_EXPIRY_DAYS:
                unit_price *= 0.5
            unit_price = round(unit_price, 2)

            # Monetary constraints operate in cents. Without cent rounding,
            # catalogue totals such as 12.3000003 falsely reject the final item
            # from a €12.30 experimental budget.
            budget_left = round(max(0.0, remaining_budget - cost_paid), 2)
            affordable_qty = int(math.floor((budget_left + 1e-9) / unit_price))
            if affordable_qty <= 0:
                self.budget_exhausted = True
                break

            desired_take = min(qty_left, batch["qty"])
            take = min(desired_take, affordable_qty)
            batch["qty"] -= take
            qty_left     -= take
            cost_paid     = round(cost_paid + take * unit_price, 2)

            if take < desired_take:
                self.budget_exhausted = True

            if days_left <= ProductAgent.NEAR_EXPIRY_DAYS:
                product.daily_near_expiry_sold += take

        # Remove empty batches
        product.shelf_batches = [b for b in product.shelf_batches if b["qty"] > 0]

        qty_purchased = wanted_qty - qty_left
        if qty_purchased > 0:
            product.daily_sales        += qty_purchased
            product.daily_revenue      += cost_paid
            product.daily_base_revenue += qty_purchased * product.base_price   # constant-price revenue
            if is_substitute:
                product.daily_substitutions += qty_purchased
                self.items_substituted += qty_purchased

            # CO2 attribution
            is_finnish = product.is_domestic
            co2_factor = CO2_PRODUCTION.get((is_finnish, product.is_bio), 2.2)
            product.daily_co2_sales += qty_purchased * co2_factor

            # Import dependency tracking
            if is_finnish:
                product.daily_domestic_sales += qty_purchased
            else:
                product.daily_import_sales   += qty_purchased

            # Consumer welfare tracking
            self.items_purchased  += qty_purchased
            self.total_fat_bought += product.fat_content * qty_purchased

            # ── Update home inventory (stockpiling model) ────────────────────
            inventory_key = pantry_key or product.prod_id
            self._home_inv[inventory_key] = (
                self._home_inv.get(inventory_key, 0.0) + qty_purchased
            )
            # Reference-price adaptation is part of the optional Prospect Theory
            # extension. In empirical-only mode the observed catalogue price stays
            # fixed as the transparent comparison point across the scenario.
            if self.model.prospect_theory_enabled:
                old_ref = self._ref_prices.get(
                    product.prod_id, product.base_price
                )
                self._ref_prices[product.prod_id] = round(
                    0.85 * old_ref + 0.15 * product.current_price, 4
                )
            if product.is_bio:
                self._bought_organic = True

        return cost_paid, qty_purchased

    # ------------------------------------------------------------------
    # Behavioural learning
    # ------------------------------------------------------------------

    def _update_preferences(self, bought_organic: bool, mean_fat_today: float):
        """
        Softly shift agent preferences after one shopping trip.

        Rules are archetype-specific and use a small LEARNING_RATE so that
        meaningful preference drift accumulates over weeks rather than days.

        price_champion
            If budget was exhausted today → nudge price_sensitivity upward
            (agent becomes even more price-conscious after financial stress).

        green_buyer
            If ≥1 organic item purchased → extend organic streak, reinforce
            organic_preference toward 1.0.
            If no organic purchased → streak resets, slight downward drift.

        health_optimizer
            Preferred fat level drifts toward the actual mean fat content
            of today's basket (reinforcement learning on actual behaviour).

        habitual_buyer
            If basket was fully fulfilled → tighten substitution tolerance
            (agent becomes more rigid after successful trips).
            If budget was exhausted → loosen it slightly.
        """
        lr = LEARNING_RATE

        if self.archetype == "price_champion":
            if self.budget_exhausted:
                self.price_sensitivity = min(
                    1.0, self.price_sensitivity + lr * (1.0 - self.price_sensitivity)
                )

        elif self.archetype == "green_buyer":
            if bought_organic:
                self._organic_streak += 1
                # Faster reinforcement with longer streaks (up to 3×)
                boost = min(3, self._organic_streak) * lr
                self.organic_preference = min(
                    1.0, self.organic_preference + boost * (1.0 - self.organic_preference)
                )
            else:
                self._organic_streak = 0
                # Small erosion when organic wasn't chosen/available
                self.organic_preference = max(
                    0.0, self.organic_preference - lr * 0.5
                )

        elif self.archetype == "health_optimizer":
            if mean_fat_today > 0:
                # Slowly move preferred fat toward experienced fat
                self._fat_history.append(mean_fat_today)
                if len(self._fat_history) > 30:
                    self._fat_history.pop(0)
                recent_avg = sum(self._fat_history) / len(self._fat_history)
                self.preferred_fat += lr * (recent_avg - self.preferred_fat)
                self.preferred_fat  = max(0.0, min(5.0, self.preferred_fat))

        elif self.archetype == "habitual_buyer":
            fulfillment_today = (
                self.items_purchased / max(1, self.items_wanted)
            )
            if fulfillment_today >= 0.9:
                # Successful trip → become more loyal (less willing to substitute)
                self.sub_tolerance = max(
                    0.05, self.sub_tolerance - lr * self.sub_tolerance
                )
            elif self.budget_exhausted:
                # Financial stress → become slightly more open to substitutes
                self.sub_tolerance = min(
                    0.80, self.sub_tolerance + lr * (0.80 - self.sub_tolerance)
                )

        # ---- Write updated preferences back to the shared profile dict ----
        # The profile dict is a reference into population_pool, so this persists
        # the learned values across visits.
        self.profile["price_sensitivity"]  = round(self.price_sensitivity,  4)
        self.profile["organic_preference"] = round(self.organic_preference, 4)
        self.profile["preferred_fat"]      = round(self.preferred_fat,      4)
        # sub_tolerance isn't stored in the profile by default; store it so
        # future ConsumerAgents created from the same profile inherit it.
        self.profile["sub_tolerance"]      = round(self.sub_tolerance,      4)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self):
        # Reset per-day welfare counters
        self.items_wanted      = 0
        self.items_base_wanted = 0
        self.items_allowed     = 0
        self.items_purchased   = 0
        self.items_substituted = 0
        self.choice_lines_purchased = 0
        self.choice_lines_substituted = 0
        self.budget_exhausted  = False
        self.total_fat_bought  = 0.0
        self._bought_organic   = False
        self._panic_signal_sent = False
        self.amount_spent      = 0.0
        self.substitution_attempts = 0
        self.substitution_candidates_considered = 0
        self.substitution_price_rejections = 0

        # Phase-one basket defines household needs. The observed phase-two basket
        # is reserved for calibration/validation; using it here would leak the
        # crisis outcome into the simulation and then apply price response twice.
        active_basket = self.baseline_basket
        if self.model.is_scenario_active:
            # A continuous reservation-spending share becomes a lumpy package
            # budget in the ABM. Adding half a typical unit price removes the
            # systematic downward bias from indivisible products without allowing
            # expenditure above the participant's stated maximum budget.
            active_budget = round(
                min(
                    self.crisis_budget,
                    self.crisis_budget * self.budget_utilization_propensity
                    + 0.5 * self.reference_price,
                ),
                2,
            )
        else:
            active_budget  = self.budget

        # TPB variables below are constructed model states, not measured TPB
        # constructs in the current export. They alter choices only in explicit
        # exploratory runs; the empirical default makes no threshold adjustment.
        if self.model.tpb_enabled:
            crowd_ratio = (
                self.model.daily_consumer_count
                / max(1, self.model.base_consumers)
            )
            self.subjective_norm = min(
                1.0, crowd_ratio * 0.40 + self.panic_level * 0.60
            )
            income_factor = min(1.0, self.income_midpoint / 3000.0)
            self.pbc = max(0.10, income_factor - self.panic_level * 0.35)
            intention = self._tpb_intention()
        else:
            self.subjective_norm = 0.0
            self.pbc = 1.0
            intention = 0.50

        # ── Temporal discounting: precautionary cover rises with panic ───────
        if self.model.panic_dynamics_enabled:
            base_stockpile_days = (
                float(self.model.stockpile_days_override)
                if self.model.stockpile_days_override is not None
                else float(self.profile.get("stockpile_days", 1.0))
            )
        else:
            base_stockpile_days = 1.0
        self.stockpile_days = max(
            1.0,
            base_stockpile_days
            + (self.panic_level * 3.0 if self.model.panic_dynamics_enabled else 0.0),
        )
        spent = 0.0

        for item in active_basket:
            wanted_name = item["product_name"]
            base_qty    = max(1, int(item.get("quantity", 1)))
            wanted_qty  = base_qty
            category    = item.get("category", "")
            product     = self.model.get_product_by_id(item.get("product_id"))
            if product is None:
                product = self.model.get_product_by_name(wanted_name)
            inventory_key = product.prod_id if product else item.get("product_id", wanted_name)
            self.items_base_wanted += base_qty

            # ── Pantry-adjusted replenishment demand ────────────────────────────
            # The experimental basket is one shopping occasion, not one day of
            # consumption. Replenish routine cover plus a precautionary buffer.
            home_have = float(self._home_inv.get(inventory_key, 0.0))
            daily_need = base_qty / max(1.0, self.expected_visit_interval)
            precautionary_days = max(0.0, self.stockpile_days - 1.0) * self.beta
            cover_days = self.expected_visit_interval + precautionary_days
            pantry_target = daily_need * cover_days
            pantry_gap = max(0.0, pantry_target - home_have)
            # Unbiased stochastic rounding avoids turning a 0.05-unit buffer into
            # one whole extra package for every SKU and every household.
            wanted_qty = int(math.floor(pantry_gap))
            if self.model._day_rng.random() < pantry_gap - wanted_qty:
                wanted_qty += 1

            # Continuous panic amplification: no arbitrary activation threshold.
            # Both panic and the cross-fitted household propensity must be non-zero.
            effective_hoarding = (
                effective_hoarding_multiplier(
                    self.model.hoarding_factor,
                    self.profile.get("hoarding_propensity", 0.0),
                    self.panic_level,
                )
                if self.model.panic_dynamics_enabled
                else 1.0
            )
            hoarded_demand = wanted_qty * effective_hoarding
            wanted_qty = int(math.floor(hoarded_demand))
            if self.model._day_rng.random() < hoarded_demand - wanted_qty:
                wanted_qty += 1

            self.items_wanted += wanted_qty
            allowed_qty = wanted_qty
            if self.model.purchase_limit is not None:
                allowed_qty = min(allowed_qty, self.model.purchase_limit)
            self.items_allowed += allowed_qty

            if spent >= active_budget:
                self.budget_exhausted = True
                continue

            # --- Product not in catalogue (shouldn't happen after validation) ---
            if not product:
                substitute = self._find_best_substitute(
                    category,
                    allowed_qty,
                    "",
                    wanted_product=None,
                    remaining_budget=active_budget - spent,
                    intention=intention,
                )
                if substitute:
                    cost, bought = self._execute_purchase(
                        substitute, allowed_qty, active_budget - spent,
                        is_substitute=True,
                        pantry_key=inventory_key,
                    )
                    spent += cost
                    if bought > 0:
                        self.choice_lines_purchased += 1
                        self.choice_lines_substituted += 1
                continue

            # Phase-transition substitution is an observed response in its own
            # right, not merely a reaction to stockout. The cross-fitted
            # participant propensity supplies the single probability gate inside
            # ``_find_best_substitute``. Candidate allocation remains subject to
            # the separately reported replacement-choice evidence limitations.
            if self.model.is_scenario_active and self._accepts_price(
                product, intention=intention
            ):
                proactive_substitute = self._find_best_substitute(
                    category,
                    allowed_qty,
                    product.prod_id,
                    wanted_product=product,
                    remaining_budget=active_budget - spent,
                    intention=intention,
                )
                if proactive_substitute is not None:
                    cost, bought = self._execute_purchase(
                        proactive_substitute,
                        allowed_qty,
                        active_budget - spent,
                        is_substitute=True,
                        pantry_key=inventory_key,
                    )
                    spent += cost
                    if bought > 0:
                        self.choice_lines_purchased += 1
                        self.choice_lines_substituted += 1
                    unmet = max(0, allowed_qty - bought)
                    if unmet > 0:
                        product.daily_lost_sales += unmet
                        reason = "Price" if self.budget_exhausted else "Stockout"
                        self.model.track_loss(
                            reason, unmet * product.current_price
                        )
                    continue

            # --- Utility check ---
            if not self._accepts_price(product, intention=intention):
                # During a phase-transition visit the calibrated substitution
                # decision was already sampled once above. Re-sampling here would
                # turn one participant probability into 1-(1-p)^2 for rejected
                # products. Baseline stock/price failures still receive one gate.
                sub = None
                if not self.model.is_scenario_active:
                    sub = self._find_best_substitute(
                        category,
                        allowed_qty,
                        product.prod_id,
                        wanted_product=product,
                        remaining_budget=active_budget - spent,
                        intention=intention,
                    )
                bought = 0
                if sub:
                    cost, bought = self._execute_purchase(
                        sub, allowed_qty, active_budget - spent,
                        is_substitute=True,
                        pantry_key=inventory_key,
                    )
                    spent += cost
                if bought > 0:
                    self.choice_lines_purchased += 1
                    self.choice_lines_substituted += 1
                unmet = max(0, allowed_qty - bought)
                product.daily_lost_sales += unmet
                self.model.track_loss("Price", unmet * product.current_price)
                continue

            # --- Buy available stock, then seek a substitute for any remainder ---
            # A fully empty shelf is the strongest scarcity signal. Previously
            # only shoppers who bought the final units could emit a signal, so a
            # prolonged stockout paradoxically produced less panic than a nearly
            # empty shelf. Keep the one-signal-per-visit guard.
            if product.stock_shelf < 3 and not self._panic_signal_sent:
                self.model.add_panic_signal()
                self._panic_signal_sent = True
            direct_target = min(allowed_qty, product.stock_shelf)
            direct_bought = 0
            if direct_target > 0:
                cost, direct_bought = self._execute_purchase(
                    product, direct_target, active_budget - spent
                )
                spent += cost
                if product.stock_shelf < 3 and not self._panic_signal_sent:
                    self.model.add_panic_signal()
                    self._panic_signal_sent = True

            remaining = max(0, allowed_qty - direct_bought)
            sub_bought = 0
            if remaining > 0 and spent < active_budget:
                sub = self._find_best_substitute(
                    category,
                    remaining,
                    product.prod_id,
                    wanted_product=product,
                    remaining_budget=active_budget - spent,
                    intention=intention,
                )
                if sub:
                    cost, sub_bought = self._execute_purchase(
                        sub, remaining, active_budget - spent,
                        is_substitute=True,
                        pantry_key=inventory_key,
                    )
                    spent += cost

            unmet = max(0, remaining - sub_bought)
            if direct_bought + sub_bought > 0:
                self.choice_lines_purchased += 1
            if sub_bought > 0:
                self.choice_lines_substituted += 1
            if unmet > 0:
                product.daily_lost_sales += unmet
                reason = "Price" if self.budget_exhausted else "Stockout"
                self.model.track_loss(reason, unmet * product.current_price)

        self.amount_spent = round(spent, 4)
        self.profile["_last_shop_day"] = self.model.current_day
        self.profile["_visit_count"] = self.visit_number

        # ── FIES (FAO Food Insecurity Experience Scale, 4-item simplified) ──────
        self.items_unmet = max(0, self.items_wanted - self.items_purchased)
        self.shopping_shortfall_rate = (
            self.items_unmet / self.items_wanted if self.items_wanted > 0 else 0.0
        )
        self.access_stress_score = int(
            self.profile.get("_access_stress_score", 0)
        )
        self.food_insecurity_score = self.access_stress_score

        # ---- Behavioural learning ----
        mean_fat_today = (
            self.total_fat_bought / max(1, self.items_purchased)
        ) if self.items_purchased > 0 else 0.0
        if self.model.preference_learning_enabled:
            self._update_preferences(self._bought_organic, mean_fat_today)


# ---------------------------------------------------------------------------
# 5. Supermarket Model
# ---------------------------------------------------------------------------

class SupermarketModel(Model):
    """
    Main ABM.  Accepts either a pre-built config dict (from data_processor)
    or a path to a mesa_config.json file.

    Household scheduling logic
    --------------------------
    Population profiles are persistent households. Each receives a stable identity,
    pantry, expected revisit interval, and next-visit day. Evidence-only runs use the
    declared constant daily traffic. Unvalidated seasonality, weekday multipliers,
    and seeded traffic noise require explicit opt-in. The model selects the most-due
    unique households instead of drawing anonymous visits.
    """

    # Seasonality multipliers (month index 1–12)
    SEASONALITY = {
        1: 0.90, 2: 0.90, 3: 1.00, 4: 1.00, 5: 1.00, 6: 1.10,
        7: 1.10, 8: 1.00, 9: 1.00, 10: 1.00, 11: 1.10, 12: 1.30,
    }
    # Weekday multipliers (0=Mon … 6=Sun). Finnish grocery stores open all 7 days.
    # Sunday has lighter traffic (~70 % of Monday). Saturday is busiest.
    WEEKDAY_WEIGHTS = {
        0: 0.80,   # Monday   — quiet start
        1: 0.90,   # Tuesday
        2: 0.90,   # Wednesday
        3: 1.00,   # Thursday
        4: 1.10,   # Friday   — pre-weekend stock-up
        5: 1.30,   # Saturday — busiest
        6: 0.70,   # Sunday   — reduced hours, lighter traffic
    }

    # ── Store size tiers (consumers/day → label) ────────────────────────────
    STORE_TIERS = [
        (  200, "Small (neighbourhood shop)"),
        (  500, "Medium (supermarket)"),
        ( 1500, "Large (hypermarket)"),
        (99999, "Very Large (wholesale / hyper)"),
    ]

    @staticmethod
    def _calibrate_store_capacities(
        products: list[dict],
        population: list[dict],
        base_consumers: int,
        lead_time: int,
    ) -> dict[str, dict]:
        """
        Compute realistic stock capacities for each product based on daily demand.

        Without calibration a store with 500 consumers gets the same
        max_shelf_capacity=20 as one with 20 consumers, creating perpetual
        stockouts or (with small stores) wasteful over-ordering.

        Formula
        -------
        • Estimate avg daily demand per product  =  base_consumers
                                                   × avg_basket_qty_for_that_product
        • max_shelf_capacity  =  demand × shelf_cover_days
            shelf_cover_days depends on shelf life:
              perishable (≤7 d)  → 1.5 days  (stock rotates fast; avoid big batches)
              medium (8–30 d)    → 2.5 days
              dry / canned (>30) → 4.0 days
        • max_storage_capacity  =  demand × (lead_time + 4-day safety buffer)
            minimum: 2 × max_shelf_capacity
        • initial_stock_shelf    =  max_shelf_capacity × 0.75  (store is stocked at start)
        • initial_stock_storage  =  max_storage_capacity × 0.60

        Returns a dict  product_id → {max_shelf_capacity, max_storage_capacity,
                                      initial_stock_shelf, initial_stock_storage}
        """
        # ── 1. Average basket quantity per product across population ─────────
        basket_totals: dict[str, float] = {}
        category_totals: dict[str, float] = defaultdict(float)
        for profile in population:
            for item in profile.get("baseline_basket", []):
                product_key = str(item.get("product_id") or item.get("product_name", ""))
                qty = float(item.get("quantity", 1))
                basket_totals[product_key] = basket_totals.get(product_key, 0.0) + qty
                category = str(item.get("category", "")).strip().casefold()
                if category:
                    category_totals[category] += qty

        n_pop = max(1, len(population))
        avg_qty: dict[str, float] = {n: v / n_pop for n, v in basket_totals.items()}
        category_avg_qty = {
            category: total / n_pop for category, total in category_totals.items()
        }

        # Identify catalogue products with direct basket evidence. Category
        # demand is used only for the unallocated remainder; it is never added
        # on top of already observed SKU demand.
        product_demand: dict[str, float | None] = {}
        observed_by_category: dict[str, float] = defaultdict(float)
        unobserved_count_by_category: dict[str, int] = defaultdict(int)
        for prod in products:
            product_id = str(prod.get("id") or prod.get("name", ""))
            product_name = str(prod.get("name", ""))
            category = str(prod.get("category", "")).strip().casefold()
            observed = avg_qty.get(product_id)
            if observed is None:
                observed = avg_qty.get(product_name)
            product_demand[product_id] = observed
            if observed is None:
                unobserved_count_by_category[category] += 1
            else:
                observed_by_category[category] += observed

        # ── 2. Per-product calibration ───────────────────────────────────────
        result: dict[str, dict] = {}
        for prod in products:
            product_id = str(prod.get("id") or prod.get("name", ""))
            shelf_life = int(prod.get("shelf_life_days", 7))

            # How many days of stock to keep on the shelf
            if shelf_life <= 7:
                shelf_cover = 1.5    # perishable — rotate quickly
            elif shelf_life <= 30:
                shelf_cover = 2.5    # medium (yogurt, cheese, …)
            else:
                shelf_cover = 4.0    # dry / canned goods

            # Expected daily demand for this product. The former generic
            # fallback of 0.5 units per household assigned 100 daily units to
            # every unobserved SKU in a 200-visitor store, creating extreme
            # capacities and synchronized refill spikes. Allocate only the
            # category demand not already attributed to observed products.
            category = str(prod.get("category", "")).strip().casefold()
            observed = product_demand[product_id]
            if observed is not None:
                per_household_demand = observed
                demand_basis = "observed_product"
            else:
                unallocated_category_demand = max(
                    0.0,
                    category_avg_qty.get(category, 0.0)
                    - observed_by_category.get(category, 0.0),
                )
                missing_count = max(1, unobserved_count_by_category.get(category, 1))
                per_household_demand = unallocated_category_demand / missing_count
                demand_basis = (
                    "unallocated_category_share"
                    if per_household_demand > 0
                    else "minimum_floor_no_product_evidence"
                )
            daily_demand = max(1.0, base_consumers * per_household_demand)

            max_shelf   = max(10, int(math.ceil(daily_demand * shelf_cover)))
            storage_days = lead_time + 4   # lead-time + safety buffer
            max_storage = max(max_shelf * 2, int(math.ceil(daily_demand * storage_days)))

            result[product_id] = {
                "max_shelf_capacity":   max_shelf,
                "max_storage_capacity": max_storage,
                "initial_stock_shelf":  int(max_shelf   * 0.75),
                "initial_stock_storage": int(max_storage * 0.60),
                "estimated_daily_demand": round(daily_demand, 6),
                "demand_basis": demand_basis,
            }

        return result

    @staticmethod
    def store_tier_label(base_consumers: int) -> str:
        for threshold, label in SupermarketModel.STORE_TIERS:
            if base_consumers < threshold:
                return label
        return "Very Large"

    def __init__(
        self,
        config_data:     dict  = None,   # in-memory config dict (preferred)
        config_file:     str   = None,   # fallback: path to mesa_config.json
        base_consumers:  int   = 50,
        start_month:     int   = 1,
        reorder_pt:      float = 0.30,
        target_stock:    float = 0.90,
        lead_time:       int   = 2,
        is_crisis_mode:  bool  = False,
        scenario_start_day: int = 30,
        crisis_duration: int   = 0,    # days the crisis lasts (0 = runs to end of sim)
        inflation_pct:   float = 0.0,
        disruption_days: int   = 0,
        panic_sens:      float = 0.50,
        hoarding_fac:    float = 1.50,
        fixed_seed:      int   = 42,
        ai_recs:         dict  = None,   # {product_name: storage_capacity}
        policy_cfg:      dict  = None,   # PolicyConfig kwargs dict
        # ── New behavioural theory parameters ────────────────────────────────
        purchase_limit:  int   = None,   # Nudge: max units per product per visit
        media_intensity: float = 0.0,    # Media channel strength (0–1)
        communication_type: str = "neutral",  # "neutral" | "panic" | "calming"
        stockpile_days_override: float = None,  # Optional: override per-agent stockpile horizon
        panic_exposure_floor: float = 0.10,
        panic_growth_rate: float = 0.50,
        panic_decay_active: float = 0.05,
        panic_decay_recovery: float = 0.10,
        inflation_panic_rate: float = 0.40,
        # Unidentified dynamic mechanisms are opt-in.  The default model keeps
        # only behaviour supported by the GROCERYsim observations/calibration.
        enable_panic_dynamics: bool = False,
        enable_tpb: bool = False,
        enable_prospect_theory: bool = False,
        enable_preference_learning: bool = False,
        enable_archetype_modifiers: bool = False,
        enable_policy_choice_effects: bool = False,
        enable_traffic_variation: bool = False,
        scenario_price_overrides: dict[str, float] | None = None,
    ):
        super().__init__()
        # Mesa's scheduler is used as an agent registry only.  Daily execution
        # is explicitly phased in ``step`` so product counter resets, logistics,
        # shopping, and aggregation cannot interleave randomly.
        self.random.seed(fixed_seed)
        self.schedule = BaseScheduler(self)

        # Seeded RNG for all explicit random calls within the model
        self.fixed_seed  = fixed_seed
        self._day_rng    = random.Random(fixed_seed)

        # General parameters
        self.base_consumers    = base_consumers
        self.traffic_variation_enabled = bool(enable_traffic_variation)
        self.scenario_price_overrides = {
            str(product_id): float(price)
            for product_id, price in (scenario_price_overrides or {}).items()
            if float(price) > 0
        }
        if not 1 <= int(start_month) <= 12:
            raise ValueError("start_month must be between 1 and 12.")
        self.start_month       = int(start_month)
        self.current_month     = self.start_month
        self.current_weekday   = 0
        self.current_day       = 0

        # Logistics parameters
        self.reorder_point      = float(reorder_pt)
        self.target_stock_level = float(target_stock)
        self.lead_time_days     = int(lead_time)
        if not 0.0 < self.reorder_point < 1.0:
            raise ValueError("reorder_pt must be a capacity fraction between 0 and 1.")
        if not 0.0 < self.target_stock_level <= 1.0:
            raise ValueError("target_stock must be a capacity fraction in (0, 1].")
        if self.target_stock_level <= self.reorder_point:
            raise ValueError("target_stock must be greater than reorder_pt.")
        if self.lead_time_days < 1:
            raise ValueError("lead_time must be at least one day.")

        # Crisis / scenario parameters
        self.is_crisis_mode         = is_crisis_mode
        self.scenario_start_day     = scenario_start_day
        self.crisis_duration        = crisis_duration   # 0 = indefinite
        self.inflation_percent      = inflation_pct
        self.supply_disruption_days = disruption_days
        # Derived: day on which crisis ends (0 means never)
        self.scenario_end_day       = (scenario_start_day + crisis_duration) if crisis_duration > 0 else 0

        # Consumer behaviour
        self.panic_sensitivity = panic_sens
        self.hoarding_factor   = hoarding_fac

        # ── Behavioural theory parameters ───────────────────────────────────
        self.purchase_limit             = purchase_limit
        self.media_intensity            = float(media_intensity)
        self.communication_type         = communication_type
        self.stockpile_days_override    = stockpile_days_override  # None = use per-agent profile
        self.panic_exposure_floor       = float(panic_exposure_floor)
        self.panic_growth_rate          = float(panic_growth_rate)
        self.panic_decay_active         = float(panic_decay_active)
        self.panic_decay_recovery       = float(panic_decay_recovery)
        self.inflation_panic_rate       = float(inflation_panic_rate)
        self.panic_dynamics_enabled     = bool(enable_panic_dynamics)
        self.tpb_enabled                = bool(enable_tpb)
        self.prospect_theory_enabled    = bool(enable_prospect_theory)
        self.preference_learning_enabled = bool(enable_preference_learning)
        self.archetype_modifiers_enabled = bool(enable_archetype_modifiers)
        self.policy_choice_effects_enabled = bool(enable_policy_choice_effects)
        for name, value in [
            ("panic_exposure_floor", self.panic_exposure_floor),
            ("panic_growth_rate", self.panic_growth_rate),
            ("panic_decay_active", self.panic_decay_active),
            ("panic_decay_recovery", self.panic_decay_recovery),
            ("inflation_panic_rate", self.inflation_panic_rate),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        # Runtime state
        self.is_scenario_active     = False
        self._crisis_phase          = "pre"   # "pre" | "active" | "recovery"
        self.loss_reasons           = {"Stockout": 0.0, "Price": 0.0}  # cumulative totals
        self.daily_loss_reasons     = {"Stockout": 0.0, "Price": 0.0}  # reset each day
        self.global_panic_level     = 0.0
        self.panic_signals          = 0
        self.total_churned_agents   = 0
        self.daily_consumer_count   = 0
        self.requested_consumer_count = 0
        self.daily_household_consumption_demand = 0.0
        self.daily_household_consumption = 0.0
        self.daily_household_consumption_unmet = 0.0

        # Food waste accumulator
        self.food_waste_log = FoodWasteLog()

        # Policy configuration (always present; default = all policies off)
        self.policy_config = PolicyConfig(policy_cfg)

        # Stable SKU lookup plus a compatibility lookup for unique display names.
        self.product_map: dict[str, ProductAgent] = {}
        self.product_name_map: dict[str, ProductAgent] = {}
        self.products: list[ProductAgent] = []

        # Per-day output records (used by app.py for charts)
        self.daily_records: list[dict] = []

        # ---------------------------------------------------------------
        # Load config
        # ---------------------------------------------------------------
        config = config_data
        if config is None and config_file:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        if config is None:
            raise ValueError("Provide either config_data or config_file.")

        dce_validation = config.get("stats", {}).get("dce_choice_validation", {})
        self.dce_nonprice_validation_passed = bool(
            dce_validation.get("status") == "ok"
            and dce_validation.get("beats_majority_benchmark", False)
        )
        self.dce_applicable_categories = {
            str(category).strip().casefold()
            for category in dce_validation.get("applicable_categories", ["Milk"])
        }
        self.dce_price_choice_supported = bool(
            dce_validation.get("status") == "ok"
            and dce_validation.get("price_coefficient_estimable", False)
            and dce_validation.get("utility_scale_compatible_with_price", False)
            and dce_validation.get(
                "beats_null_benchmark",
                dce_validation.get("beats_majority_benchmark", False),
            )
            and dce_validation.get("model_converged", True)
        )
        self.dce_choice_coefficients = {
            "price": float(dce_validation.get("price_coefficient", 0.0) or 0.0),
            "origin": float(dce_validation.get("origin_coefficient", 0.0) or 0.0),
            "organic": float(dce_validation.get("organic_coefficient", 0.0) or 0.0),
            "fat_linear": float(dce_validation.get("fat_linear_coefficient", 0.0) or 0.0),
            "fat_quadratic": float(dce_validation.get("fat_quadratic_coefficient", 0.0) or 0.0),
        }
        self.dce_generic_feature_coefficients = {
            str(name): float(value)
            for name, value in (
                dce_validation.get("feature_coefficients", {}) or {}
            ).items()
        }
        substitution_validation = config.get("stats", {}).get(
            "substitution_choice_validation", {}
        )
        self.substitution_price_gate_supported = bool(
            substitution_validation.get("candidate_price_gate_supported", False)
        )
        self.substitution_ranking_categories = {
            str(category).strip().casefold()
            for category in substitution_validation.get(
                "supported_ranking_categories", []
            )
        }
        self.substitution_transition_categories = {
            str(category).strip().casefold()
            for category in substitution_validation.get(
                "supported_transition_categories", []
            )
        }
        self.substitution_transition_weights = {
            str(category).strip().casefold(): {
                str(product_id): float(weight)
                for product_id, weight in weights.items()
            }
            for category, weights in substitution_validation.get(
                "empirical_transition_weights", {}
            ).items()
        }
        self.substitution_choice_evidence_events = int(
            substitution_validation.get("n_unambiguous_events", 0)
        )
        self.substitution_ranking_method = (
            "dce_mnl_candidates_plus_validated_phase_transition_categories"
            if self.dce_price_choice_supported
            else "validated_phase_transition_target_shares"
            if self.substitution_transition_categories
            else "validated_participant_compatibility"
            if self.substitution_ranking_categories
            else substitution_validation.get(
                "operational_fallback",
                "seeded_uniform_affordable_same_category",
            )
        )

        # ── Auto-calibrate store stock levels to match consumer count ────────
        # Without this, max_shelf_capacity=20 is the same for 20 consumers/day
        # and 500 consumers/day — the latter will be in perpetual stockout.
        # Calibration overrides the raw product_data values with demand-scaled
        # capacities. The ai_recs dict (AI recommendation override) still
        # takes precedence when provided.
        _population_for_calibration = config.get("population", [])
        _calib = SupermarketModel._calibrate_store_capacities(
            products        = config.get("products", []),
            population      = _population_for_calibration,
            base_consumers  = base_consumers,
            lead_time       = lead_time,
        )
        self.store_calibration = _calib   # expose for app.py display
        self.store_tier        = SupermarketModel.store_tier_label(base_consumers)

        # Build ProductAgents — inject calibrated capacities into product_data copy
        import copy as _copy
        for i, p_data in enumerate(config.get("products", [])):
            p_data_cal = _copy.copy(p_data)   # shallow copy to avoid mutating original
            product_id = str(p_data.get("id", "")).strip()
            if not product_id:
                raise ValueError(f"Product {p_data.get('name', i)!r} has no stable id.")
            if product_id in self.product_map:
                raise ValueError(f"Duplicate product id {product_id!r} in catalogue.")
            if p_data["name"] in self.product_name_map:
                raise ValueError(
                    f"Duplicate product name {p_data['name']!r}. Canonicalize the "
                    "catalogue before constructing the model."
                )

            cal = _calib.get(product_id, {})
            # Only override if no explicit ai_recs for this product
            ai_cap = ai_recs.get(p_data["name"]) if ai_recs else None
            if not ai_cap:
                for field in ("max_shelf_capacity", "max_storage_capacity",
                              "initial_stock_shelf", "initial_stock_storage"):
                    if field in cal:
                        p_data_cal[field] = cal[field]
            agent  = ProductAgent(f"prod_{i}", self, p_data_cal, ai_capacity=ai_cap)
            self.schedule.add(agent)
            self.products.append(agent)
            self.product_map[product_id] = agent
            self.product_name_map[p_data["name"]] = agent

        # Add truck
        self.truck = SupplyTruck("truck_1", self)
        self.schedule.add(self.truck)

        # Population pool. New scientific configs retain only observed participant
        # profiles and declare a target simulation size. Each model seed resamples
        # complete profiles with replacement, propagating empirical-sample
        # uncertainty without fabricating combinations of attributes. Legacy/test
        # configs without population_target_size retain their supplied population.
        import copy
        empirical_profiles = config.get("population", [])
        if not empirical_profiles:
            raise ValueError("Population pool is empty — run data_processor first.")
        target_size = int(config.get("population_target_size", len(empirical_profiles)))
        if target_size < 1:
            raise ValueError("population_target_size must be positive.")
        if config.get("population_target_size") is not None:
            self.population_pool = []
            for draw_index in range(target_size):
                template = empirical_profiles[self._day_rng.randrange(len(empirical_profiles))]
                profile = copy.deepcopy(template)
                empirical_id = str(profile.get(
                    "empirical_source_id", profile.get("source_id", "participant")
                ))
                profile["empirical_source_id"] = empirical_id
                profile["source_id"] = f"{empirical_id}::seed_{fixed_seed}_draw_{draw_index}"
                profile["is_real"] = False
                profile["is_participant_resample"] = True
                profile["resample_draw_index"] = draw_index
                self.population_pool.append(profile)
            self.population_sampling_method = "complete_profile_resampling_with_replacement"
        else:
            self.population_pool = copy.deepcopy(empirical_profiles)
            self.population_sampling_method = "supplied_population"
        self.empirical_sampling_units = len(empirical_profiles)
        self._initialize_household_states()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_product_by_name(self, name: str) -> ProductAgent | None:
        return self.product_name_map.get(name)

    def get_product_by_id(self, product_id: str | None) -> ProductAgent | None:
        if product_id is None:
            return None
        return self.product_map.get(str(product_id))

    def add_panic_signal(self):
        self.panic_signals += 1

    def track_loss(self, reason: str, amount: float):
        if reason in self.loss_reasons:
            self.loss_reasons[reason]       += amount   # cumulative (never reset)
            self.daily_loss_reasons[reason] += amount   # reset each day

    def _initialize_household_states(self):
        """Create a steady-state pantry and visit calendar for every profile."""
        pool_size = len(self.population_pool)
        mean_weekday_factor = (
            sum(self.WEEKDAY_WEIGHTS.values()) / len(self.WEEKDAY_WEIGHTS)
            if self.traffic_variation_enabled else 1.0
        )
        month_factor = (
            self.SEASONALITY.get(self.start_month, 1.0)
            if self.traffic_variation_enabled else 1.0
        )
        expected_daily_visits = (
            self.base_consumers * month_factor * mean_weekday_factor
        )
        expected_daily_visits = max(1.0, min(expected_daily_visits, pool_size))
        visits_per_day = max(1, int(round(expected_daily_visits)))
        expected_interval = max(1.0, pool_size / expected_daily_visits)
        self.expected_household_visit_interval = expected_interval

        for idx, profile in enumerate(self.population_pool):
            source_id = str(profile.get("source_id", "household"))
            profile["_household_id"] = f"{source_id}:{idx}"
            profile["_expected_visit_interval"] = expected_interval
            first_visit = 1 + (idx // visits_per_day)
            profile["_next_shop_day"] = float(first_visit)
            profile["_last_shop_day"] = None
            profile["_visit_count"] = 0
            profile["_home_inv"] = {}
            profile["_daily_consumption_demand"] = 0.0
            profile["_daily_consumption_unmet"] = 0.0
            profile["_consumption_shortfall_rate"] = 0.0
            profile["_cumulative_consumption_demand"] = 0.0
            profile["_cumulative_consumption_unmet"] = 0.0
            profile["_consecutive_shortfall_days"] = 0
            profile["_access_stress_score"] = 0

            # Start in a staggered steady state: households shopping later hold
            # enough of the observed basket to cover expected consumption until
            # their first modelled visit.
            for item in profile.get("baseline_basket", []):
                inventory_key = str(
                    item.get("product_id") or item.get("product_name", "")
                )
                if not inventory_key:
                    continue
                daily_need = max(0.0, float(item.get("quantity", 1))) / expected_interval
                profile["_home_inv"][inventory_key] = (
                    profile["_home_inv"].get(inventory_key, 0.0)
                    + daily_need * first_visit
                )

    def _advance_household_consumption(self) -> tuple[float, float, float]:
        """Consume every pantry and update population-wide access outcomes.

        Access stress is an objective model diagnostic, not a psychometric food-
        insecurity scale: 0=no shortfall, 1=(0,25%), 2=[25,50%),
        3=[50,90%), and 4=[90,100%] of today's required units unmet.
        """
        demanded = consumed = unmet = 0.0
        for profile in self.population_pool:
            interval = max(1.0, float(profile.get("_expected_visit_interval", 1.0)))
            pantry = profile.setdefault("_home_inv", {})
            household_demand = household_consumed = 0.0
            for item in profile.get("baseline_basket", []):
                inventory_key = str(
                    item.get("product_id") or item.get("product_name", "")
                )
                if not inventory_key:
                    continue
                daily_need = max(0.0, float(item.get("quantity", 1))) / interval
                available = max(0.0, float(pantry.get(inventory_key, 0.0)))
                used = min(available, daily_need)
                pantry[inventory_key] = max(0.0, available - used)
                demanded += daily_need
                consumed += used
                unmet += daily_need - used
                household_demand += daily_need
                household_consumed += used

            household_unmet = max(0.0, household_demand - household_consumed)
            shortfall_rate = (
                household_unmet / household_demand if household_demand > 0 else 0.0
            )
            if shortfall_rate <= 1e-12:
                access_score = 0
            elif shortfall_rate < 0.25:
                access_score = 1
            elif shortfall_rate < 0.50:
                access_score = 2
            elif shortfall_rate < 0.90:
                access_score = 3
            else:
                access_score = 4

            profile["_daily_consumption_demand"] = household_demand
            profile["_daily_consumption_unmet"] = household_unmet
            profile["_consumption_shortfall_rate"] = shortfall_rate
            profile["_cumulative_consumption_demand"] += household_demand
            profile["_cumulative_consumption_unmet"] += household_unmet
            profile["_consecutive_shortfall_days"] = (
                int(profile.get("_consecutive_shortfall_days", 0)) + 1
                if household_unmet > 1e-12 else 0
            )
            profile["_access_stress_score"] = access_score
        self.daily_household_consumption_demand = demanded
        self.daily_household_consumption = consumed
        self.daily_household_consumption_unmet = unmet
        return demanded, consumed, unmet

    def _schedule_next_visit(self, profile: dict):
        interval = max(1.0, float(profile.get("_expected_visit_interval", 1.0)))
        profile["_next_shop_day"] = self.current_day + interval

    def _get_daily_profiles(self, target_count: int) -> list[dict]:
        """
        Return the most-due unique households for today's visits.

        Ties are shuffled with the seeded model RNG. A household can visit at most
        once per day; requested footfall above the represented population is capped
        and exposed separately in the daily output.
        """
        ranked = [
            (float(p.get("_next_shop_day", 1)), self._day_rng.random(), p)
            for p in self.population_pool
        ]
        ranked.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in ranked[:min(target_count, len(ranked))]]

    # ------------------------------------------------------------------
    # Step (one simulation day)
    # ------------------------------------------------------------------

    def step(self):
        self.current_day += 1

        # Advance calendar — 7-day week (0 Mon … 6 Sun)
        self.current_weekday = (self.current_day - 1) % 7
        # Advance month roughly every 30 days
        month_idx = (
            (self.start_month - 1) + ((self.current_day - 1) // 30)
        ) % 12
        self.current_month = (month_idx + 1)   # 1–12

        # ── Crisis phase management ──────────────────────────────────────────────
        # Phase 1 — "pre":      before scenario_start_day
        # Phase 2 — "active":   scenario_start_day … scenario_end_day (or end of sim)
        # Phase 3 — "recovery": scenario_end_day … end of sim  (prices normalised,
        #                        supply restored; panic decays naturally)
        has_shock = (
            self.inflation_percent > 0
            or self.supply_disruption_days > 0
            or bool(self.scenario_price_overrides)
        )
        if self.is_crisis_mode and self.current_day >= self.scenario_start_day and has_shock:
            # Deactivate if crisis_duration has elapsed (0 = indefinite → never deactivate)
            if self.scenario_end_day > 0 and self.current_day >= self.scenario_end_day:
                self.is_scenario_active = False   # prices revert, supply resumes
            else:
                self.is_scenario_active = True

        # Track which phase we are in for chart annotations
        if not self.is_crisis_mode or self.current_day < self.scenario_start_day:
            self._crisis_phase = "pre"
        elif self.is_scenario_active:
            self._crisis_phase = "active"
        else:
            # Crisis mode but scenario deactivated → recovery
            self._crisis_phase = "recovery" if (self.scenario_end_day > 0 and
                                                 self.current_day >= self.scenario_end_day) else "pre"

        # Reset daily panic signals and daily loss counters
        self.panic_signals      = 0
        self.daily_loss_reasons = {"Stockout": 0.0, "Price": 0.0}
        if not self.panic_dynamics_enabled:
            self.global_panic_level = 0.0

        # Household pantries evolve every calendar day, including days on which
        # a household does not visit the store.
        self._advance_household_consumption()

        # ---- Calculate today's visitor count ----
        month_factor = (
            self.SEASONALITY.get(self.current_month, 1.0)
            if self.traffic_variation_enabled else 1.0
        )
        day_factor = (
            self.WEEKDAY_WEIGHTS.get(self.current_weekday, 1.0)
            if self.traffic_variation_enabled else 1.0
        )
        noise = (
            self._day_rng.uniform(0.90, 1.10)
            if self.traffic_variation_enabled else 1.0
        )
        requested_target = max(1, int(
            self.base_consumers * month_factor * day_factor * noise
        ))
        target_count = min(requested_target, len(self.population_pool))
        self.requested_consumer_count = requested_target
        self.daily_consumer_count = target_count

        # ---- Sample profiles and create consumer agents ----
        todays_profiles = self._get_daily_profiles(target_count)
        daily_agents: list[ConsumerAgent] = []

        for k, profile in enumerate(todays_profiles):
            c_agent = ConsumerAgent(
                f"visit_{profile['_household_id']}_d{self.current_day}", self, profile
            )
            # Inject current global panic level
            c_agent.panic_level = self.global_panic_level
            self.schedule.add(c_agent)
            daily_agents.append(c_agent)

        # ---- Explicit daily phases -----------------------------------------
        # 1) Products reset counters, replenish shelves, update prices and age.
        # 2) The truck delivers arrived orders and places new ones.
        # 3) Only shopper order is randomised, representing shelf competition.
        for product in self.products:
            product.step()
        self.truck.step()
        self.random.shuffle(daily_agents)
        for consumer in daily_agents:
            consumer.step()
            self._schedule_next_visit(consumer.profile)

        # ---- Update global panic level ----
        if self.panic_dynamics_enabled and target_count > 0:
            panic_ratio = self.panic_signals / target_count
            # Up to 10% scarcity exposure is treated as normal retail friction.
            # Above that, panic grows continuously with both exposure and the
            # configured sensitivity. sensitivity=0 must imply zero contagion.
            scarcity_excess = max(0.0, panic_ratio - self.panic_exposure_floor)
            panic_growth = (
                self.panic_sensitivity * scarcity_excess * self.panic_growth_rate
            )
            # Recovery phase decays faster because normal prices/supply are visible.
            decay = (
                self.panic_decay_recovery
                if self._crisis_phase == "recovery"
                else self.panic_decay_active
            )
            self.global_panic_level = min(
                1.0, max(0.0, self.global_panic_level + panic_growth - decay)
            )
        else:
            panic_ratio = 0.0

        # ── Direct inflation → panic pathway ─────────────────────────────────────
        # Without this, low-disruption scenarios never trigger enough stockout signals,
        # so panic_sensitivity had no visible effect even with high inflation.
        # Multiplier=0.40 means: at 25% inflation, panic_sens needs to be ≥ ~0.5
        # for the signal (0.05+) to overcome daily panic decay (0.05/day).
        # At panic_sens=0 → signal=0 (no effect). At panic_sens=1, 25% inflation → +0.10/day.
        if (
            self.panic_dynamics_enabled
            and self.is_scenario_active
            and self.inflation_percent > 0
        ):
            price_shock_signal = (
                (self.inflation_percent / 100.0)
                * self.panic_sensitivity
                * self.inflation_panic_rate
            )
            self.global_panic_level = min(1.0, self.global_panic_level + price_shock_signal)

        # ── Media / Communication Channel (McCombs & Shaw 1972 agenda-setting) ─
        media_panic_effect = 0.0
        if self.panic_dynamics_enabled and self.media_intensity > 0:
            if self.communication_type == "panic":
                boost = self.media_intensity * 0.12
                self.global_panic_level = min(1.0, self.global_panic_level + boost)
                media_panic_effect = +round(boost, 4)
            elif self.communication_type == "calming":
                cut   = self.media_intensity * 0.10
                self.global_panic_level = max(0.0, self.global_panic_level - cut)
                media_panic_effect = -round(cut, 4)

        # ---- Collect consumer welfare metrics BEFORE removing agents ----
        n_consumers  = len(daily_agents)
        n_exhausted  = sum(1 for c in daily_agents if c.budget_exhausted)
        # Low-income access stress combines affordability and physical access.
        # Budget exhaustion alone is downward-biased during stockouts because
        # unavailable products prevent shoppers from spending their budget.
        n_stressed = sum(
            1 for c in daily_agents if is_low_income_access_stressed(c)
        )
        total_fat    = sum(c.total_fat_bought   for c in daily_agents)
        total_items  = sum(c.items_purchased    for c in daily_agents)
        total_base_demand = sum(c.items_base_wanted for c in daily_agents)
        total_requested_demand = sum(c.items_wanted for c in daily_agents)
        total_allowed_demand = sum(c.items_allowed for c in daily_agents)
        substitution_attempts = sum(
            c.substitution_attempts for c in daily_agents
        )
        substitution_candidates = sum(
            c.substitution_candidates_considered for c in daily_agents
        )
        substitution_price_rejections = sum(
            c.substitution_price_rejections for c in daily_agents
        )
        mean_fat     = (total_fat / total_items) if total_items > 0 else 0.0
        repeat_visitor_share = (
            sum(1 for c in daily_agents if c.visit_number > 1) / n_consumers
            if n_consumers > 0 else 0.0
        )
        pantry_units = sum(
            sum(max(0.0, float(qty)) for qty in p.get("_home_inv", {}).values())
            for p in self.population_pool
        )

        # ---- Income bracket welfare breakdown ----
        # Brackets: Low <€1500/mo, Mid €1500–3000, High >€3000
        BRACKET_THRESHOLDS = {"Low": (0, 1500), "Mid": (1500, 3000), "High": (3000, 1e9)}
        bracket_stats: dict[str, dict] = {}
        for bname, (lo, hi) in BRACKET_THRESHOLDS.items():
            grp = [c for c in daily_agents if lo <= c.income_midpoint < hi]
            n_g = len(grp)
            if n_g > 0:
                g_exh  = sum(1 for c in grp if c.budget_exhausted) / n_g
                g_items_want = sum(c.items_wanted    for c in grp)
                g_items_got  = sum(c.items_purchased for c in grp)
                g_fat_total  = sum(c.total_fat_bought for c in grp)
                g_ful  = g_items_got  / max(1, g_items_want)
                g_fat  = g_fat_total  / max(1, g_items_got)
            else:
                g_exh = g_ful = g_fat = 0.0
            bracket_stats[bname] = {
                "n": n_g,
                "budget_exh": round(g_exh, 4),
                "fulfillment": round(g_ful, 4),
                "mean_fat":    round(g_fat, 4),
            }

        # ── Gini coefficient of access inequality (Sen 1973) ───────────────────
        def _gini(vals: list) -> float:
            n = len(vals)
            if n <= 1:
                return 0.0
            s     = sorted(vals)
            numer = sum((2*(i+1) - n - 1) * v for i, v in enumerate(s))
            denom = n * sum(s)
            return round(numer / denom, 4) if denom > 0 else 0.0

        access_ratios = [c.items_purchased / max(1, c.items_wanted)
                         for c in daily_agents]
        gini_access   = _gini(access_ratios)

        # ── TPB: average subjective norm across consumers ────────────────────
        avg_tpb_norm = (sum(c.subjective_norm for c in daily_agents) / n_consumers
                        if n_consumers > 0 else 0.0)
        avg_tpb_int  = (sum(c._tpb_intention() for c in daily_agents) / n_consumers
                        if n_consumers > 0 else 0.0)

        # ── FIES per income bracket ──────────────────────────────────────────
        access_bracket: dict = {}
        for bname, (lo, hi) in BRACKET_THRESHOLDS.items():
            grp = [
                p for p in self.population_pool
                if lo <= float(p.get("income_midpoint", 2500.0)) < hi
            ]
            n_g = len(grp)
            demand_g = sum(float(p.get("_daily_consumption_demand", 0.0)) for p in grp)
            unmet_g = sum(float(p.get("_daily_consumption_unmet", 0.0)) for p in grp)
            access_bracket[bname] = {
                "n": n_g,
                "mean": round(
                    sum(int(p.get("_access_stress_score", 0)) for p in grp)
                    / max(1, n_g), 4
                ),
                "high": round(
                    sum(1 for p in grp if int(p.get("_access_stress_score", 0)) >= 3)
                    / max(1, n_g), 4
                ),
                "shortfall": round(unmet_g / demand_g, 4) if demand_g > 0 else 0.0,
            }

        # ── Stockpile demand pressure ────────────────────────────────────────
        base_items   = sum(sum(it.get("quantity", 1) for it in c.baseline_basket)
                          for c in daily_agents)
        actual_items = sum(c.items_wanted for c in daily_agents)
        stockpile_pressure = round(actual_items / max(1, base_items), 4)

        # ── Per-archetype behavioral snapshot (captured before agents are removed) ─
        # Stored on the model and read by collect_preference_snapshot() which is
        # called from app.py after model.step() returns.
        _by_arch_beh: dict[str, list] = {}
        for c in daily_agents:
            _by_arch_beh.setdefault(c.archetype, []).append(c)
        self._daily_arch_beh: dict[str, dict] = {}
        for _arch, _agents in _by_arch_beh.items():
            _n = max(1, len(_agents))
            self._daily_arch_beh[_arch] = {
                "BudgetExhaustionRate": sum(1 for a in _agents if a.budget_exhausted) / _n,
                "MeanFulfillment":      sum(a.items_purchased / max(1, a.items_wanted)
                                           for a in _agents) / _n,
                "MeanPanicLevel":       sum(a.panic_level for a in _agents) / _n,
                "MeanAccessStress":     sum(a.access_stress_score for a in _agents) / _n,
                "MeanFIES":             sum(a.access_stress_score for a in _agents) / _n,
                "MeanItemsUnmet":       sum(a.items_unmet for a in _agents) / _n,
            }

        # ---- Expose agent list for per-agent analytics (app.py reads this) ----
        # Objects remain alive in memory after removal; app.py collects snapshots
        # by calling _collect_agent_snapshot() immediately after model.step().
        self.last_daily_agents = daily_agents

        # ---- Remove daily consumer agents ----
        for c in daily_agents:
            self.schedule.remove(c)

        # ---- Record daily aggregates (products still in schedule) ----
        products = self.products

        d_rev        = sum(a.daily_base_revenue for a in products)   # constant-price (base_price × units)
        d_rev_nominal= sum(a.daily_revenue      for a in products)   # nominal (current_price × units)
        d_waste = sum(a.daily_waste      for a in products)
        d_lost  = sum(a.daily_lost_sales for a in products)
        d_sales = sum(a.daily_sales      for a in products)
        # Average price index across all products — useful for verifying inflation is applied
        d_avg_price = round(sum(a.current_price for a in products) / len(products), 4) if products else 0.0

        # Environmental metrics
        d_co2_sales = sum(a.daily_co2_sales  for a in products)
        d_co2_waste = sum(a.daily_co2_waste  for a in products)
        d_domestic  = sum(a.daily_domestic_sales for a in products)
        d_import    = sum(a.daily_import_sales   for a in products)
        d_organic   = sum(
            a.daily_sales for a in products if a.is_bio
        )
        category_sales: dict[str, float] = {}
        for product in products:
            category = str(product.category).strip() or "Unknown"
            category_sales[category] = (
                category_sales.get(category, 0.0) + product.daily_sales
            )
        import_dep  = (d_import / (d_domestic + d_import)) if (d_domestic + d_import) > 0 else 0.0

        # Consumer welfare metrics
        budget_exhaustion_rate = (n_exhausted / n_consumers) if n_consumers > 0 else 0.0
        food_stressed_pct      = (n_stressed  / n_consumers) if n_consumers > 0 else 0.0
        fulfillment_rate       = (
            sum(c.items_purchased for c in daily_agents) /
            max(1, sum(c.items_wanted for c in daily_agents))
        ) if daily_agents else 0.0  # daily_agents already removed but objects still live
        consumption_fulfillment_rate = (
            self.daily_household_consumption
            / self.daily_household_consumption_demand
            if self.daily_household_consumption_demand > 0 else 1.0
        )
        households_with_shortfall = sum(
            1 for p in self.population_pool
            if float(p.get("_daily_consumption_unmet", 0.0)) > 1e-12
        )
        cumulative_demand = sum(
            float(p.get("_cumulative_consumption_demand", 0.0))
            for p in self.population_pool
        )
        cumulative_unmet = sum(
            float(p.get("_cumulative_consumption_unmet", 0.0))
            for p in self.population_pool
        )

        self.daily_records.append({
            "Day":        self.current_day,
            "Revenue":        d_rev,         # constant-price revenue (base_price × units) — drops with inflation/disruption
            "NominalRevenue": d_rev_nominal, # nominal cash (current_price × units) — rises with inflation
            "AvgPrice":       d_avg_price,   # mean catalogue price — rises with inflation
            "Waste":      d_waste,
            "LostSales":  d_lost,
            "Sales":      d_sales,
            "Consumers":  target_count,
            "RequestedConsumers": requested_target,
            "EmpiricalSamplingUnits": self.empirical_sampling_units,
            "SimulatedHouseholdDraws": len(self.population_pool),
            "PopulationSamplingMethod": self.population_sampling_method,
            "BehaviorEvidenceMode": (
                "exploratory_extensions"
                if any((
                    self.panic_dynamics_enabled,
                    self.tpb_enabled,
                    self.prospect_theory_enabled,
                    self.preference_learning_enabled,
                    self.archetype_modifiers_enabled,
                    self.policy_choice_effects_enabled,
                ))
                else "empirical_only"
            ),
            "PanicDynamicsEnabled": int(self.panic_dynamics_enabled),
            "TPBEnabled": int(self.tpb_enabled),
            "ProspectTheoryEnabled": int(self.prospect_theory_enabled),
            "PreferenceLearningEnabled": int(self.preference_learning_enabled),
            "ArchetypeModifiersEnabled": int(self.archetype_modifiers_enabled),
            "PolicyChoiceEffectsEnabled": int(self.policy_choice_effects_enabled),
            "DCEAttributeRankingEnabled": int(
                self.dce_price_choice_supported
                or bool(self.substitution_ranking_categories)
            ),
            "DCEAttributeRankingCategories": (
                ",".join(sorted(
                    self.substitution_ranking_categories
                    | (self.dce_applicable_categories if self.dce_price_choice_supported else set())
                )) or "none"
            ),
            "ChoicePriceScaleIdentified": int(self.dce_price_choice_supported),
            "SubstitutionPriceGateEnabled": int(
                self.substitution_price_gate_supported
            ),
            "SubstitutionRankingMethod": self.substitution_ranking_method,
            "SubstitutionChoiceEvidenceEvents": (
                self.substitution_choice_evidence_events
            ),
            "SubstitutionAttempts": substitution_attempts,
            "SubstitutionCandidatesConsidered": substitution_candidates,
            "SubstitutionPriceRejections": substitution_price_rejections,
            "VisitorCapacityCapped": int(requested_target > target_count),
            "TrafficVariationEnabled": int(self.traffic_variation_enabled),
            "RepeatVisitorShare": round(repeat_visitor_share, 4),
            "ExpectedVisitIntervalDays": round(
                self.expected_household_visit_interval, 4
            ),
            "PanicLevel": self.global_panic_level,
            "ScarcityExposureRate": round(panic_ratio, 4),
            "CrisisPhase":    self._crisis_phase,        # "pre" | "active" | "recovery"
            "ScenarioEndDay": self.scenario_end_day,     # 0 if indefinite
            # Daily loss breakdown (stockout vs price-driven — reset each day)
            "DailyLossStockout": round(self.daily_loss_reasons.get("Stockout", 0.0), 2),
            "DailyLossPrice":    round(self.daily_loss_reasons.get("Price",    0.0), 2),
            # Store metadata
            "StoreTier":  self.store_tier,
            # Environmental
            "CO2Sales":   round(d_co2_sales, 2),
            "CO2Waste":   round(d_co2_waste, 2),
            "CO2Total":   round(d_co2_sales + d_co2_waste, 2),
            "ImportDepPct": round(import_dep * 100, 2),
            "DomesticSales": d_domestic,
            "ImportSales":   d_import,
            "OrganicSalesUnits": d_organic,
            "CategorySalesUnits": category_sales,
            # Consumer welfare — aggregate
            "BudgetExhaustionRate": round(budget_exhaustion_rate, 4),
            "FoodStressedPct":      round(food_stressed_pct,      4),
            "FulfillmentRate":      round(fulfillment_rate,        4),
            "BaseDemandUnits":      total_base_demand,
            "RequestedDemandUnits": total_requested_demand,
            "PolicyAllowedUnits":   total_allowed_demand,
            "UnmetDemandUnits":     sum(c.items_unmet for c in daily_agents),
            "HouseholdConsumptionDemand": round(
                self.daily_household_consumption_demand, 4
            ),
            "HouseholdConsumption": round(self.daily_household_consumption, 4),
            "HouseholdConsumptionUnmet": round(
                self.daily_household_consumption_unmet, 4
            ),
            "ConsumptionFulfillmentRate": round(consumption_fulfillment_rate, 4),
            "HouseholdsWithConsumptionShortfall": households_with_shortfall,
            "HouseholdConsumptionShortfallShare": round(
                households_with_shortfall / max(1, len(self.population_pool)), 4
            ),
            "CumulativeConsumptionShortfallRate": round(
                cumulative_unmet / cumulative_demand, 4
            ) if cumulative_demand > 0 else 0.0,
            "HouseholdPantryUnits": round(pantry_units, 4),
            "MeanFatPurchased":     round(mean_fat,                4),
            # Consumer welfare — by income bracket
            "BudgetExh_Low":    bracket_stats["Low"]["budget_exh"],
            "BudgetExh_Mid":    bracket_stats["Mid"]["budget_exh"],
            "BudgetExh_High":   bracket_stats["High"]["budget_exh"],
            "Fulfillment_Low":  bracket_stats["Low"]["fulfillment"],
            "Fulfillment_Mid":  bracket_stats["Mid"]["fulfillment"],
            "Fulfillment_High": bracket_stats["High"]["fulfillment"],
            "MeanFat_Low":      bracket_stats["Low"]["mean_fat"],
            "MeanFat_Mid":      bracket_stats["Mid"]["mean_fat"],
            "MeanFat_High":     bracket_stats["High"]["mean_fat"],
            "N_Low":            bracket_stats["Low"]["n"],
            "N_Mid":            bracket_stats["Mid"]["n"],
            "N_High":           bracket_stats["High"]["n"],
            # ── Nudge / Rationing ──────────────────────────────────────────
            "GiniAccess":         gini_access,
            "PurchaseLimitOn":    int(self.purchase_limit is not None),
            "PurchaseLimit":      self.purchase_limit if self.purchase_limit else 0,
            # ── Theory of Planned Behaviour ───────────────────────────────
            "AvgSubjectiveNorm":  round(avg_tpb_norm, 4),
            "AvgTPBIntention":    round(avg_tpb_int,  4),
            # ── FIES Food Security ─────────────────────────────────────────
            "AccessStress_Low":       access_bracket["Low"]["mean"],
            "AccessStress_Mid":       access_bracket["Mid"]["mean"],
            "AccessStress_High":      access_bracket["High"]["mean"],
            "AccessStressHigh_Low":   access_bracket["Low"]["high"],
            "AccessStressHigh_Mid":   access_bracket["Mid"]["high"],
            "AccessStressHigh_High":  access_bracket["High"]["high"],
            "ConsumptionShortfall_Low":  access_bracket["Low"]["shortfall"],
            "ConsumptionShortfall_Mid":  access_bracket["Mid"]["shortfall"],
            "ConsumptionShortfall_High": access_bracket["High"]["shortfall"],
            "AccessStressN_Low":      access_bracket["Low"]["n"],
            "AccessStressN_Mid":      access_bracket["Mid"]["n"],
            "AccessStressN_High":     access_bracket["High"]["n"],
            # Deprecated aliases retained for saved analyses and UI compatibility.
            "FIES_Low":           access_bracket["Low"]["mean"],
            "FIES_Mid":           access_bracket["Mid"]["mean"],
            "FIES_High":          access_bracket["High"]["mean"],
            "FIESSevere_Low":     access_bracket["Low"]["high"],
            "FIESSevere_Mid":     access_bracket["Mid"]["high"],
            "FIESSevere_High":    access_bracket["High"]["high"],
            # ── Stockpile demand pressure ──────────────────────────────────
            "StockpilePressure":  stockpile_pressure,
            # ── Media / Communication ──────────────────────────────────────
            "MediaIntensity":     self.media_intensity,
            "MediaType":          self.communication_type,
            "MediaPanicEffect":   media_panic_effect,
        })

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def get_product_daily_records(self) -> list[dict]:
        """
        Return per-product snapshot records collected during step().
        NOTE: This requires per-product logging to be enabled in app.py
        by calling collect_product_snapshot() after each step.
        """
        return getattr(self, "_product_snapshots", [])

    def collect_product_snapshot(self):
        """Call this once per day (from app.py) to log per-product state."""
        if not hasattr(self, "_product_snapshots"):
            self._product_snapshots = []
        for a in self.schedule.agents:
            if isinstance(a, ProductAgent):
                self._product_snapshots.append({
                    "Day":            self.current_day,
                    "ProductID":      a.prod_id,
                    "Product":        a.name,
                    "Category":       a.category,
                    "Shelf":          a.snap_shelf,
                    "Storage":        a.snap_storage,
                    "Pending":        a.snap_pending,
                    "Revenue":        a.daily_revenue,
                    "Sales":          a.daily_sales,
                    "Waste":          a.daily_waste,
                    "LostSales":      a.daily_lost_sales,
                    "Price":          a.current_price,
                    "NearExpirySold": a.daily_near_expiry_sold,
                    "CO2Sales":       round(a.daily_co2_sales,  2),
                    "CO2Waste":       round(a.daily_co2_waste,  2),
                    "DomesticSales":  a.daily_domestic_sales,
                    "ImportSales":    a.daily_import_sales,
                    "Origin":         a.origin,
                    "IsOrganic":      a.is_bio,
                    "FatContent":     a.fat_content,
                })

    def collect_preference_snapshot(self):
        """
        Log mean preference values per archetype per day.
        Captures behavioural learning drift for visualisation.
        Called from app.py alongside collect_product_snapshot().
        """
        if not hasattr(self, "_pref_snapshots"):
            self._pref_snapshots: list[dict] = []

        # Group population_pool profiles by archetype — these are the LIVE agents
        # in the model's pool (updated in-place by ConsumerAgent._update_preferences
        # since profiles are shared by reference).
        from collections import defaultdict
        by_arch: dict[str, list] = defaultdict(list)
        for p in self.population_pool:
            by_arch[p.get("archetype", "unknown")].append(p)

        # Behavioral stats were captured inside model.step() before agents were removed.
        arch_beh = getattr(self, "_daily_arch_beh", {})

        for arch, profiles in by_arch.items():
            n = len(profiles)
            if n == 0:
                continue
            beh = arch_beh.get(arch, {})
            mean_access_stress = (
                sum(int(p.get("_access_stress_score", 0)) for p in profiles) / n
            )
            self._pref_snapshots.append({
                "Day":                   self.current_day,
                "Archetype":             arch,
                "N":                     n,
                # Preference attributes (learned, persisted in profile dict)
                "MeanPriceSensitivity":  sum(p.get("price_sensitivity",  0.5) for p in profiles) / n,
                "MeanOrganicPref":       sum(p.get("organic_preference",  0.2) for p in profiles) / n,
                "MeanFinnishPref":       sum(p.get("finnish_preference",  0.5) for p in profiles) / n,
                "MeanPreferredFat":      sum(p.get("preferred_fat",       1.5) for p in profiles) / n,
                "MeanSubTolerance":      sum(p.get("sub_tolerance",       0.5) for p in profiles) / n,
                # Behavioral outcomes (pre-computed in step() before agent removal)
                "BudgetExhaustionRate":  beh.get("BudgetExhaustionRate", 0.0),
                "MeanFulfillment":       beh.get("MeanFulfillment",      0.0),
                "MeanPanicLevel":        beh.get("MeanPanicLevel",       0.0),
                "MeanAccessStress":      mean_access_stress,
                # Deprecated alias retained for existing charts/exports.
                "MeanFIES":              mean_access_stress,
                "MeanItemsUnmet":        beh.get("MeanItemsUnmet",       0.0),
            })
