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
ConsumerAgent    – daily visitor.  Makes utility-based purchase decisions using
                   DCE-calibrated preference scores; switches between baseline
                   and crisis baskets when the scenario is active.

Model
-----
SupermarketModel – orchestrates the simulation.  Accepts the in-memory config
                   dict produced by data_processor.run_pipeline_from_data(),
                   handles daily consumer sampling (two modes: bootstrap-up when
                   real pool is smaller than target, or sample-down when it is
                   larger), and records per-day aggregate and per-product metrics.
"""

import json
import math
import random
from collections import defaultdict

import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation


# ---------------------------------------------------------------------------
# Archetype behavioural modifiers
# ---------------------------------------------------------------------------

# Learning rate: fraction of the gap closed per day toward a target preference value.
# Small enough that meaningful drift takes weeks, not hours.
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
        is_finnish: bool, is_organic: bool,
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
        self.is_plant_based = bool(product_data.get("is_plant_based", False))

        # --- Shelf (FIFO batches) ---
        initial_shelf         = int(product_data.get("initial_stock_shelf", 10))
        self.max_shelf_capacity = int(product_data.get("max_shelf_capacity", 20))
        self.shelf_batches    = [{"qty": initial_shelf, "age": 0}]

        # --- Storage (scalar) ---
        default_storage            = int(product_data.get("initial_stock_storage", 20))
        self.max_storage_capacity  = int(ai_capacity) if ai_capacity else 50
        self.stock_storage         = int(ai_capacity) if ai_capacity else default_storage

        # Shelf life
        self.max_shelf_life = int(product_data.get("shelf_life_days", 14))

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

        # Set during step(); initialised here so consumers can safely read it
        # even if RandomActivation runs a ConsumerAgent before this ProductAgent.
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
          2. Refill shelf from storage if below 30 %
          3. Take snapshots
          4. Apply inflation if scenario is active
          5. Age all shelf batches by one day
          6. Remove expired batches (waste) and apply near-expiry discount
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

        # 3. Snapshots
        self.snap_shelf   = self.stock_shelf
        self.snap_storage = self.stock_storage
        self.snap_pending = self.model.truck.get_pending_stock(self.name)

        # 4. Price update — crisis inflation first, then policy modifiers
        if self.model.is_scenario_active:
            inflated = round(
                self.base_price * (1.0 + self.model.inflation_percent / 100.0), 4
            )
        else:
            inflated = self.base_price

        # Apply fat tax / subsidy on top of (possibly inflated) base price
        policy: PolicyConfig = self.model.policy_config
        is_finnish = (self.origin == "Suomi")
        self.current_price = policy.apply_price_policy(
            inflated, self.fat_content, is_finnish, self.is_bio
        )

        # 5 & 6. Age batches, apply near-expiry discount, remove expired
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


# ---------------------------------------------------------------------------
# 2. Food Waste Log
# ---------------------------------------------------------------------------

class FoodWasteLog:
    """Accumulates food waste events for the entire simulation run."""

    def __init__(self):
        self.records: list[dict] = []

    def record(self, day: int, product: str, category: str, quantity: int, reason: str):
        self.records.append({
            "Day": day, "Product": product, "Category": category,
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

    def get_pending_stock(self, product_name: str) -> int:
        return sum(
            manifest.get(product_name, 0)
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
                for prod_name, qty in self.delivery_queue[d_date].items():
                    product = self.model.get_product_by_name(prod_name)
                    if product:
                        # Policy domestic supply shock — randomly block a fraction
                        # of Finnish-origin deliveries proportional to severity
                        if shock_active and product.origin == "Suomi":
                            blocked_frac = policy.domestic_shock_severity
                            qty = max(0, int(qty * (1.0 - blocked_frac)))
                            if qty == 0:
                                self.log.append({
                                    "Day": today, "Product": prod_name,
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
                                product  = prod_name,
                                category = product.category,
                                quantity = refused,
                                reason   = "Refused Delivery",
                            )
                        self.log.append({
                            "Day": today, "Product": prod_name,
                            "Action": "Delivery", "Quantity": accepted,
                            "Refused": refused,
                            "Note": "Storage Full" if refused > 0 else "OK",
                        })
                del self.delivery_queue[d_date]

        # --- Place new orders ---
        arrival_day = today + self.model.lead_time_days
        todays_order: dict[str, int] = {}

        for agent in self.model.schedule.agents:
            if not isinstance(agent, ProductAgent):
                continue

            # Check total supply pipeline: storage already on hand + stock in transit.
            # Old logic blocked ALL new orders if even 1 unit was pending — this caused
            # under-ordering during demand surges (e.g. hoarding events) because the
            # small pending delivery would become inadequate by the time it arrived.
            pending       = self.get_pending_stock(agent.name)
            total_supply  = agent.stock_storage + pending
            trigger       = agent.max_storage_capacity * self.model.reorder_point

            if total_supply >= trigger:
                continue   # enough supply in pipeline, no order needed

            # Order enough to fill storage to target level, net of what's already coming
            target_qty = int(agent.max_storage_capacity * self.model.target_stock_level)
            order_qty  = max(0, target_qty - total_supply)
            if order_qty > 0:
                todays_order[agent.name] = order_qty
                self.log.append({
                    "Day": today, "Product": agent.name,
                    "Action": "Order", "Quantity": order_qty,
                    "Explanation": (
                        f"TotalSupply {total_supply} (storage {agent.stock_storage} "
                        f"+ pending {pending}) < trigger {int(trigger)}"
                    ),
                })

        if todays_order:
            if arrival_day not in self.delivery_queue:
                self.delivery_queue[arrival_day] = {}
            for name, qty in todays_order.items():
                self.delivery_queue[arrival_day][name] = (
                    self.delivery_queue[arrival_day].get(name, 0) + qty
                )


# ---------------------------------------------------------------------------
# 4. Consumer Agent
# ---------------------------------------------------------------------------

class ConsumerAgent(Agent):
    """
    Represents a single shopping visit.

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

    Utility function (higher = more desirable)
    ------------------------------------------
      U = origin_bonus + organic_bonus + fat_match - price_disutility

    where each component is scaled to keep U in roughly [−1, 2] at normal prices,
    and the purchase threshold is calibrated from the agent's reference price so
    that agents will reliably buy their preferred products at baseline prices.
    """

    def __init__(self, unique_id, model, profile: dict):
        super().__init__(unique_id, model)
        self.profile = profile

        # Preference scores (0–1 each)
        self.price_sensitivity  = float(profile.get("price_sensitivity",  0.5))
        self.finnish_preference = float(profile.get("finnish_preference",  0.5))
        self.organic_preference = float(profile.get("organic_preference",  0.2))
        self.preferred_fat      = float(profile.get("preferred_fat",       1.5))
        self.reference_price    = float(profile.get("reference_price",     1.5))
        self.substitution_rate  = float(profile.get("substitution_rate",   0.5))
        self.archetype          = profile.get("archetype", "habitual_buyer")

        # Archetype-specific modifiers
        mods = ARCHETYPE_MODIFIERS.get(self.archetype, (0.5, 1.0, 0.0))
        # If behavioural learning has already updated sub_tolerance in the profile,
        # use that value; otherwise fall back to the archetype default.
        self.sub_tolerance        = float(profile.get("sub_tolerance", mods[0]))
        self.hoarding_multiplier  = mods[1]   # hoarding boost during panic
        self.price_tolerance_extra = mods[2]  # bonus tolerance on top of base

        # Baskets
        self.baseline_basket = profile.get("baseline_basket", [])
        self.crisis_basket   = profile.get("crisis_basket",   self.baseline_basket)
        self.budget          = float(profile.get("budget",        50.0))
        self.crisis_budget   = float(profile.get("crisis_budget", self.budget))

        # Income proxy — used for affordability / food-stress calculations.
        # Stored as the midpoint of the reported income bracket (€ / month).
        self.income_midpoint = float(profile.get("income_midpoint", 2500.0))

        # Panic state (updated by model)
        self.panic_level = 0.0

        # Purchase utility threshold — calibrated so agents buy their preferred
        # products at baseline prices (U ≥ threshold).
        self.utility_threshold = 0.30 + self.price_sensitivity * 0.25 - self.price_tolerance_extra

        # ---- Policy / welfare tracking (reset each step) ----
        self.items_wanted     = 0      # items in active basket
        self.items_purchased  = 0      # items actually bought
        self.budget_exhausted = False  # ran out of budget before finishing basket
        self.total_fat_bought = 0.0    # sum(fat_content × qty) for nutrition scoring

        # ---- Behavioural learning state — persisted in profile dict ----
        # Without this, the organic-streak and fat-history reset every day
        # because agents are recreated, eliminating streak-based learning boosts.
        self._organic_streak: int         = profile.setdefault("_organic_streak", 0)
        self._fat_history:    list[float] = profile.setdefault("_fat_history", [])

        # ── Prospect Theory (Kahneman & Tversky 1979) ──────────────────────────
        # Loss aversion λ=2.25 and curvature α=0.88 from Tversky & Kahneman (1992)
        self.loss_aversion   = float(profile.get("loss_aversion", 2.25))
        self.kt_alpha        = 0.88
        # Per-product reference prices seeded from base (pre-crisis) prices.
        # Because agents are recreated every day the dict must be pre-populated;
        # otherwise every consumer falls back to a single scalar reference_price
        # (mean of their whole basket) which breaks per-product Prospect Theory.
        # Using base_price (not current_price) ensures the reference is always the
        # pre-inflation price, so every crisis day feels correctly expensive.
        self._ref_prices: dict = {
            name: pa.base_price
            for name, pa in model.product_map.items()
        }

        # ── Theory of Planned Behaviour (Ajzen 1991) ────────────────────────────
        # Weights from Armitage & Conner (2001) meta-analysis
        self.attitude         = max(0.3, 1.0 - self.price_sensitivity * 0.4)
        self.subjective_norm  = 0.0   # updated each step from store-crowding signal
        self.pbc              = 1.0   # perceived behavioural control
        self._tpb_att_w       = 0.49
        self._tpb_norm_w      = 0.26
        self._tpb_pbc_w       = 0.39  # Armitage & Conner (2001) meta-analytic weights

        # ── Temporal Discounting / Stockpiling (O'Donoghue & Rabin 1999) ───────
        # β-δ quasi-hyperbolic discounting: present-biased agents stockpile more
        # when panic rises (Hendel & Nevo 2006 pantry model)
        self.beta           = max(0.5, 0.90 - self.price_sensitivity * 0.15)

        # Home inventory persists across days via the profile dict.
        # ConsumerAgents are recreated every day, so any instance variable
        # is lost between days — home inventory was always starting at zero,
        # meaning every consumer always believed they needed a full stockpile.
        # Fix: store _home_inv inside the profile dict; since profile is a
        # reference into population_pool (not a copy), changes persist.
        self._home_inv: dict = profile.setdefault("_home_inv", {})

        # Allow model-level override of stockpile_days (from sidebar slider)
        if getattr(model, "stockpile_days_override", None) is not None:
            base_days = float(model.stockpile_days_override)
        else:
            base_days = float(profile.get("stockpile_days", 3.0))
        self.stockpile_days  = base_days   # rises with panic during step()

        # ── FIES (FAO Food Insecurity Experience Scale, simplified 4-item) ─────
        self.food_insecurity_score = 0   # 0=secure 1=mild 2–3=moderate 4=severe
        self.items_unmet           = 0   # items wanted but not obtained

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
        Uses Armitage & Conner (2001) meta-analytic weights:
          I = 0.49·A + 0.26·SN + 0.39·PBC
        Returns a value in [0, 1] representing purchase motivation strength.
        """
        raw = (self._tpb_att_w  * self.attitude
             + self._tpb_norm_w * self.subjective_norm
             + self._tpb_pbc_w  * self.pbc)
        return min(1.0, max(0.0, raw))

    # ------------------------------------------------------------------
    # Utility computation
    # ------------------------------------------------------------------

    def _compute_utility(self, product: ProductAgent) -> float:
        """
        Calculate utility for purchasing `product`.
        Scale: positive = acceptable, negative = unacceptable.

        If the labelling policy is active, organic_preference and
        health (fat-match) weights receive a small boost per PolicyConfig.
        """
        policy: PolicyConfig = self.model.policy_config
        today = self.model.current_day

        # ── Prospect Theory price evaluation (Kahneman & Tversky 1979) ─────────
        effective_p = product.current_price
        if product._has_near_expiry:
            effective_p *= 0.5

        # Reference price: last price seen for this product, or personal ref price
        ref_p       = self._ref_prices.get(product.name, self.reference_price)
        # Normalised deviation: positive = price fell (gain), negative = rose (loss)
        price_delta = (ref_p - effective_p) / max(ref_p, 0.01)
        # KT value: amplifies losses relative to equivalent gains
        kt_val      = self._kt_value(price_delta, self.loss_aversion, self.kt_alpha)
        # Map to disutility: at reference price (delta=0, kt=0) → sensitivity * 1.0
        #   price fall of 20% → kt≈+0.22 → disutility *0.84 (clear relief)
        #   price rise of 20% → kt≈−0.52 → disutility *1.47 (clearly amplified pain)
        # Multiplier raised from 0.3 → 0.6 so inflation differences are clearly visible
        price_disutility = self.price_sensitivity * max(0.01, 1.0 - kt_val * 0.6)

        # Origin preference bonus (max 0.4)
        is_finnish   = 1.0 if product.origin == "Suomi" else 0.0
        origin_bonus = self.finnish_preference * is_finnish * 0.40

        # Organic preference bonus (max 0.35) — boosted by labelling policy
        organic_pref = self.organic_preference
        if policy.is_labelling_active(today):
            organic_pref = min(1.0, organic_pref + policy.labelling_organic_boost)
        is_organic    = 1.0 if product.is_bio else 0.0
        organic_bonus = organic_pref * is_organic * 0.35

        # Fat content match — Gaussian similarity (max 0.25)
        # Health-labelling slightly increases sensitivity to fat content
        fat_weight = 0.25
        if policy.is_labelling_active(today):
            fat_weight = min(0.40, fat_weight + policy.labelling_health_boost * 0.25)
        fat_diff  = abs(product.fat_content - self.preferred_fat)
        fat_match = math.exp(-fat_diff / 2.0) * fat_weight

        utility = origin_bonus + organic_bonus + fat_match - price_disutility
        return utility

    # ------------------------------------------------------------------
    # Substitution search
    # ------------------------------------------------------------------

    def _find_best_substitute(
        self,
        category: str,
        wanted_qty: int,
        exclude_name: str,
    ) -> ProductAgent | None:
        """
        Return the highest-utility in-stock product in the same category,
        or None if none is acceptable.
        """
        # Respect archetype substitution tolerance
        if self.model._day_rng.random() > self.sub_tolerance:
            return None   # agent refuses to substitute

        candidates = [
            a for a in self.model.schedule.agents
            if isinstance(a, ProductAgent)
            and a.category == category
            and a.name != exclude_name
            and a.stock_shelf >= wanted_qty
        ]
        if not candidates:
            return None

        ranked = sorted(candidates, key=self._compute_utility, reverse=True)
        best   = ranked[0]

        if self._compute_utility(best) < self.utility_threshold:
            return None   # best substitute still unacceptable
        return best

    # ------------------------------------------------------------------
    # Purchase execution (FIFO with near-expiry discount)
    # ------------------------------------------------------------------

    def _execute_purchase(
        self,
        product: ProductAgent,
        wanted_qty: int,
        remaining_budget: float,
        is_substitute: bool = False,
    ) -> float:
        """
        Deduct stock from shelf batches (oldest first) and record revenue.
        Returns the actual money spent.
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

            # Check budget
            if cost_paid + unit_price > remaining_budget:
                break

            take = min(qty_left, batch["qty"])
            batch["qty"] -= take
            qty_left     -= take
            cost_paid    += take * unit_price

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

            # CO2 attribution
            is_finnish = (product.origin == "Suomi")
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
            self._home_inv[product.name] = (
                self._home_inv.get(product.name, 0) + qty_purchased
            )
            # Update reference price slowly (EMA) so loss aversion persists.
            # _ref_prices is pre-seeded with base_price in __init__, so old_ref is
            # always the pre-crisis price for a fresh agent — this is the correct
            # Prospect Theory anchor.  The EMA here is within-trip only (agents are
            # ephemeral) but the seed ensures the right reference is always present.
            old_ref = self._ref_prices.get(product.name, product.base_price)
            self._ref_prices[product.name] = round(0.85 * old_ref + 0.15 * product.current_price, 4)

        return cost_paid

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
        # the learned values across days (agents are recreated each day from pool).
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
        self.items_wanted     = 0
        self.items_purchased  = 0
        self.budget_exhausted = False
        self.total_fat_bought = 0.0

        # Determine active basket and budget
        if self.model.is_scenario_active:
            active_basket  = self.crisis_basket
            active_budget  = self.crisis_budget
        else:
            active_basket  = self.baseline_basket
            active_budget  = self.budget

        # ── Panic propagation (existing) ────────────────────────────────────────
        if self.model.global_panic_level > 0.5:
            self.panic_level = min(1.0, self.model.global_panic_level + self.model._day_rng.uniform(0, 0.2))

        # ── Theory of Planned Behaviour: update norm + PBC (Ajzen 1991) ─────────
        # Subjective norm rises with store crowding and observed panic
        crowd_ratio          = (self.model.daily_consumer_count
                                 / max(1, self.model.base_consumers))
        self.subjective_norm = min(1.0, crowd_ratio * 0.40 + self.panic_level * 0.60)
        # Perceived behavioural control drops when income is low or panic is high
        income_factor        = min(1.0, self.income_midpoint / 3000.0)
        self.pbc             = max(0.10, income_factor - self.panic_level * 0.35)
        # TPB intention modulates utility threshold: higher intention → more willing
        intention            = self._tpb_intention()
        self.utility_threshold = max(0.05,
            (0.30 + self.price_sensitivity * 0.25 - self.price_tolerance_extra)
            - (intention - 0.50) * 0.12
        )

        # ── Temporal discounting: stockpile target rises with panic ─────────────
        # Multiplier reduced from 7.0 → 3.0 so max stockpile_days = 5 (not 10).
        # panic*7 was unrealistic: full-panic agents wanted 10 days of every item,
        # exhausting shelves in one visit and creating revenue spikes far above baseline.
        self.stockpile_days = max(1.0,
            self.profile.get("stockpile_days", 2.0) + self.panic_level * 3.0
        )
        # Deplete home inventory (daily consumption between store visits)
        for pname in list(self._home_inv.keys()):
            self._home_inv[pname] = max(0, self._home_inv[pname] - 1)

        spent = 0.0
        self.items_wanted = sum(item.get("quantity", 1) for item in active_basket)

        for item in active_basket:
            if spent >= active_budget:
                self.budget_exhausted = True
                break

            wanted_name = item["product_name"]
            wanted_qty  = item["quantity"]
            category    = item.get("category", "")

            # ── Temporal discounting: stockpile demand ──────────────────────────
            # Agent targets β-discounted home stock = stockpile_days * daily_need
            home_have = self._home_inv.get(wanted_name, 0)
            # proxy: basket quantity ≈ daily consumption for this item
            daily_need       = max(1, item.get("quantity", 1))
            stockpile_target = int(math.ceil(daily_need * self.stockpile_days * self.beta))
            stockpile_gap    = max(0, stockpile_target - home_have)
            if stockpile_gap > wanted_qty:
                wanted_qty = stockpile_gap   # stockpile drive overrides base basket

            # Panic hoarding: scale current demand by hoarding_factor (the sidebar slider).
            # Previously hoarding was multiplied on top of stockpile demand, producing
            # e.g. 48 units of milk per consumer (18 stockpile × 1.4 × 1.9 slider),
            # emptying shelves in one visit and creating nominal-revenue spikes above
            # baseline.  Fix: apply hoarding_factor as a proportional scalar AFTER the
            # stockpile calculation, without the archetype multiplier (which is already
            # expressed through utility_threshold / price_tolerance_extra differences).
            # hoarding_factor = 1.0 → no change; 2.0 → double demand; 3.0 → triple.
            if self.panic_level > 0.4:
                wanted_qty = math.ceil(wanted_qty * self.model.hoarding_factor)

            product = self.model.get_product_by_name(wanted_name)

            # --- Product not in catalogue (shouldn't happen after validation) ---
            if not product:
                substitute = self._find_best_substitute(category, 1, "")
                if substitute:
                    cost = self._execute_purchase(substitute, 1, active_budget - spent, is_substitute=True)
                    spent += cost
                continue

            # --- Utility check ---
            utility = self._compute_utility(product)
            if utility < self.utility_threshold:
                product.daily_lost_sales += wanted_qty
                self.model.track_loss("Price", wanted_qty * product.current_price)
                # Try substitute
                sub = self._find_best_substitute(category, wanted_qty, wanted_name)
                if sub:
                    cost = self._execute_purchase(sub, wanted_qty, active_budget - spent, is_substitute=True)
                    spent += cost
                continue

            # --- Stock check ---
            if product.stock_shelf >= wanted_qty:
                cost = self._execute_purchase(product, wanted_qty, active_budget - spent)
                spent += cost
                # Near-empty shelf triggers panic signal
                if product.stock_shelf < 3:
                    self.model.add_panic_signal()
            else:
                product.daily_lost_sales += wanted_qty
                self.model.track_loss("Stockout", wanted_qty * product.current_price)
                sub = self._find_best_substitute(category, wanted_qty, wanted_name)
                if sub:
                    cost = self._execute_purchase(sub, wanted_qty, active_budget - spent, is_substitute=True)
                    spent += cost

        # ── FIES (FAO Food Insecurity Experience Scale, 4-item simplified) ──────
        fulfillment        = self.items_purchased / max(1, self.items_wanted)
        self.items_unmet   = max(0, self.items_wanted - self.items_purchased)
        fies = 0
        if self.panic_level > 0.5:                           # Q1 worried about food
            fies += 1
        if fulfillment < 0.70:                               # Q2 couldn't eat variety
            fies += 1
        if self.budget_exhausted and self.items_unmet > 0:   # Q3 ran out of food
            fies += 1
        if fulfillment < 0.30:                               # Q4 severe deprivation
            fies += 1
        self.food_insecurity_score = fies

        # ---- Behavioural learning ----
        bought_organic   = any(
            isinstance(a, ProductAgent) and a.is_bio and a.daily_sales > 0
            for a in self.model.schedule.agents
        )
        mean_fat_today = (
            self.total_fat_bought / max(1, self.items_purchased)
        ) if self.items_purchased > 0 else 0.0
        self._update_preferences(bought_organic, mean_fat_today)


# ---------------------------------------------------------------------------
# 5. Supermarket Model
# ---------------------------------------------------------------------------

class SupermarketModel(Model):
    """
    Main ABM.  Accepts either a pre-built config dict (from data_processor)
    or a path to a mesa_config.json file.

    Consumer sampling logic
    -----------------------
    Each day a target_count of consumers is calculated (base × seasonality × weekday
    × random ±10 %).

    • If pool_size >= target_count : sample without replacement for maximum variety.
    • If pool_size <  target_count : use all real profiles + fill remaining from
      synthetic pool (random.choices with replacement).

    This means the model automatically handles both the "20 real / 100 agents"
    and the "200 real / 50 agents" scenarios described in the project brief.
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

        Returns a dict  product_name → {max_shelf_capacity, max_storage_capacity,
                                         initial_stock_shelf, initial_stock_storage}
        """
        # ── 1. Average basket quantity per product across population ─────────
        basket_totals: dict[str, float] = {}
        for profile in population:
            for item in profile.get("baseline_basket", []):
                name = item.get("product_name", "")
                qty  = float(item.get("quantity", 1))
                basket_totals[name] = basket_totals.get(name, 0.0) + qty

        n_pop = max(1, len(population))
        avg_qty: dict[str, float] = {n: v / n_pop for n, v in basket_totals.items()}

        # ── 2. Per-product calibration ───────────────────────────────────────
        result: dict[str, dict] = {}
        for prod in products:
            name       = prod.get("name", "")
            shelf_life = int(prod.get("shelf_life_days", 7))

            # How many days of stock to keep on the shelf
            if shelf_life <= 7:
                shelf_cover = 1.5    # perishable — rotate quickly
            elif shelf_life <= 30:
                shelf_cover = 2.5    # medium (yogurt, cheese, …)
            else:
                shelf_cover = 4.0    # dry / canned goods

            # Expected daily demand for this product
            daily_demand = max(1.0, base_consumers * avg_qty.get(name, 0.5))

            max_shelf   = max(10, int(math.ceil(daily_demand * shelf_cover)))
            storage_days = lead_time + 4   # lead-time + safety buffer
            max_storage = max(max_shelf * 2, int(math.ceil(daily_demand * storage_days)))

            result[name] = {
                "max_shelf_capacity":   max_shelf,
                "max_storage_capacity": max_storage,
                "initial_stock_shelf":  int(max_shelf   * 0.75),
                "initial_stock_storage": int(max_storage * 0.60),
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
    ):
        super().__init__()
        # Seed Mesa's own internal RNG (used by RandomActivation.step() to
        # shuffle agent execution order).  Without this, each model instance
        # gets a different random state from the OS, so two models running with
        # the same fixed_seed still diverge because agents visit shelves in a
        # different order and deplete stock differently.
        self.random.seed(fixed_seed)
        self.schedule = RandomActivation(self)

        # Seeded RNG for all explicit random calls within the model
        self.fixed_seed  = fixed_seed
        self._day_rng    = random.Random(fixed_seed)

        # General parameters
        self.base_consumers    = base_consumers
        self.current_month     = start_month
        self.current_weekday   = 0
        self.current_day       = 0

        # Logistics parameters
        self.reorder_point      = reorder_pt
        self.target_stock_level = target_stock
        self.lead_time_days     = lead_time

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

        # Runtime state
        self.is_scenario_active     = False
        self._crisis_phase          = "pre"   # "pre" | "active" | "recovery"
        self.loss_reasons           = {"Stockout": 0.0, "Price": 0.0}  # cumulative totals
        self.daily_loss_reasons     = {"Stockout": 0.0, "Price": 0.0}  # reset each day
        self.global_panic_level     = 0.0
        self.panic_signals          = 0
        self.total_churned_agents   = 0
        self.daily_consumer_count   = 0

        # Food waste accumulator
        self.food_waste_log = FoodWasteLog()

        # Policy configuration (always present; default = all policies off)
        self.policy_config = PolicyConfig(policy_cfg)

        # Lookup dict: product_name → ProductAgent
        self.product_map: dict[str, ProductAgent] = {}

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
            cal = _calib.get(p_data["name"], {})
            # Only override if no explicit ai_recs for this product
            ai_cap = ai_recs.get(p_data["name"]) if ai_recs else None
            if not ai_cap:
                for field in ("max_shelf_capacity", "max_storage_capacity",
                              "initial_stock_shelf", "initial_stock_storage"):
                    if field in cal:
                        p_data_cal[field] = cal[field]
            agent  = ProductAgent(f"prod_{i}", self, p_data_cal, ai_capacity=ai_cap)
            self.schedule.add(agent)
            self.product_map[p_data["name"]] = agent

        # Add truck
        self.truck = SupplyTruck("truck_1", self)
        self.schedule.add(self.truck)

        # Population pool — deep-copied so that behavioural learning (which
        # writes updated preferences back into each profile dict) cannot bleed
        # across model instances sharing the same config_data object.
        # Without this copy, a Baseline and Crisis model running in the same
        # process would contaminate each other's agent profiles from Day 1,
        # causing divergence even when all crisis parameters are zero.
        import copy
        self.population_pool: list[dict] = copy.deepcopy(config.get("population", []))
        if not self.population_pool:
            raise ValueError("Population pool is empty — run data_processor first.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_product_by_name(self, name: str) -> ProductAgent | None:
        return self.product_map.get(name)

    def add_panic_signal(self):
        self.panic_signals += 1

    def track_loss(self, reason: str, amount: float):
        if reason in self.loss_reasons:
            self.loss_reasons[reason]       += amount   # cumulative (never reset)
            self.daily_loss_reasons[reason] += amount   # reset each day

    def _get_daily_profiles(self, target_count: int) -> list[dict]:
        """
        Sample `target_count` consumer profiles from the pool.

        If pool size >= target_count : sample WITHOUT replacement (maximum variety).
        If pool size <  target_count : resample WITH replacement (bootstrap mode).
        """
        pool_size = len(self.population_pool)
        if pool_size >= target_count:
            return self._day_rng.sample(self.population_pool, target_count)
        else:
            return self._day_rng.choices(self.population_pool, k=target_count)

    # ------------------------------------------------------------------
    # Step (one simulation day)
    # ------------------------------------------------------------------

    def step(self):
        self.current_day += 1

        # Advance calendar — 7-day week (0 Mon … 6 Sun)
        self.current_weekday = (self.current_day - 1) % 7
        # Advance month roughly every 30 days
        month_idx = ((self.current_day - 1) // 30) % 12
        self.current_month = (month_idx + 1)   # 1–12

        # ── Crisis phase management ──────────────────────────────────────────────
        # Phase 1 — "pre":      before scenario_start_day
        # Phase 2 — "active":   scenario_start_day … scenario_end_day (or end of sim)
        # Phase 3 — "recovery": scenario_end_day … end of sim  (prices normalised,
        #                        supply restored; panic decays naturally)
        has_shock = (self.inflation_percent > 0 or self.supply_disruption_days > 0)
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

        # ---- Calculate today's visitor count ----
        month_factor  = self.SEASONALITY.get(self.current_month, 1.0)
        day_factor    = self.WEEKDAY_WEIGHTS.get(self.current_weekday, 1.0)
        noise         = self._day_rng.uniform(0.90, 1.10)
        target_count  = max(1, int(self.base_consumers * month_factor * day_factor * noise))
        self.daily_consumer_count = target_count

        # ---- Sample profiles and create consumer agents ----
        todays_profiles = self._get_daily_profiles(target_count)
        daily_agents: list[ConsumerAgent] = []

        for k, profile in enumerate(todays_profiles):
            c_agent = ConsumerAgent(
                f"cust_d{self.current_day}_{k}", self, profile
            )
            # Inject current global panic level
            c_agent.panic_level = self.global_panic_level
            self.schedule.add(c_agent)
            daily_agents.append(c_agent)

        # ---- Run all agents (ProductAgents + Truck + Consumers) ----
        self.schedule.step()

        # ---- Update global panic level ----
        if target_count > 0:
            panic_ratio = self.panic_signals / target_count
            threshold   = (1.0 - self.panic_sensitivity) * 0.15
            if panic_ratio > threshold:
                self.global_panic_level = min(1.0, self.global_panic_level + 0.15)
            else:
                # Recovery phase: faster decay — consumers can SEE prices are normal again.
                # Active/pre: slow decay (0.05/day); Recovery: faster (0.10/day).
                decay = 0.10 if self._crisis_phase == "recovery" else 0.05
                self.global_panic_level = max(0.0, self.global_panic_level - decay)

        # ── Direct inflation → panic pathway ─────────────────────────────────────
        # Without this, low-disruption scenarios never trigger enough stockout signals,
        # so panic_sensitivity had no visible effect even with high inflation.
        # Multiplier=0.40 means: at 25% inflation, panic_sens needs to be ≥ ~0.5
        # for the signal (0.05+) to overcome daily panic decay (0.05/day).
        # At panic_sens=0 → signal=0 (no effect). At panic_sens=1, 25% inflation → +0.10/day.
        if self.is_scenario_active and self.inflation_percent > 0:
            price_shock_signal = (self.inflation_percent / 100.0) * self.panic_sensitivity * 0.40
            self.global_panic_level = min(1.0, self.global_panic_level + price_shock_signal)

        # ── Media / Communication Channel (McCombs & Shaw 1972 agenda-setting) ─
        media_panic_effect = 0.0
        if self.media_intensity > 0:
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
        # Food-stressed = budget exhausted AND income below median proxy (€2000/mo)
        n_stressed   = sum(
            1 for c in daily_agents
            if c.budget_exhausted and c.income_midpoint < 2000.0
        )
        total_fat    = sum(c.total_fat_bought   for c in daily_agents)
        total_items  = sum(c.items_purchased    for c in daily_agents)
        mean_fat     = (total_fat / total_items) if total_items > 0 else 0.0

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
        FIES_THRESHOLDS = {"Low": (0, 1500), "Mid": (1500, 3000), "High": (3000, 1e9)}
        fies_bracket: dict = {}
        for bname, (lo, hi) in FIES_THRESHOLDS.items():
            grp   = [c for c in daily_agents if lo <= c.income_midpoint < hi]
            n_g   = len(grp)
            fies_bracket[bname] = {
                "mean":   round(sum(c.food_insecurity_score for c in grp) / max(1, n_g), 4),
                "severe": round(sum(1 for c in grp if c.food_insecurity_score >= 3)
                                / max(1, n_g), 4),
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
                "MeanFIES":             sum(a.food_insecurity_score for a in _agents) / _n,
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
        products = [a for a in self.schedule.agents if isinstance(a, ProductAgent)]

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
        import_dep  = (d_import / (d_domestic + d_import)) if (d_domestic + d_import) > 0 else 0.0

        # Consumer welfare metrics
        budget_exhaustion_rate = (n_exhausted / n_consumers) if n_consumers > 0 else 0.0
        food_stressed_pct      = (n_stressed  / n_consumers) if n_consumers > 0 else 0.0
        fulfillment_rate       = (
            sum(c.items_purchased for c in daily_agents) /
            max(1, sum(c.items_wanted for c in daily_agents))
        ) if daily_agents else 0.0  # daily_agents already removed but objects still live

        self.daily_records.append({
            "Day":        self.current_day,
            "Revenue":        d_rev,         # constant-price revenue (base_price × units) — drops with inflation/disruption
            "NominalRevenue": d_rev_nominal, # nominal cash (current_price × units) — rises with inflation
            "AvgPrice":       d_avg_price,   # mean catalogue price — rises with inflation
            "Waste":      d_waste,
            "LostSales":  d_lost,
            "Sales":      d_sales,
            "Consumers":  target_count,
            "PanicLevel": self.global_panic_level,
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
            # Consumer welfare — aggregate
            "BudgetExhaustionRate": round(budget_exhaustion_rate, 4),
            "FoodStressedPct":      round(food_stressed_pct,      4),
            "FulfillmentRate":      round(fulfillment_rate,        4),
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
            "FIES_Low":           fies_bracket["Low"]["mean"],
            "FIES_Mid":           fies_bracket["Mid"]["mean"],
            "FIES_High":          fies_bracket["High"]["mean"],
            "FIESSevere_Low":     fies_bracket["Low"]["severe"],
            "FIESSevere_Mid":     fies_bracket["Mid"]["severe"],
            "FIESSevere_High":    fies_bracket["High"]["severe"],
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
                "MeanFIES":              beh.get("MeanFIES",             0.0),
                "MeanItemsUnmet":        beh.get("MeanItemsUnmet",       0.0),
            })
