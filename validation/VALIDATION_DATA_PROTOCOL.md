# GROCERYsim independent validation data protocol — draft

Status: **draft, not preregistered, and not evidence of validity**.

## Scope decision

GROCERYsim currently represents a synthetic Finnish store and a dairy-only catalogue at daily resolution. National annual all-grocery statistics cannot be compared directly with store-day dairy outcomes. They may constrain scenario inputs or establish scale plausibility, but they do not validate household choice, stockouts, replenishment, fulfilment, or waste mechanisms.

The minimum defensible external-validation package therefore needs two untouched sources:

1. An independent household transaction panel, preferably a non-overlapping LoCard cohort, for visit timing, dairy quantities, expenditure, product switching, heterogeneity, and price response.
2. An independent retailer operational extract from stores and dates never used in model development, containing POS sales, prices, on-hand inventory, deliveries/orders, waste, and store/SKU metadata.

## Freeze order

1. Declare the model version/commit and catalogue crosswalk.
2. Declare the target population, store formats, dates, SKUs, crisis definition, and exclusions.
3. Produce the transformation code and measurement-error analysis without examining validation outcomes.
4. Estimate acceptance intervals from the independent observational data or register the exact interval-estimation procedure while outcomes remain blinded.
5. Complete every source and interval field in `validation_plan_DRAFT.csv`.
6. Register the plan in a timestamped repository such as OSF. Replace `preregistered=false` only after registration and add the immutable reference.
7. Lock the ABM and random-seed schedule.
8. Run sufficient Monte Carlo replicates based on a convergence rule fixed in the registration. Never validate from the quick-preview run or an already-averaged trajectory.
9. Report every registered target, including missing, failed, and directionally wrong results. Do not tune the model after opening the external data. Any subsequent tuning creates a new model version and requires new untouched validation data.

## Required retailer fields

- anonymised store identifier and store format/area;
- timestamp/date and SKU identifier;
- units and revenue sold, realised unit price, discount/promotion flag;
- opening and closing on-hand inventory, receipts/deliveries, adjustments and returns;
- waste quantity, reason and date, preferably batch expiry;
- assortment/availability flag and stockout observation;
- product category, package size, origin, organic status, fat content and shelf life;
- disruption/intervention dates and documented operational changes.

POS sales alone cannot identify lost demand or basket fulfilment. Those outcomes require on-hand/availability observations and, ideally, substitution or customer-choice data.

## Acceptance rule

For each stochastic registered target, compute the declared statistic separately for every Monte Carlo run. The current conservative gate requires the replicate mean and its central 95% simulation interval to lie inside the registered empirical acceptance interval. At least three runs are required technically, but the preregistration must specify a larger convergence-based run count appropriate to the target.

Passing all targets supports validity only for the registered outcomes, population, period, store formats and scenarios. It does not establish causal validity of policy counterfactuals.

## Public sources and their limited roles

- LoCard: restricted candidate for independent household purchase validation, subject to data access, selection weighting and non-overlap confirmation.
- Statistics Finland CPI: observed exogenous price path, not behavioural validation.
- PTY/NielsenIQ annual statistics: market/store scale and all-grocery context; requires an explicit dairy/store scaling model before quantitative comparison.
- Luke food-waste reporting: national sector boundary only; not a direct store-dairy waste target.
- Luke food commodity balance: national availability rather than realised household consumption.
- Saarinen et al. COVID-19 sales index: useful crisis pattern candidate, but the published series covers all groceries and two-chain market aggregates; numerical data and compatible dairy scope are required.
