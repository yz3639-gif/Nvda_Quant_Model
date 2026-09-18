# Data, model and execution contracts

## Price inputs

A run consumes a unique chronological daily OHLCV CSV. Prices must be finite and positive, volume nonnegative, high/low must enclose open/close, and the research runner requires a complete XNYS session sequence over the supplied range. No missing price is fabricated. Corporate actions must be applied consistently to all OHLC fields; the old caches are used as stored, with adjustment lineage explicitly unverified. Splits/dividends cannot be inferred conclusively from a price envelope test.

The calendar is pinned to `exchange-calendars==4.11.2`, XNYS, including holidays, early closes and DST. Session labels and UTC instants are separate concepts. Daily prices are *assumed* available at the official close; a vendor's delivery latency is not independently proven.

The new controlled research uses only OHLCV-derived features. Legacy macro/fundamental caches without release/revision timestamps carry availability warnings and are excluded from the new comparison. QQQ/SMH legacy inputs may supply context; they are not substitutes for the NVDA buy-and-hold benchmark.

## Clocks and labels

| Field | Meaning |
|---|---|
| `published_at` / `timestamp` | Original publication timestamp, UTC normalized |
| `available_at` | Earliest documented availability, or an explicitly tagged assumption |
| `fit_at` | Timestamp when the fitted state could be constructed |
| `decision_at` | Close after which the research signal is available |
| `fill_at` | Next session open for the new daily strategy |
| `label_end_at` | Timestamp when the entire target outcome has matured |

The new direction target is open(t+2)/open(t+1)−1 for a decision at close(t). Training requires both availability and label maturity at or before the relevant fit time. Inner tuning, calibration and outer model fitting each use their own cutoff.

Event features use the most recent completed close at event availability. Event targets use the first close at or after availability as their baseline and subsequent 1/3/5-session closes as outcomes. This deliberately excludes already-realized prepublication returns and the immediate event move. It is a delayed-close forecast, not a causal estimate of the headline's impact. The longest auxiliary label controls training maturity. Timestamp/story clusters remain together across event folds.

Historical eligibility requires traceable availability provenance, not merely a populated timestamp. Publication-time fallback is explicitly an assumption. News cache fallback requires current fetched/generated/available timestamps within the TTL; expired data remain historical context and cannot change the active signal.

## Fitted state

`TrainingPreprocessor.fit` estimates clipping thresholds and medians on mature training rows. `transform` reuses those values without batch-dependent filling. Missing training columns have a fixed zero fallback. Features are structurally cleaned before splitting, but no full-sample statistical imputation is performed.

The registered research comparator called `rule` is a fixed volume/momentum family fitted within each fold. It is distinct from the legacy adaptive rule and the historical strict PrecisionRule. `main.py` also exposes explicit `rule`, `precision_rule`, and `ml_ensemble` identities so diagnostic ML fits cannot masquerade as the traded signal.

## Execution and account

New research explicitly selects `ExecutionConfig(mode='next_open', stop_reference='entry')`. Existing diagnostic calls default to corrected close execution with daily-reset brackets. The original flawed algorithm is available only through explicit `legacy_close` for historical comparison.

A signal is a target fraction of equity after fees. Each trade changes cash and shares; fees are proportional to absolute traded notional. `commission + slippage` is a single-side rate. `round_trip_cost` remains a documented legacy alias; it is not a round-trip rate. Slippage is represented as a monetary fee, not charged again through the execution price.

Gap stops/takes fill at the observed open. Old-inventory gap exits precede queued new orders and suppress same-session re-entry. Intrabar conflicts use the declared stop-first or take-first convention. Trailing stops use a prior-bar favorable watermark and current open, not an invented high/low sequence.

The drawdown policy checks open and close account equity against the closing high-water mark, liquidates with fees and permanently halts the run. It is not a continuous intrabar risk guarantee. An active strategy's flat equity after a halt is not evidence that risk disappeared. Long-only research does not model market impact, financing income, borrow availability, taxes, or liquidity limits.

Closed trade net P&L includes every episode fee. Open episodes are excluded from closed-trade win rate and profit factor, while their unrealized P&L remains in account equity. For every marked bar:

`initial equity + realized net P&L + unrealized P&L = cash + shares × mark price`.
