# NVDA Multi-Factor Quant Model

This package implements a walk-forward, multi-factor NVDA strategy research workflow:

- yfinance OHLCV, macro proxies, sector ETF features, and optional best-effort NVDA fundamentals
- Semiconductor peer features for AMD, AVGO, TSM, ASML, MU, QCOM, INTC, and ARM
- Peer business-event proxies from abnormal peer moves, volume spikes, breadth, and stress
- Optional Alpaca option snapshot overlay for ATM straddle implied move, IV/HV, and skew
- Optional Alpaca stock top-of-book/order-flow overlay for best bid/ask, spread, quote imbalance, latest trade, and recent 1-minute trend
- Daily technical features plus weekly macro/sector returns
- ElasticNet/Logit, Random Forest, HistGradientBoosting, optional XGBoost, and a heuristic baseline
- Walk-forward validation with optimized 189-day train windows and 42-day test windows
- Transaction costs, slippage, stop-loss/take-profit, drawdown control, statistical tests, robustness checks, and charts

Run with the Python 3.13 environment available on this machine:

```sh
python3.13 -m nvda_quant_model.main
```

By default this resolves `--end latest` and uses a rolling 24-month window. To anchor
stop/take-profit levels to a live quote you trust:

```sh
python3.13 -m nvda_quant_model.main --price-override 214.30
```

Best-effort Yahoo fundamentals are disabled by default because they are slower and
not strict point-in-time data. Enable them only when you explicitly want that layer:

```sh
python3.13 -m nvda_quant_model.main --include-fundamentals
```

Run a compact walk-forward optimization pass:

```sh
python3.13 -m nvda_quant_model.optimize --max-runs 18
```

Run a high-precision signal search with hard minimum sample-size gates:

```sh
python3.13 -m nvda_quant_model.precision_search --max-candidates 8000 --min-active-days 25 --min-trades 18 --price-override 214.30
```

Add a live option-volatility overlay when an Alpaca option snapshot JSON is available:

```sh
python3.13 -m nvda_quant_model.main \
  --price-override 215.34 \
  --options-snapshot-json nvda_quant_model/outputs/options_snapshot_selected_2026-05-22.json
```

The options layer is intentionally kept out of historical model training unless a
point-in-time options history is supplied. It is used as a current-market overlay
for implied range, IV/HV regime, and call/put skew so the daily signal does not
quietly learn from future option prices.

Validate a selected precision run by year, quarter, rolling windows, and event nodes:

```sh
python3.13 -m nvda_quant_model.time_node_backtest \
  --signals-dir nvda_quant_model/outputs/precision_search_peer_hybrid_12000 \
  --output-dir nvda_quant_model/outputs/time_node_peer_hybrid
```

Run the checkpointed long optimizer for extended searches:

```sh
python3.13 -m nvda_quant_model.long_run_optimizer \
  --hours 30 \
  --resume \
  --price-override 215.34
```

It writes every evaluated rule to `long_run_results.csv`, continuously updates
`best_precision.json`, and can resume without rechecking labels already present
in the results file.

Current audited production baseline:

```text
tw147_sw63_10d_return_mq0.60_vq0.45_smh++obv++rsi<75+p60>0.96+vix<0.9
```

The source optimizer row used a 2.5% stop and 4.5% take-profit. A 2026-05-25
24/36/60M retest found that keeping the 2.5% stop and tightening take-profit
to 4.0% passed the full threshold set across all three windows. That audited
override lives in `production_model.py` and is applied when rows are converted
back into executable precision rules.

Run the overfit/fragility audit before promoting any new production rule:

```sh
python3.13 -m nvda_quant_model.overfit_audit
```

The audit checks hard-gate margins, train/validation degradation, stop/take
parameter-neighborhood fragility, yearly regime weakness, probability
calibration, and terminal-return interval coverage.

Search for a baseline-preserving 60M repair overlay:

```sh
python3.13 -m nvda_quant_model.stable_repair_optimizer \
  --hours 6 \
  --resume \
  --price-override 215.34 \
  --output-dir nvda_quant_model/outputs/stable_repair_optimizer
```

This optimizer implements the constrained repair route: the audited production
baseline remains the core signal, and a repair rule may only add small exposure
when the core is flat. A candidate is rejected if 24M or 36M annualized return,
Sharpe, max drawdown, win rate, or profit factor is worse than the current
baseline. Only then is it scored on 60M lift and rolling OOS behavior. The
repair overlay is a watchlist/search layer; it does not replace the production
baseline unless a candidate later passes the full 24/36/60M and OOS promotion
gates.

Build the top-rules ensemble with market-regime gating, calibration output, and
optional dynamic sizing:

```sh
python3.13 -m nvda_quant_model.ensemble_research \
  --price-override 215.34 \
  --output-dir nvda_quant_model/outputs/ensemble_research_best_regime_gated \
  --top-n 30 \
  --min-active-days 50 \
  --min-trades 30 \
  --min-vote-weight 0.30 \
  --min-vote-count 3 \
  --prob-threshold 0.50 \
  --block-realized-vol-hot
```

Download or import point-in-time historical option bars and transform them into
model-ready option features:

```sh
python3.13 -m nvda_quant_model.options_history \
  --symbols NVDA260612C00215000,NVDA260612P00215000 \
  --start 2024-05-22 \
  --end 2026-05-22 \
  --underlying-csv nvda_quant_model/cache/NVDA_2022-11-09_2026-05-23.csv
```

The Alpaca option-bars path requires `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY` with options market-data permission. Without a
point-in-time historical option dataset, current option snapshots remain an
overlay and are not inserted into historical backtests.

Fetch real-time NVDA/news-event sentiment and add it to the current prediction:

```sh
python3.13 -m nvda_quant_model.news_sentiment \
  --days 3 \
  --output-dir nvda_quant_model/outputs/news_live

python3.13 -m nvda_quant_model.main \
  --price-override 215.34 \
  --include-live-news \
  --news-days 3 \
  --options-snapshot-json nvda_quant_model/outputs/options_snapshot_selected_2026-05-22.json
```

For historical training, provide a point-in-time news article/features CSV:

```sh
python3.13 -m nvda_quant_model.main \
  --news-history-csv nvda_quant_model/outputs/news_history.csv
```

Live news is intentionally treated as a current overlay unless a historical
point-in-time cache is present.

Train and validate the Hugging Face MTBench news impact overlay as a
distribution-adjustment layer:

```python
from nvda_quant_model.news_impact_overlay import (
    adjust_monte_carlo_distribution,
    load_mtbench_finance_dataset,
    predict_news_impact_scores,
    train_news_impact_bundle,
    walk_forward_news_overlay_validation,
)

df = load_mtbench_finance_dataset()
bundle, metrics = train_news_impact_bundle(df, model_type="logistic", text_method="tfidf")
scores = predict_news_impact_scores(bundle, df.tail(1)).iloc[0]
adjusted = adjust_monte_carlo_distribution(
    base_prob_up=0.52,
    base_mu=0.10,
    base_sigma=0.45,
    option_implied_volatility=0.50,
    news_scores=scores,
)
validation = walk_forward_news_overlay_validation(df)
```

This module uses only article text plus `input_window` / `input_timestamps`
and parsed `technical` keys beginning with `in_`. It excludes `output_window`,
`output_timestamps`, raw mixed `technical`, `out_*`, `overall_*`, `trend`, and
`alignment` from the feature matrix. `trend` and `alignment` are label-only
fields. The layer adjusts Monte Carlo drift and volatility with conservative
parameters and does not emit buy/sell/hold decisions:

```text
adjusted_mu = base_mu + 0.05 * (news_bullish_score - news_bearish_score)
adjusted_sigma = base_sigma * (1 + 0.25 * news_shock_strength)
```

Core principle: this module adjusts the distribution, not the trading decision.

Build the NVDA Event Impact Overlay v1 on point-in-time event rows and pre-event
market data:

```python
from nvda_quant_model.event_overlay import (
    apply_event_overlay_to_distribution,
    score_event_overlay,
    train_event_overlay_model,
    walk_forward_event_overlay_validation,
)

model, metrics = train_event_overlay_model(
    events=nvda_events,
    market_data=nvda_ohlcv,
    external_data=market_context,
    target_horizon=1,
    text_method="tfidf",
)
overlay = score_event_overlay(model, nvda_events.tail(1), nvda_ohlcv, market_context).iloc[0]
adjusted = apply_event_overlay_to_distribution(
    base_prob_up=0.52,
    base_mu=0.10,
    base_sigma=0.45,
    event_overlay=overlay,
)
validation = walk_forward_event_overlay_validation(nvda_events, nvda_ohlcv, market_context)
```

The event layer classifies event types such as earnings, guidance, analyst,
product, supply-chain, export-control, competition, customer-capex, valuation,
and macro AI trade. It builds labels from future returns only inside validation
and training targets; those labels are blocked from feature matrices. The output
contains event distribution adjustments (`event_mu_adjustment`,
`event_sigma_multiplier`, `event_tail_risk_multiplier`) plus reason codes, and
still never emits `signal` or `position`.

Build a clean NVDA point-in-time historical event store before using news/events
in historical validation:

```sh
python3.13 -m nvda_quant_model.event_overlay.history_builder \
  --start 2019-01-01 \
  --end latest \
  --sources sec,gdelt \
  --with-labels \
  --output-dir nvda_quant_model/outputs/event_overlay_history
```

The builder writes separate files for event inputs and future labels:

- `nvda_historical_events.csv`: publication-time fields only.
- `nvda_event_labels.csv`: future return/direction labels for training and validation only.
- `nvda_news_history_articles.csv`: compatibility export for `--news-history-csv`.
- `event_store_audit.json`: sample count, month coverage, source concentration, and whether the store is ready for serious walk-forward validation.

If the store has fewer than 250 labeled events, fewer than 24 covered months, or
too much single-source concentration, the backtest path returns
`insufficient_point_in_time_sample` instead of pretending the event layer has
validated. Live news remains a current overlay only.

Run the 60M/OOS stable-candidate validator before promoting any optimizer rule:

```sh
python3.13 -m nvda_quant_model.stable_candidate_optimizer \
  --strict-results nvda_quant_model/outputs/long_run_optimizer_repaired/long_run_results.csv \
  --high-sample-results nvda_quant_model/outputs/high_sample_optimizer_repaired/high_sample_results.csv \
  --output-dir nvda_quant_model/outputs/stable_candidate_optimizer \
  --hours 6 \
  --resume
```

This validator keeps the production baseline unless a challenger passes
24/36/60M hard gates, 60M sample requirements, rolling OOS degradation checks,
and regime diagnostics. A candidate that looks good in 24M but fails 60M is
reported as `reject_long_window_failure`, not promoted.

Fetch real-time top-of-book buy/sell pressure and recent 1-minute trend:

```sh
python3.13 -m nvda_quant_model.live_order_flow \
  --symbol NVDA \
  --feed iex \
  --minutes 60 \
  --output-dir nvda_quant_model/outputs/live_order_flow

python3.13 -m nvda_quant_model.main \
  --precision-rule-json nvda_quant_model/outputs/strict_selection_trades30/strict_best_precision.json \
  --include-live-order-flow \
  --order-flow-feed iex \
  --order-flow-minutes 60
```

For US equities, Alpaca stock snapshots provide top-of-book best bid/ask,
latest trade, latest minute bar, daily bar, and previous daily bar. That is
not a full Level-2 depth book. If you need full multi-level order-book depth,
connect a dedicated Level-2 provider such as Nasdaq TotalView, IEX DEEP, or
another depth feed and keep it as a separate real-time overlay.

Backtest the live order-flow proxy on recent historical 1-minute bars:

```sh
python3.13 -m nvda_quant_model.order_flow_backtest \
  --symbol NVDA \
  --period 8d \
  --horizon-minutes 15 \
  --output-dir nvda_quant_model/outputs/order_flow_backtest

python3.13 -m nvda_quant_model.order_flow_backtest \
  --symbol NVDA \
  --period 8d \
  --calibrate \
  --train-sessions 4 \
  --output-dir nvda_quant_model/outputs/order_flow_backtest_calibrated
```

This proxy backtest validates the minute-trend, VWAP, volume-acceleration, and
day-range-position parts of the live overlay. It does not validate historical
bid/ask-size imbalance because Yahoo 1-minute bars do not contain quotes. The
calibrated mode disables the proxy when rolling training sessions fail minimum
edge gates, so the live layer remains an execution filter instead of becoming
an unvalidated entry engine.

Outputs are written to `nvda_quant_model/outputs/`:

- `report.md`
- `summary.json`
- `signals.csv`
- `equity_curve.csv`
- `trades.csv`
- `walk_forward.csv`
- `feature_importance.csv`
- `model_comparison.csv`
- `data_quality.csv`
- `data_quality.json`
- `options_volatility_report.json`
- `equity_curve.png`
- `drawdown.png`
- `monthly_returns_heatmap.png`
- `rolling_sharpe.png`

Optimization runs write CSV/JSON artifacts under `nvda_quant_model/outputs/optimization*/`.
Precision searches write artifacts under `nvda_quant_model/outputs/precision_search*/`.
Long optimizer runs write artifacts under `nvda_quant_model/outputs/long_run_optimizer/`.
Ensemble runs write artifacts under `nvda_quant_model/outputs/ensemble_research*/`.
Live news runs write artifacts under `nvda_quant_model/outputs/news_live*/`.

The report is a research artifact, not investment advice. Free yfinance fundamentals are not a professional point-in-time dataset, so live-grade use should replace that layer with audited point-in-time fundamentals.
