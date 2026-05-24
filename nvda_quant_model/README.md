# NVDA Multi-Factor Quant Model

This package implements a walk-forward, multi-factor NVDA strategy research workflow:

- yfinance OHLCV, macro proxies, sector ETF features, and optional best-effort NVDA fundamentals
- Semiconductor peer features for AMD, AVGO, TSM, ASML, MU, QCOM, INTC, and ARM
- Peer business-event proxies from abnormal peer moves, volume spikes, breadth, and stress
- Optional Alpaca option snapshot overlay for ATM straddle implied move, IV/HV, and skew
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
