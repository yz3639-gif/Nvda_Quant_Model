---
name: nvda-news-sentiment
description: Use when updating the NVDA quant model with real-time news, event sentiment, earnings/analyst/regulatory/product headlines, or when the user says "nvda新闻更新", "nvda更新", "抓新闻", or asks to include news in NVDA forecasts.
---

# NVDA News Sentiment Workflow

Run commands from the repository root; the core module is `nvda_quant_model/`.

## Workflow

1. Fetch live news first:

```sh
python3.13 -m nvda_quant_model.news_sentiment \
  --days 3 \
  --output-dir nvda_quant_model/outputs/news_live
```

2. Run the model with live news overlay:

```sh
python3.13 -m nvda_quant_model.main \
  --price-override 215.34 \
  --include-live-news \
  --news-days 3 \
  --options-snapshot-json nvda_quant_model/outputs/options_snapshot_selected_2026-05-22.json
```

3. For larger-sample production signal, prefer:

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

## Integrity Rules

- Live news is an overlay for the current prediction only.
- Do not insert live news into historical backtests unless a point-in-time news cache is supplied.
- If a historical news CSV is available, load it via `--news-history-csv` on `nvda_quant_model.main`.
- Report article count, sentiment score, signal, risk flags, and top headlines.
