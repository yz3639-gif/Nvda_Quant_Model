from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nvda_quant_model.config import PROJECT_ROOT
from nvda_quant_model.data.load_data import load_ohlcv
from nvda_quant_model.event_overlay.news_history import (
    DEFAULT_USER_AGENT,
    build_historical_event_store,
    run_historical_event_overlay_backtest,
    save_event_store_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a point-in-time NVDA historical news/event store.")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--ticker", default="NVDA", choices=["NVDA"], help="Current implementation is NVDA-only.")
    parser.add_argument("--sources", default="sec,gdelt", help="Comma-separated sources: sec,gdelt")
    parser.add_argument("--vendor-csv", action="append", default=[], help="Optional point-in-time vendor event CSV.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "event_overlay_history"))
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--with-labels", action="store_true", help="Build separate future-label table from NVDA OHLCV.")
    parser.add_argument("--run-backtest", action="store_true", help="Run walk-forward validation only if sample gates pass.")
    parser.add_argument("--min-events", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true", help="Build and audit without writing artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    end = pd.Timestamp.utcnow().strftime("%Y-%m-%d") if str(args.end).lower() in {"latest", "today", "now", "auto"} else args.end
    sources = tuple(item.strip() for item in args.sources.split(",") if item.strip())
    result = build_historical_event_store(
        start=args.start,
        end=end,
        sources=sources,
        vendor_csvs=tuple(args.vendor_csv),
        ticker=args.ticker,
        user_agent=args.user_agent,
    )
    market_data = None
    if args.with_labels or args.run_backtest:
        market_data = load_ohlcv(args.ticker, args.start, end)
    print(json.dumps(result.audit, indent=2, ensure_ascii=False, default=str))
    if not args.dry_run:
        paths = save_event_store_outputs(result.events, output_dir, market_data=market_data if args.with_labels else None)
        print(json.dumps(paths, indent=2, ensure_ascii=False))
    if args.run_backtest:
        backtest = run_historical_event_overlay_backtest(result.events, market_data, min_events=args.min_events)
        print(json.dumps({"status": backtest["status"], "audit": backtest["audit"]}, indent=2, ensure_ascii=False, default=str))
        if not args.dry_run:
            (output_dir / "backtest_summary.csv").write_text("", encoding="utf-8")
            backtest["summary"].to_csv(output_dir / "backtest_summary.csv", index=False)
            backtest["windows"].to_csv(output_dir / "backtest_windows.csv", index=False)
            backtest["group_summary"].to_csv(output_dir / "backtest_group_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
