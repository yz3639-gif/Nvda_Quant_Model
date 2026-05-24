#!/usr/bin/env python3
"""
Build a simple consensus-holdings growth model from public 13F holdings.

The model is deliberately transparent:
- consensus breadth: how many tracked public portfolios hold the security
- capital support: total reported 13F market value
- six-month momentum: adjusted close return over the lookback window
- trend confirmation: latest price vs. moving averages
- risk-adjusted momentum: six-month return adjusted by realized volatility

It produces a ranking score, not investment advice and not a backtested
statistical probability.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests


USER_AGENT = "Codex research yuangzuo@example.com"
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
TICKER_OVERRIDES = {
    # OpenFIGI occasionally returns a European venue before the US-listed ticker.
    "436440101": "HOLX",  # Hologic
    "03152W109": "FOLD",  # Amicus Therapeutics
}


@dataclass
class HoldingGroup:
    cusip: str
    issuer: str
    security_class: str
    holder_count: int
    holders: list[str]
    total_value_usd: int
    total_amount: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdings",
        default="outputs/investor_holdings_2026_05_07/all_public_13f_holdings.csv",
        help="Path to the combined public 13F holdings CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/quant_model_2026_05_07",
        help="Directory for model outputs.",
    )
    parser.add_argument(
        "--min-holder-count",
        type=int,
        default=3,
        help="Only score securities held by at least this many tracked portfolios.",
    )
    parser.add_argument(
        "--start-date",
        default="2025-11-07",
        help="Lookback start date for six-month performance.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-05-07",
        help="Lookback end date.",
    )
    return parser.parse_args()


def read_holding_groups(path: str, min_holder_count: int) -> list[HoldingGroup]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            is_share = row.get("amount_type") == "SH"
            is_common_side = not row.get("put_call")
            if is_share and is_common_side and row.get("cusip"):
                grouped[row["cusip"]].append(row)

    groups: list[HoldingGroup] = []
    for cusip, rows in grouped.items():
        holders = sorted({row["proxy_manager"] for row in rows})
        if len(holders) < min_holder_count:
            continue
        first = rows[0]
        groups.append(
            HoldingGroup(
                cusip=cusip,
                issuer=first["issuer"],
                security_class=first["class"],
                holder_count=len(holders),
                holders=holders,
                total_value_usd=sum(int(row["value_usd"]) for row in rows),
                total_amount=sum(int(row["amount"]) for row in rows),
            )
        )

    return sorted(groups, key=lambda g: (-g.holder_count, -g.total_value_usd, g.issuer))


def map_cusips_with_openfigi(groups: list[HoldingGroup], cache_path: str) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cache = json.load(handle)

    missing = [group.cusip for group in groups if group.cusip not in cache]
    if missing:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        for batch_start in range(0, len(missing), 10):
            batch = missing[batch_start : batch_start + 10]
            payload = [{"idType": "ID_CUSIP", "idValue": cusip} for cusip in batch]
            response = requests.post(OPENFIGI_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 429:
                time.sleep(65)
                response = requests.post(OPENFIGI_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json()
            for cusip, result in zip(batch, results):
                cache[cusip] = choose_figi_mapping(result)
            time.sleep(6.5)

        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)

    return cache


def choose_figi_mapping(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("data") or []
    if not data:
        return {"ticker": "", "name": "", "security_type": "", "market_sector": "", "exch_code": ""}

    def score(item: dict[str, Any]) -> tuple[int, int, int, int]:
        exch = item.get("exchCode") or ""
        sector = item.get("marketSector") or ""
        sec_type2 = item.get("securityType2") or ""
        sec_type = item.get("securityType") or ""
        return (
            1 if exch == "US" else 0,
            1 if sector == "Equity" else 0,
            1 if sec_type2 in {"Common Stock", "ADR", "REIT", "ETF"} else 0,
            0 if "Warrant" in sec_type or "Right" in sec_type else 1,
        )

    best = max(data, key=score)
    return {
        "ticker": best.get("ticker", ""),
        "name": best.get("name", ""),
        "security_type": best.get("securityType2") or best.get("securityType", ""),
        "market_sector": best.get("marketSector", ""),
        "exch_code": best.get("exchCode", ""),
        "figi": best.get("figi", ""),
    }


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace("/", "-").replace(" ", "-")


def fetch_yahoo_prices(symbol: str, start_date: str, end_date: str) -> list[dict[str, float]]:
    start = dt.datetime.fromisoformat(start_date).replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromisoformat(end_date).replace(tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    response = requests.get(
        YAHOO_CHART_URL.format(symbol=quote(symbol)),
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    chart = response.json().get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])
    result = (chart.get("result") or [None])[0]
    if not result:
        return []
    timestamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []
    prices: list[dict[str, float]] = []
    for ts, close in zip(timestamps, quote_data):
        if close is None:
            continue
        prices.append(
            {
                "date": dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat(),
                "close": float(close),
            }
        )
    return prices


def price_metrics(prices: list[dict[str, float]]) -> dict[str, float | str]:
    if len(prices) < 30:
        return {}

    closes = [row["close"] for row in prices]
    returns = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes)) if closes[i - 1] > 0]
    first = closes[0]
    last = closes[-1]
    ret_6m = (last / first) - 1
    ret_3m = (last / closes[max(0, len(closes) - 63)]) - 1 if len(closes) > 63 else ret_6m
    ret_1m = (last / closes[max(0, len(closes) - 21)]) - 1 if len(closes) > 21 else ret_6m
    ma50 = sum(closes[-50:]) / min(50, len(closes))
    ma120 = sum(closes[-120:]) / min(120, len(closes))
    volatility = (sum((r - (sum(returns) / len(returns))) ** 2 for r in returns) / max(1, len(returns) - 1)) ** 0.5
    annualized_volatility = volatility * (252**0.5)
    peak = closes[0]
    max_drawdown = 0.0
    for close in closes:
        peak = max(peak, close)
        max_drawdown = min(max_drawdown, (close / peak) - 1)

    return {
        "price_start_date": prices[0]["date"],
        "price_end_date": prices[-1]["date"],
        "start_adj_close": first,
        "end_adj_close": last,
        "return_6m": ret_6m,
        "return_3m": ret_3m,
        "return_1m": ret_1m,
        "ma50": ma50,
        "ma120": ma120,
        "above_ma50": 1.0 if last > ma50 else 0.0,
        "ma50_above_ma120": 1.0 if ma50 > ma120 else 0.0,
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
    }


def percentile_rank(values: list[float], value: float) -> float:
    if not values:
        return 0.5
    ordered = sorted(values)
    below = sum(1 for v in ordered if v < value)
    equal = sum(1 for v in ordered if v == value)
    return (below + 0.5 * equal) / len(ordered)


def clipped(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def add_model_scores(rows: list[dict[str, Any]]) -> None:
    max_holders = max(row["holder_count"] for row in rows)
    max_log_value = max(math.log1p(row["total_value_usd"]) for row in rows)
    risk_adjusted_values = [
        row["return_6m"] / row["annualized_volatility"]
        for row in rows
        if row.get("annualized_volatility", 0) > 0
    ]

    for row in rows:
        breadth_score = row["holder_count"] / max_holders
        capital_score = math.log1p(row["total_value_usd"]) / max_log_value
        momentum_score = clipped((row["return_6m"] + 0.20) / 0.80, 0.0, 1.0)
        trend_score = 0.5 * row["above_ma50"] + 0.5 * row["ma50_above_ma120"]
        risk_adjusted = row["return_6m"] / row["annualized_volatility"] if row["annualized_volatility"] > 0 else 0
        risk_adjusted_score = percentile_rank(risk_adjusted_values, risk_adjusted)
        drawdown_penalty = clipped(abs(row["max_drawdown"]) / 0.45, 0.0, 1.0)

        raw_score = (
            0.35 * breadth_score
            + 0.20 * capital_score
            + 0.25 * momentum_score
            + 0.12 * trend_score
            + 0.08 * risk_adjusted_score
            - 0.08 * drawdown_penalty
        )
        growth_probability = sigmoid(-1.15 + 3.2 * raw_score)

        row.update(
            {
                "breadth_score": breadth_score,
                "capital_score": capital_score,
                "momentum_score": momentum_score,
                "trend_score": trend_score,
                "risk_adjusted_score": risk_adjusted_score,
                "drawdown_penalty": drawdown_penalty,
                "model_score": raw_score,
                "growth_probability": growth_probability,
            }
        )


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_outputs(rows: list[dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "consensus_growth_model.csv")
    fieldnames = [
        "rank_by_probability",
        "rank_by_holder_count",
        "ticker",
        "issuer",
        "security_class",
        "security_type",
        "cusip",
        "holder_count",
        "holders",
        "total_value_usd",
        "total_amount",
        "return_6m",
        "return_3m",
        "return_1m",
        "annualized_volatility",
        "max_drawdown",
        "model_score",
        "growth_probability",
        "price_start_date",
        "price_end_date",
        "start_adj_close",
        "end_adj_close",
        "breadth_score",
        "capital_score",
        "momentum_score",
        "trend_score",
        "risk_adjusted_score",
        "drawdown_penalty",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    md_path = os.path.join(out_dir, "top_consensus_growth_model.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Consensus holdings growth model\n\n")
        handle.write(
            "Universe: common-share 13F securities held by at least 3 tracked public portfolios. "
            "Probability is a transparent ranking score for the next 3-6 months, not a backtested forecast.\n\n"
        )
        handle.write("| Rank | Ticker | Issuer | Holders | 6M return | Growth probability | 13F value |\n")
        handle.write("|---:|---|---|---:|---:|---:|---:|\n")
        for row in rows[:30]:
            handle.write(
                f"| {row['rank_by_probability']} | {row['ticker']} | {row['issuer']} | "
                f"{row['holder_count']} | {fmt_pct(row['return_6m'])} | "
                f"{fmt_pct(row['growth_probability'])} | ${row['total_value_usd']:,.0f} |\n"
            )


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    groups = read_holding_groups(args.holdings, args.min_holder_count)
    mapping = map_cusips_with_openfigi(groups, os.path.join(args.out_dir, "openfigi_cache.json"))

    rows: list[dict[str, Any]] = []
    for group in groups:
        mapped = mapping.get(group.cusip, {})
        ticker = TICKER_OVERRIDES.get(group.cusip) or mapped.get("ticker") or ""
        if not ticker:
            continue
        try:
            prices = fetch_yahoo_prices(yahoo_symbol(ticker), args.start_date, args.end_date)
            metrics = price_metrics(prices)
        except Exception as exc:  # noqa: BLE001 - keep the batch moving and expose misses.
            print(f"skip {ticker} {group.issuer}: {exc}")
            continue
        if not metrics:
            continue

        rows.append(
            {
                "ticker": ticker,
                "issuer": group.issuer,
                "security_class": group.security_class,
                "security_type": mapped.get("security_type", ""),
                "cusip": group.cusip,
                "holder_count": group.holder_count,
                "holders": "; ".join(group.holders),
                "total_value_usd": group.total_value_usd,
                "total_amount": group.total_amount,
                **metrics,
            }
        )

    rows.sort(key=lambda row: (-row["holder_count"], -row["total_value_usd"]))
    for rank, row in enumerate(rows, 1):
        row["rank_by_holder_count"] = rank

    add_model_scores(rows)

    rows.sort(key=lambda row: (-row["growth_probability"], -row["holder_count"], -row["total_value_usd"]))
    for rank, row in enumerate(rows, 1):
        row["rank_by_probability"] = rank

    write_outputs(rows, args.out_dir)
    print(f"scored {len(rows)} securities")
    print(os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
