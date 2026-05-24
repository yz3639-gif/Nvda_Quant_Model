#!/usr/bin/env python3
"""
Quant valuation model for Cerebras Systems (NASDAQ: CBRS).

The model fetches comparable-company data with yfinance when available and
falls back to manually maintained public-market multiples if Yahoo data is
unavailable. All major assumptions live in CONFIG for quick tuning.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / "outputs" / "cbrs_valuation_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:  # pragma: no cover - fallback is exercised only without yfinance.
    yf = None


CONFIG = {
    "as_of_date": "2026-05-13",
    "output_dir": "outputs/cbrs_valuation_2026_05_13",
    "cbrs": {
        "ipo_mid_price": 155.0,
        "ipo_low_price": 150.0,
        "ipo_high_price": 160.0,
        "reported_2025_revenue": 510_000_000,
        "reported_2025_net_income": 237_800_000,
        "fully_diluted_valuation_at_high": 48_800_000_000,
        "fully_diluted_reference_price": 160.0,
        "class_a_offered": 30_000_000,
        "estimated_post_ipo_net_cash": 4_500_000_000,
    },
    "comps": ["NVDA", "AMD", "AVGO", "MRVL", "ARM", "SMCI"],
    "manual_comp_fallback": {
        "NVDA": {"ps_ttm": 24.85, "ev_sales_ntm": 14.32, "pe_ntm": 19.52, "eps_growth_next_year": 0.3531},
        "AMD": {"ps_ttm": 19.52, "ev_sales_ntm": 14.66, "pe_ntm": 34.73, "eps_growth_next_year": 0.7594},
        "AVGO": {"ps_ttm": 29.07, "ev_sales_ntm": 19.52, "pe_ntm": 23.14, "eps_growth_next_year": 0.5857},
        "MRVL": {"ps_ttm": 17.55, "ev_sales_ntm": 13.44, "pe_ntm": 30.27, "eps_growth_next_year": 0.4184},
        "ARM": {"ps_ttm": 47.36, "ev_sales_ntm": 36.48, "pe_ntm": 68.84, "eps_growth_next_year": 0.3978},
        "SMCI": {"ps_ttm": 0.59, "ev_sales_ntm": 0.69, "pe_ntm": 10.18, "eps_growth_next_year": 0.2427},
    },
    "scenario_order": ["bear", "base", "bull"],
    "scenario_labels": {"bear": "悲观情景", "base": "基准情景", "bull": "乐观情景"},
    "ai_pure_play_premium": {"bear": 0.30, "base": 0.55, "bull": 0.80},
    "revenue_growth_y1_to_y3": {"bear": 0.80, "base": 1.20, "bull": 1.50},
    "terminal_growth_range_y4_to_y10": {"bear": 0.25, "base": 0.325, "bull": 0.40},
    "operating_margin": {"bear": 0.35, "base": 0.45, "bull": 0.55},
    "wacc": {"bear": 0.16, "base": 0.14, "bull": 0.12},
    "terminal_growth": 0.03,
    "cash_tax_rate": 0.18,
    "fcf_conversion": 0.85,
    "risk_discounts": {
        "bear": {
            "customer_concentration": 0.25,
            "lockup_overhang": 0.15,
            "ai_capex_cycle": 0.10,
            "nvidia_competition_gap": 0.10,
        },
        "base": {
            "customer_concentration": 0.20,
            "lockup_overhang": 0.125,
            "ai_capex_cycle": 0.075,
            "nvidia_competition_gap": 0.10,
        },
        "bull": {
            "customer_concentration": 0.15,
            "lockup_overhang": 0.10,
            "ai_capex_cycle": 0.05,
            "nvidia_competition_gap": 0.10,
        },
    },
    "weights": {
        "可比 P/S": 0.25,
        "可比 EV/Sales": 0.25,
        "DCF": 0.30,
        "PEG": 0.20,
    },
    "peg_reference_tickers": ["NVDA", "AVGO"],
}


def cbrs_diluted_shares(config: dict) -> float:
    cbrs = config["cbrs"]
    return cbrs["fully_diluted_valuation_at_high"] / cbrs["fully_diluted_reference_price"]


def risk_multiplier(config: dict, scenario: str) -> float:
    multiplier = 1.0
    for discount in config["risk_discounts"][scenario].values():
        multiplier *= 1.0 - discount
    return multiplier


def fetch_comps(config: dict) -> tuple[pd.DataFrame, str]:
    rows: list[dict] = []
    source = "yfinance"

    if yf is None:
        source = "manual_fallback"
    else:
        for ticker in config["comps"]:
            try:
                tk = yf.Ticker(ticker)
                info = tk.get_info()
                revenue_estimate = tk.revenue_estimate
                earnings_estimate = tk.earnings_estimate

                fy1_revenue = float(revenue_estimate.loc["0y", "avg"])
                next_year_eps_growth = float(earnings_estimate.loc["+1y", "growth"])
                ev = float(info["enterpriseValue"])
                rows.append(
                    {
                        "ticker": ticker,
                        "price": info.get("currentPrice"),
                        "market_cap": info.get("marketCap"),
                        "enterprise_value": ev,
                        "ttm_revenue": info.get("totalRevenue"),
                        "fy1_revenue_estimate": fy1_revenue,
                        "ps_ttm": info.get("priceToSalesTrailing12Months"),
                        "ev_sales_ttm": info.get("enterpriseToRevenue"),
                        "ev_sales_ntm": ev / fy1_revenue if fy1_revenue else np.nan,
                        "pe_ntm": info.get("forwardPE"),
                        "eps_growth_next_year": next_year_eps_growth,
                    }
                )
            except Exception as exc:
                print(f"warning: yfinance failed for {ticker}: {exc}", file=sys.stderr)

    if len(rows) != len(config["comps"]):
        source = "manual_fallback"
        rows = []
        for ticker, data in config["manual_comp_fallback"].items():
            rows.append({"ticker": ticker, **data})

    df = pd.DataFrame(rows)
    for col in ["ps_ttm", "ev_sales_ntm", "pe_ntm", "eps_growth_next_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, source


def comp_stats(comps: pd.DataFrame) -> pd.DataFrame:
    metrics = ["ps_ttm", "ev_sales_ntm", "pe_ntm"]
    rows = []
    for metric in metrics:
        values = comps[metric].dropna()
        rows.append({"metric": metric, "median": values.median(), "mean": values.mean()})
    return pd.DataFrame(rows)


def dcf_price(config: dict, scenario: str, wacc: float | None = None, margin: float | None = None) -> float:
    revenue = config["cbrs"]["reported_2025_revenue"]
    shares = cbrs_diluted_shares(config)
    net_cash = config["cbrs"]["estimated_post_ipo_net_cash"]
    wacc = config["wacc"][scenario] if wacc is None else wacc
    margin = config["operating_margin"][scenario] if margin is None else margin

    high_growth = config["revenue_growth_y1_to_y3"][scenario]
    mature_growth = config["terminal_growth_range_y4_to_y10"][scenario]
    growths = [high_growth] * 3 + list(np.linspace(high_growth * 0.70, mature_growth, 7))

    pv = 0.0
    final_fcf = 0.0
    for year, growth in enumerate(growths, start=1):
        revenue *= 1.0 + growth
        fcf = revenue * margin * (1.0 - config["cash_tax_rate"]) * config["fcf_conversion"]
        pv += fcf / ((1.0 + wacc) ** year)
        final_fcf = fcf

    terminal_fcf = final_fcf * (1.0 + config["terminal_growth"])
    terminal_value = terminal_fcf / (wacc - config["terminal_growth"])
    pv += terminal_value / ((1.0 + wacc) ** len(growths))
    equity_value = pv + net_cash
    return equity_value / shares


def valuation(config: dict, comps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shares = cbrs_diluted_shares(config)
    revenue_2025 = config["cbrs"]["reported_2025_revenue"]
    net_cash = config["cbrs"]["estimated_post_ipo_net_cash"]

    ps_median = comps["ps_ttm"].median()
    ev_sales_median = comps["ev_sales_ntm"].median()

    peg_refs = comps[comps["ticker"].isin(config["peg_reference_tickers"])].copy()
    peg_refs["peg"] = peg_refs["pe_ntm"] / (peg_refs["eps_growth_next_year"] * 100.0)
    peer_peg = peg_refs["peg"].median()

    raw_rows = []
    risk_rows = []
    for scenario in config["scenario_order"]:
        premium = 1.0 + config["ai_pure_play_premium"][scenario]
        y1_growth = config["revenue_growth_y1_to_y3"][scenario]
        revenue_y1 = revenue_2025 * (1.0 + y1_growth)
        margin = config["operating_margin"][scenario]

        ps_price = (ps_median * premium * revenue_2025) / shares
        ev_sales_price = ((ev_sales_median * premium * revenue_y1) + net_cash) / shares
        dcf = dcf_price(config, scenario)

        forward_eps = (revenue_y1 * margin) / shares
        peg_pe = peer_peg * (y1_growth * 100.0)
        peg_price = forward_eps * peg_pe

        raw = {
            "scenario": scenario,
            "可比 P/S": ps_price,
            "可比 EV/Sales": ev_sales_price,
            "DCF": dcf,
            "PEG": peg_price,
        }
        raw_rows.append(raw)

        multiplier = risk_multiplier(config, scenario)
        risk = {"scenario": scenario, "risk_multiplier": multiplier}
        for method in config["weights"]:
            risk[method] = raw[method] * multiplier
        risk["加权平均"] = sum(risk[method] * weight for method, weight in config["weights"].items())
        risk_rows.append(risk)

    raw_df = pd.DataFrame(raw_rows)
    risk_df = pd.DataFrame(risk_rows)

    heatmap_rows = []
    for wacc in [0.12, 0.14, 0.16]:
        for margin in [0.35, 0.45, 0.55]:
            price = dcf_price(config, "base", wacc=wacc, margin=margin)
            heatmap_rows.append({"WACC": wacc, "长期利润率": margin, "DCF价格": price})
    heatmap_df = pd.DataFrame(heatmap_rows)
    return raw_df, risk_df, heatmap_df


def format_price(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"${value:,.0f}"


def save_heatmap(heatmap_df: pd.DataFrame, output_dir: Path) -> Path:
    matrix = heatmap_df.pivot(index="长期利润率", columns="WACC", values="DCF价格")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    image = ax.imshow(matrix.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"{col:.0%}" for col in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{idx:.0%}" for idx in matrix.index])
    ax.set_xlabel("WACC")
    ax.set_ylabel("Long-term operating margin")
    ax.set_title("CBRS DCF Sensitivity: Price per Share")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format_price(matrix.iloc[i, j]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, label="Price per share")
    fig.tight_layout()
    output = output_dir / "cbrs_dcf_sensitivity_heatmap.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    output_dir = PROJECT_ROOT / CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    comps, data_source = fetch_comps(CONFIG)
    stats = comp_stats(comps)
    raw_df, risk_df, heatmap_df = valuation(CONFIG, comps)

    presentation = (
        risk_df.set_index("scenario")[["可比 P/S", "可比 EV/Sales", "DCF", "PEG", "加权平均"]]
        .rename(index=CONFIG["scenario_labels"])
        .T[["乐观情景", "基准情景", "悲观情景"]]
    )

    comps.to_csv(output_dir / "comparable_multiples.csv", index=False)
    stats.to_csv(output_dir / "comparable_multiples_summary.csv", index=False)
    raw_df.to_csv(output_dir / "valuation_raw_pre_risk.csv", index=False)
    risk_df.to_csv(output_dir / "valuation_risk_adjusted.csv", index=False)
    presentation.map(format_price).to_csv(output_dir / "valuation_summary_formatted.csv")
    heatmap_df.to_csv(output_dir / "dcf_sensitivity.csv", index=False)
    heatmap_path = save_heatmap(heatmap_df, output_dir)

    result = {
        "data_source": data_source,
        "diluted_shares": cbrs_diluted_shares(CONFIG),
        "risk_multipliers": {scenario: risk_multiplier(CONFIG, scenario) for scenario in CONFIG["scenario_order"]},
        "peer_peg_nvda_avgo": (
            comps[comps["ticker"].isin(CONFIG["peg_reference_tickers"])]
            .assign(peg=lambda x: x["pe_ntm"] / (x["eps_growth_next_year"] * 100.0))[["ticker", "peg"]]
            .to_dict("records")
        ),
        "summary": presentation.to_dict(),
        "heatmap": str(heatmap_path),
    }
    with open(output_dir / "valuation_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Data source: {data_source}")
    print("\nComparable multiples:")
    print(comps[["ticker", "ps_ttm", "ev_sales_ntm", "pe_ntm", "eps_growth_next_year"]].round(3).to_string(index=False))
    print("\nComparable summary:")
    print(stats.round(3).to_string(index=False))
    print("\nRisk-adjusted valuation summary:")
    print(presentation.map(format_price).to_string())
    print(f"\nHeatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
