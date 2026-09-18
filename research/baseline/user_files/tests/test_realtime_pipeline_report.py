from __future__ import annotations

from research_ai_supply_chain.realtime_data_pipeline.pipeline.report import build_markdown_report


def test_report_includes_vix_risk_gate() -> None:
    report = build_markdown_report(
        run_id="test",
        pages=[],
        events=[],
        filings=[],
        prices=[],
        global_markets=[
            {
                "name": "CBOE VIX",
                "symbol": "^VIX",
                "price": "24.2",
                "pct_from_prev_close": "9.1",
                "pct_from_open": "3.2",
                "error": "",
            }
        ],
        global_summary=[],
    )

    assert "## VIX Risk Gate" in report
    assert "`^VIX` level `24.20`" in report
    assert "gate `risk_watch`" in report
    assert "cap SOXL size" in report
