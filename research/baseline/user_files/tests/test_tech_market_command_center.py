from __future__ import annotations

import json
from pathlib import Path

from research_ai_supply_chain.tech_market_command_center.candidate_ranking import build_candidate_ranking
from research_ai_supply_chain.tech_market_command_center.cli import main
from research_ai_supply_chain.tech_market_command_center.data_quality import build_data_audit
from research_ai_supply_chain.tech_market_command_center.data_sources import ProviderChainQuoteProvider
from research_ai_supply_chain.tech_market_command_center.execution_gates import build_execution_gates
from research_ai_supply_chain.tech_market_command_center.executive_summary import build_executive_summary
from research_ai_supply_chain.tech_market_command_center.historical_cases import rank_historical_cases
from research_ai_supply_chain.tech_market_command_center.historical_research import compute_analogues, summarize_events
from research_ai_supply_chain.tech_market_command_center.open_time_triggers import build_open_time_triggers
from research_ai_supply_chain.tech_market_command_center.operator_checklist import build_operator_checklist
from research_ai_supply_chain.tech_market_command_center.operator_manifest import build_manifest_validation, build_operator_manifest
from research_ai_supply_chain.tech_market_command_center.live_readiness import build_live_readiness
from research_ai_supply_chain.tech_market_command_center.market_snapshot import (
    DEFAULT_FIXTURE,
    build_provider,
    build_snapshot,
    classify_data_quality,
    load_config,
)
from research_ai_supply_chain.tech_market_command_center.models import Quote
from research_ai_supply_chain.tech_market_command_center.portfolio_risk import build_risk_report
from research_ai_supply_chain.tech_market_command_center.pre_open_runbook import build_pre_open_runbook, render_terminal
from research_ai_supply_chain.tech_market_command_center.preflight import build_live_market_preflight
from research_ai_supply_chain.tech_market_command_center.report import build_strategy_table
from research_ai_supply_chain.tech_market_command_center.runbook_validation import build_runbook_validation
from research_ai_supply_chain.tech_market_command_center.scenario_lab import build_matrix_rows
from research_ai_supply_chain.tech_market_command_center.scoring import build_scorecard
from research_ai_supply_chain.tech_market_command_center.sector_breadth import build_sector_breadth
from research_ai_supply_chain.tech_market_command_center.strategy_engine import build_scenarios, probability_summary


def fixture_snapshot():
    config = load_config()
    provider = build_provider("fixture", fixture_path=DEFAULT_FIXTURE)
    return build_snapshot(as_of="2026-06-08", session="monday_open", config=config, provider=provider), config


def market_hours_snapshot():
    config = load_config()
    fixture_path = Path("research_ai_supply_chain/tech_market_command_center/fixtures/market_hours_sample.json")
    provider = build_provider("fixture", fixture_path=fixture_path)
    return (
        build_snapshot(
            as_of="2026-06-08",
            session="monday_open",
            config=config,
            provider=provider,
            include_paid_placeholder=False,
            generated_at_utc="2026-06-08T14:35:00+00:00",
        ),
        config,
    )


def scenario_bundle(fixture_name: str):
    config = load_config()
    fixture_path = Path("research_ai_supply_chain/tech_market_command_center/fixtures") / fixture_name
    provider = build_provider("fixture", fixture_path=fixture_path)
    snapshot = build_snapshot(
        as_of="2026-06-08",
        session="monday_open",
        config=config,
        provider=provider,
        include_paid_placeholder=False,
        generated_at_utc="2026-06-08T14:45:00+00:00",
    )
    scores = build_scorecard(snapshot, config)
    probs = probability_summary(snapshot, scores)
    risk = build_risk_report(snapshot, config)
    audit = build_data_audit(snapshot, config)
    gates = build_execution_gates(
        snapshot=snapshot,
        scores=scores,
        probabilities=probs,
        risk_report=risk,
        data_audit=audit,
        config=config,
    )
    breadth = build_sector_breadth(snapshot, config)
    strategy_table = build_strategy_table(snapshot, scores, config)
    candidate_ranking = build_candidate_ranking(
        snapshot=snapshot,
        scores=scores,
        strategy_table=strategy_table,
        execution_gates=gates,
        sector_breadth=breadth,
    )
    open_time_triggers = build_open_time_triggers(candidate_ranking)
    summary = build_executive_summary(
        scores=scores,
        data_audit=audit,
        probabilities=probs,
        execution_gates=gates,
        sector_breadth=breadth,
    )
    return {
        "snapshot": snapshot,
        "config": config,
        "scores": scores,
        "probabilities": probs,
        "risk": risk,
        "audit": audit,
        "gates": gates,
        "breadth": breadth,
        "strategy_table": strategy_table,
        "candidate_ranking": candidate_ranking,
        "open_time_triggers": open_time_triggers,
        "summary": summary,
    }


def test_offline_fixture_scores_required_thresholds():
    snapshot, config = fixture_snapshot()
    scores = build_scorecard(snapshot, config)
    assert snapshot.data_quality == "DATA_QUALITY_YELLOW"
    assert scores.labels["qqq"] == "stable"
    assert scores.labels["avgo"] == "danger_support_hold"
    assert scores.labels["soxl"] == "semiconductor_local_stampede"
    assert 0 <= scores.avgo_repair_probability <= 100


def test_data_quality_red_when_required_symbol_missing():
    quotes = {
        "QQQ": Quote(symbol="QQQ", price=100, previous_close=100),
        "NQ=F": Quote(symbol="NQ=F", price=30000, previous_close=30100),
        "NVDA": Quote(symbol="NVDA", price=200, previous_close=201),
        "SOXL": Quote(symbol="SOXL", price=20, previous_close=21),
    }
    assert classify_data_quality(quotes, required=["QQQ", "NQ=F", "AVGO", "NVDA", "SOXL"]) == "DATA_QUALITY_RED"


def test_provider_failure_generates_red_snapshot_and_blocks_adds():
    class EmptyProvider:
        name = "empty"

        def fetch_quotes(self, symbols):
            return {}, ["network unavailable"], [{"provider": "empty", "status": "error"}]

    config = load_config()
    snapshot = build_snapshot(
        as_of="2026-06-08",
        session="monday_open",
        config=config,
        provider=EmptyProvider(),
        include_paid_placeholder=False,
    )
    scores = build_scorecard(snapshot, config)
    rows = build_strategy_table(snapshot, scores, config)
    assert snapshot.data_quality == "DATA_QUALITY_RED"
    assert "Required market data missing; active add strategies must be blocked." in snapshot.warnings
    assert {row["symbol"]: row["decision"] for row in rows}["TQQQ"] == "DO_NOT_DO"


def test_auto_provider_chain_falls_back_without_fake_prices():
    class BrokenPaidProvider:
        name = "broken_paid"

        def fetch_quotes(self, symbols):
            return (
                {
                    symbol: Quote(
                        symbol=symbol,
                        price=None,
                        previous_close=None,
                        source=self.name,
                        fetched_at_utc="2026-06-08T13:25:00+00:00",
                        freshness="missing",
                        error="paid source unavailable",
                    )
                    for symbol in symbols
                },
                ["paid source unavailable"],
                [{"provider": self.name, "status": "error"}],
            )

    class BackupProvider:
        name = "backup"

        def fetch_quotes(self, symbols):
            return (
                {
                    symbol: Quote(
                        symbol=symbol,
                        price=100.0,
                        previous_close=95.0,
                        source=self.name,
                        fetched_at_utc="2026-06-08T13:25:00+00:00",
                        market_time_utc="2026-06-08T13:25:00+00:00",
                        freshness="test_backup",
                    )
                    for symbol in symbols
                },
                [],
                [{"provider": self.name, "status": "ok"}],
            )

    chain = ProviderChainQuoteProvider([BrokenPaidProvider(), BackupProvider()])
    quotes, warnings, status = chain.fetch_quotes(["AVGO"])
    assert quotes["AVGO"].source == "backup"
    assert quotes["AVGO"].price == 100.0
    assert warnings == ["paid source unavailable"]
    assert status[-1]["missing"] == 0


def test_scenario_probabilities_sum_to_100():
    snapshot, config = fixture_snapshot()
    scores = build_scorecard(snapshot, config)
    scenarios = build_scenarios(snapshot, scores, config)
    assert sum(row.probability for row in scenarios) == 100
    probs = probability_summary(snapshot, scores)
    assert probs["nq_5d_reclaim_31000"] <= 100


def test_historical_cases_are_reproducible_and_2025_is_top_match():
    snapshot, _ = fixture_snapshot()
    ranked = rank_historical_cases(snapshot)
    assert [case.case_id for case in ranked] == [case.case_id for case in rank_historical_cases(snapshot)]
    assert ranked[0].case_id == "avgo_2025_12_12"


def test_portfolio_shocks_are_directional():
    snapshot, config = fixture_snapshot()
    risk = build_risk_report(snapshot, config)
    shocks = {row["scenario"]: row["impact_usd"] for row in risk["shock_table"]}
    assert shocks["qqq_down_3"] < shocks["qqq_down_1"]
    assert shocks["smh_down_6"] < shocks["smh_down_2"]
    assert shocks["avgo_to_380"] < shocks["avgo_to_400"]
    assert risk["estimated_daily_decay_usd"] > 0


def test_strategy_table_blocks_active_adds_on_red_data():
    snapshot, config = fixture_snapshot()
    snapshot.data_quality = "DATA_QUALITY_RED"
    scores = build_scorecard(snapshot, config)
    rows = build_strategy_table(snapshot, scores, config)
    levered = {row["symbol"]: row["decision"] for row in rows if row["symbol"] in {"TQQQ", "SOXL", "AVGX"}}
    assert all(decision == "DO_NOT_DO" for decision in levered.values())


def test_data_audit_and_execution_gates_are_conservative():
    snapshot, config = fixture_snapshot()
    scores = build_scorecard(snapshot, config)
    probs = probability_summary(snapshot, scores)
    risk = build_risk_report(snapshot, config)
    audit = build_data_audit(snapshot, config)
    gates = build_execution_gates(
        snapshot=snapshot,
        scores=scores,
        probabilities=probs,
        risk_report=risk,
        data_audit=audit,
        config=config,
    )
    gate_map = {row["action"]: row for row in gates}
    assert audit["quality_score"] == 100
    assert audit["hard_block_adds"] is False
    assert audit["live_trade_ready"] is False
    assert gate_map["SOXL_ADD"]["status"] == "PREBUILD_ONLY"
    assert gate_map["SOXL_ADD"]["max_notional_usd"] == 0
    assert gate_map["DELEVERAGE"]["status"] == "ACTIVE"
    assert gate_map["SOXL_ADD"]["status_reason"]
    assert gate_map["SOXL_ADD"]["rule_checks"]


def test_market_hours_fixture_is_live_trade_ready():
    snapshot, config = market_hours_snapshot()
    scores = build_scorecard(snapshot, config)
    probs = probability_summary(snapshot, scores)
    risk = build_risk_report(snapshot, config)
    audit = build_data_audit(snapshot, config)
    gates = build_execution_gates(
        snapshot=snapshot,
        scores=scores,
        probabilities=probs,
        risk_report=risk,
        data_audit=audit,
        config=config,
    )
    statuses = {row["action"]: row["status"] for row in gates}
    assert audit["session_phase"] == "market_hours"
    assert audit["live_trade_ready"] is True
    assert statuses["TQQQ_ADD"] != "PREBUILD_ONLY"
    assert statuses["AVGO_LINEAR_ADD"] != "PREBUILD_ONLY"


def test_live_market_preflight_classifies_execution_permission():
    live_snapshot, live_config = market_hours_snapshot()
    live_audit = build_data_audit(live_snapshot, live_config)
    live_preflight = build_live_market_preflight(live_snapshot, live_audit, live_config)
    assert live_preflight["status"] == "PASS"
    assert live_preflight["action_permission"] == "LIVE_ACTIONS_ALLOWED"
    assert live_preflight["trusted_for_position_changes"] is True
    assert all(row["verdict"] == "PASS" for row in live_preflight["rows"] if row["importance"] == "required")

    weekend_snapshot, weekend_config = fixture_snapshot()
    weekend_audit = build_data_audit(weekend_snapshot, weekend_config)
    weekend_preflight = build_live_market_preflight(weekend_snapshot, weekend_audit, weekend_config)
    assert weekend_preflight["action_permission"] == "PREBUILD_ONLY"
    assert weekend_preflight["trusted_for_position_changes"] is False
    assert weekend_preflight["warnings"]


def test_live_market_preflight_blocks_missing_required_data():
    class EmptyProvider:
        name = "empty"

        def fetch_quotes(self, symbols):
            return {}, ["network unavailable"], [{"provider": "empty", "status": "error"}]

    config = load_config()
    snapshot = build_snapshot(
        as_of="2026-06-08",
        session="monday_open",
        config=config,
        provider=EmptyProvider(),
        include_paid_placeholder=False,
        generated_at_utc="2026-06-08T14:35:00+00:00",
    )
    audit = build_data_audit(snapshot, config)
    preflight = build_live_market_preflight(snapshot, audit, config)
    assert preflight["status"] == "BLOCK"
    assert preflight["action_permission"] == "DATA_BLOCKED"
    assert preflight["trusted_for_adds"] is False
    assert "DATA_QUALITY_RED" in preflight["blockers"][0]


def test_strong_repair_open_fixture_unlocks_core_probe_gates():
    bundle = scenario_bundle("strong_repair_open.json")
    gate_map = {row["action"]: row for row in bundle["gates"]}
    scores = bundle["scores"]

    assert bundle["audit"]["session_phase"] == "market_hours"
    assert bundle["audit"]["live_trade_ready"] is True
    assert scores.labels["qqq"] == "stable"
    assert scores.labels["avgo"] == "repair_started"
    assert gate_map["TQQQ_ADD"]["status"] == "READY_SMALL"
    assert gate_map["SOXL_ADD"]["status"] == "READY_SMALL"
    assert gate_map["AVGO_LINEAR_ADD"]["status"] == "READY_SMALL"
    assert gate_map["DELEVERAGE"]["status"] == "WATCH"
    assert bundle["summary"]["mode"] == "SELECTIVE_OFFENSE"
    assert all(gate_map[action]["max_notional_usd"] > 0 for action in ["TQQQ_ADD", "SOXL_ADD", "AVGO_LINEAR_ADD"])
    assert any(check["name"] == "avgo_price_above_418" and check["passed"] for check in gate_map["AVGO_LINEAR_ADD"]["rule_checks"])
    assert "Green because" in gate_map["AVGO_LINEAR_ADD"]["status_reason"]
    top_symbols = [row["symbol"] for row in bundle["candidate_ranking"][:5]]
    assert "AVGO" in top_symbols
    assert any(row["scenario_fit"] == "core_repair_path" for row in bundle["candidate_ranking"][:5])
    assert bundle["open_time_triggers"][0]["symbol"] == "AVGO"
    assert bundle["open_time_triggers"][0]["action_bias"] == "PROBE_ALLOWED"
    assert "small probe" in bundle["open_time_triggers"][0]["then_action"]


def test_failed_repair_open_fixture_forces_defense_and_blocks_adds():
    bundle = scenario_bundle("failed_repair_open.json")
    gate_map = {row["action"]: row for row in bundle["gates"]}
    scores = bundle["scores"]

    assert bundle["audit"]["session_phase"] == "market_hours"
    assert bundle["audit"]["live_trade_ready"] is True
    assert scores.labels["qqq"] == "full_tech_risk_off"
    assert scores.labels["avgo"] == "breakdown_watch_390_400"
    assert scores.labels["soxl"] == "semiconductor_capitulation"
    assert gate_map["DELEVERAGE"]["status"] == "ACTIVE"
    assert gate_map["TQQQ_ADD"]["status"] == "BLOCKED_RISK"
    assert gate_map["SOXL_ADD"]["status"] == "BLOCKED_RISK"
    assert gate_map["AVGO_LINEAR_ADD"]["status"] == "BLOCKED_RISK"
    assert bundle["summary"]["mode"] == "DEFENSE"
    assert all(gate_map[action]["max_notional_usd"] == 0 for action in ["TQQQ_ADD", "SOXL_ADD", "AVGO_LINEAR_ADD"])
    assert "Blocked by risk rule" in gate_map["TQQQ_ADD"]["status_reason"]
    assert any(check["name"] == "total_risk_below_75" and not check["passed"] for check in gate_map["TQQQ_ADD"]["rule_checks"])
    assert any(row["action_bias"] == "DEFENSE_FIRST" for row in bundle["candidate_ranking"][:6])
    assert bundle["open_time_triggers"][0]["action_bias"] == "DEFENSE_FIRST"
    assert "no averaging down" in bundle["open_time_triggers"][0]["then_action"]


def test_space_rotation_open_fixture_identifies_non_ai_hardware_rotation():
    bundle = scenario_bundle("space_rotation_open.json")
    gate_map = {row["action"]: row for row in bundle["gates"]}
    breadth_map = {row["group"]: row for row in bundle["breadth"]}
    scores = bundle["scores"]

    assert bundle["audit"]["live_trade_ready"] is True
    assert scores.space_rotation_strength >= 70
    assert breadth_map["space_proxy"]["status"] == "ROTATION_IN"
    assert breadth_map["space_proxy"]["relative_to_qqq_pct"] >= 2.0
    assert gate_map["SPACE_ROTATION_ADD"]["status"] == "READY_SMALL"
    assert gate_map["AVGO_LINEAR_ADD"]["status"] == "BLOCKED_RISK"
    assert gate_map["AVGO_LINEAR_ADD"]["max_notional_usd"] == 0
    assert gate_map["SPACE_ROTATION_ADD"]["max_notional_usd"] > 0
    assert any(check["name"] == "space_rotation_score" and check["passed"] for check in gate_map["SPACE_ROTATION_ADD"]["rule_checks"])
    top_symbols = [row["symbol"] for row in bundle["candidate_ranking"][:6]]
    assert any(symbol in top_symbols for symbol in {"ORBX", "RKLB", "RDW", "ASTS"})
    assert any(row["scenario_fit"] == "space_rotation_path" for row in bundle["candidate_ranking"][:6])
    assert bundle["open_time_triggers"][0]["symbol"] in {"ORBX", "RKLB", "RDW", "ASTS"}
    assert bundle["open_time_triggers"][0]["action_bias"] == "PROBE_ALLOWED"


def test_historical_research_computed_analogues_are_sorted():
    snapshot, _ = fixture_snapshot()
    histories = synthetic_histories()
    analogues = compute_analogues(snapshot, histories)
    assert analogues
    assert analogues[0]["similarity"] >= analogues[-1]["similarity"]
    stats = summarize_events(
        [
            {"forward_1d": 1, "forward_4d": 5, "forward_10d": 8, "forward_20d": 12},
            {"forward_1d": -2, "forward_4d": 1, "forward_10d": -4, "forward_20d": 6},
        ]
    )
    assert stats["sample_size"] == 2
    assert stats["forward_4d"]["positive_rate_pct"] == 100.0


def test_cli_offline_fixture_generates_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECH_COMMAND_CENTER_DISABLE_PROGRESS", "1")
    output_dir = tmp_path / "latest"
    code = main(
        [
            "run",
            "--as-of",
            "2026-06-08",
            "--session",
            "monday_open",
            "--offline-fixture",
            "--output-dir",
            str(output_dir),
            "--no-paid-placeholder",
        ]
    )
    assert code == 0
    expected = [
        "snapshot_latest.json",
        "report_latest.md",
        "strategy_table_latest.csv",
        "dashboard_latest.html",
    ]
    for name in expected:
        assert (output_dir / name).exists()
    payload = json.loads((output_dir / "snapshot_latest.json").read_text(encoding="utf-8"))
    assert payload["snapshot"]["data_quality"] == "DATA_QUALITY_YELLOW"
    assert sum(row["probability"] for row in payload["scenarios"]) == 100
    assert payload["data_audit"]["quality_score"] > 0
    assert payload["preflight"]["action_permission"] == "PREBUILD_ONLY"
    assert payload["preflight"]["trusted_for_position_changes"] is False
    assert payload["execution_gates"]
    assert payload["opening_checklist"]
    assert payload["executive_summary"]["mode"]
    assert payload["sector_breadth"]
    assert payload["candidate_ranking"]
    assert payload["open_time_triggers"]
    html = (output_dir / "dashboard_latest.html").read_text(encoding="utf-8")
    assert "Scenario Tree" in html
    assert "Strategy Table" in html
    assert "Execution Gates" in html
    assert "Live-Market Preflight" in html
    assert "Gate Rule Checks" in (output_dir / "report_latest.md").read_text(encoding="utf-8")
    assert "Live-Market Preflight" in (output_dir / "report_latest.md").read_text(encoding="utf-8")
    assert "Opening Checklist" in html
    assert "Provider Status" in html
    assert "Executive Summary" in html
    assert "Reason" in html
    assert "Sector Breadth" in html
    assert "Candidate Ranking" in html
    assert "Open-Time Triggers" in html
    assert "DATA_QUALITY_YELLOW" in html
    assert (output_dir / "execution_gates_latest.csv").exists()
    gate_csv = (output_dir / "execution_gates_latest.csv").read_text(encoding="utf-8")
    assert "status_reason" in gate_csv
    assert "rule_checks" in gate_csv
    assert (output_dir / "opening_checklist_latest.csv").exists()
    assert (output_dir / "candidate_ranking_latest.csv").exists()
    ranking_csv = (output_dir / "candidate_ranking_latest.csv").read_text(encoding="utf-8")
    assert "priority_score" in ranking_csv
    assert "action_bias" in ranking_csv
    assert (output_dir / "open_time_triggers_latest.csv").exists()
    trigger_csv = (output_dir / "open_time_triggers_latest.csv").read_text(encoding="utf-8")
    assert "if_trigger" in trigger_csv
    assert "then_action" in trigger_csv
    assert (output_dir / "preflight_latest.csv").exists()
    preflight_csv = (output_dir / "preflight_latest.csv").read_text(encoding="utf-8")
    assert "verdict" in preflight_csv
    assert "importance" in preflight_csv
    assert (output_dir / "source_status_latest.csv").exists()


def test_cli_run_blocks_actions_when_required_symbol_feed_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECH_COMMAND_CENTER_DISABLE_PROGRESS", "1")
    fixture_path = Path("research_ai_supply_chain/tech_market_command_center/fixtures/premarket_partial_outage.json")
    output_dir = tmp_path / "partial_outage"
    code = main(
        [
            "run",
            "--as-of",
            "2026-06-08",
            "--session",
            "monday_open",
            "--offline-fixture",
            "--fixture-path",
            str(fixture_path),
            "--output-dir",
            str(output_dir),
            "--no-paid-placeholder",
            "--skip-history",
            "--generated-at-utc",
            "2026-06-08T13:25:00+00:00",
        ]
    )
    assert code == 0

    payload = json.loads((output_dir / "snapshot_latest.json").read_text(encoding="utf-8"))
    decisions = {row["symbol"]: row["decision"] for row in payload["strategy_table"]}
    gates = {row["action"]: row for row in payload["execution_gates"]}

    assert payload["snapshot"]["data_quality"] == "DATA_QUALITY_RED"
    assert "AVGO: fixture row missing price or previous_close." in payload["snapshot"]["warnings"]
    assert payload["data_audit"]["missing_required"] == ["AVGO"]
    assert payload["data_audit"]["hard_block_adds"] is True
    assert payload["preflight"]["status"] == "BLOCK"
    assert payload["preflight"]["action_permission"] == "DATA_BLOCKED"
    assert payload["preflight"]["trusted_for_adds"] is False
    assert any("Missing required symbols: AVGO" == blocker for blocker in payload["preflight"]["blockers"])
    assert decisions["TQQQ"] == "DO_NOT_DO"
    assert decisions["SOXL"] == "DO_NOT_DO"
    assert gates["TQQQ_ADD"]["status"] == "BLOCKED_DATA"
    assert gates["SOXL_ADD"]["status"] == "BLOCKED_DATA"
    assert gates["AVGO_LINEAR_ADD"]["status"] == "BLOCKED_DATA"
    assert not any(row["status"] == "READY_SMALL" for row in payload["execution_gates"])
    assert any(row["provider"] == "avgo_direct_feed" and row["status"] == "error" for row in payload["source_status_summary"])
    report_md = (output_dir / "report_latest.md").read_text(encoding="utf-8")
    assert "DATA_QUALITY_RED" in report_md
    assert "DATA_BLOCKED" in report_md
    preflight_csv = (output_dir / "preflight_latest.csv").read_text(encoding="utf-8")
    assert "AVGO,required,missing,BLOCK" in preflight_csv


def test_provider_smoke_command_reports_provider_health(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECH_COMMAND_CENTER_DISABLE_PROGRESS", "1")
    fixture_dir = Path("research_ai_supply_chain/tech_market_command_center/fixtures")
    output_dir = tmp_path / "provider_smoke_ok"
    code = main(
        [
            "provider-smoke",
            "--as-of",
            "2026-06-08",
            "--provider",
            "fixture",
            "--fixture-path",
            str(fixture_dir / "market_hours_sample.json"),
            "--output-dir",
            str(output_dir),
            "--generated-at-utc",
            "2026-06-08T14:45:00+00:00",
        ]
    )
    assert code == 0
    payload = json.loads((output_dir / "provider_smoke_latest.json").read_text(encoding="utf-8"))
    assert payload["overall_status"] in {"PASS", "WARN"}
    assert payload["exit_code"] == 0
    assert payload["summary"]["preflight_permission"] == "LIVE_ACTIONS_ALLOWED"
    assert payload["summary"]["missing_required"] == []
    assert payload["summary"]["coverage_summary"]["mode"] == "FIXTURE_ONLY"
    assert payload["summary"]["coverage_summary"]["required_fixture_count"] == 5
    assert payload["summary"]["coverage_summary"]["required_total"] == 5
    assert payload["operator_recovery"]["coverage_summary"]["mode"] == "FIXTURE_ONLY"
    assert payload["operator_recovery"]["status"] == "CLEAR"
    assert payload["operator_recovery"]["lockout"] is False
    assert any(row["symbol"] == "AVGO" and row["verdict"] == "PASS" and row["source_tier"] == "FIXTURE" for row in payload["rows"])
    assert "PROVIDER SMOKE CHECK" in (output_dir / "provider_smoke_latest.txt").read_text(encoding="utf-8")
    assert "Provider Smoke Check" in (output_dir / "provider_smoke_latest.md").read_text(encoding="utf-8")
    assert "Source coverage" in (output_dir / "provider_smoke_latest.md").read_text(encoding="utf-8")
    assert "Recovery Plan" in (output_dir / "provider_smoke_latest.md").read_text(encoding="utf-8")
    assert "Recovery Plan" in (output_dir / "provider_smoke_latest.html").read_text(encoding="utf-8")
    assert (output_dir / "provider_smoke_latest.csv").exists()
    assert (output_dir / "provider_smoke_latest.html").exists()

    fail_dir = tmp_path / "provider_smoke_fail"
    fail_code = main(
        [
            "provider-smoke",
            "--as-of",
            "2026-06-08",
            "--provider",
            "fixture",
            "--fixture-path",
            str(fixture_dir / "premarket_partial_outage.json"),
            "--output-dir",
            str(fail_dir),
            "--generated-at-utc",
            "2026-06-08T13:25:00+00:00",
        ]
    )
    assert fail_code == 1
    fail_payload = json.loads((fail_dir / "provider_smoke_latest.json").read_text(encoding="utf-8"))
    assert fail_payload["overall_status"] == "FAIL"
    assert fail_payload["summary"]["missing_required"] == ["AVGO"]
    assert fail_payload["summary"]["coverage_summary"]["mode"] == "INCOMPLETE"
    assert fail_payload["summary"]["coverage_summary"]["required_fixture_count"] == 4
    assert fail_payload["summary"]["coverage_summary"]["required_missing_count"] == 1
    assert fail_payload["summary"]["coverage_summary"]["required_missing_symbols"] == ["AVGO"]
    assert fail_payload["operator_recovery"]["status"] == "LOCKED_READ_ONLY"
    assert fail_payload["operator_recovery"]["lockout"] is True
    assert "NO_ACTIVE_ADDS" in fail_payload["operator_recovery"]["prohibited_actions"]
    assert "Rerun open-pack --validate" in fail_payload["operator_recovery"]["next_steps"][-1]
    assert any(row["symbol"] == "AVGO" and row["verdict"] == "BLOCK" for row in fail_payload["rows"])
    fail_text = (fail_dir / "provider_smoke_latest.txt").read_text(encoding="utf-8")
    fail_md = (fail_dir / "provider_smoke_latest.md").read_text(encoding="utf-8")
    assert "SOURCE_COVERAGE: mode=INCOMPLETE" in fail_text
    assert "RECOVERY_STATUS: LOCKED_READ_ONLY" in fail_text
    assert "Treat the full open pack as read-only" in fail_md


def test_provider_diagnostics_reports_credentials_without_network(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECH_COMMAND_CENTER_DISABLE_PROGRESS", "1")
    for key in ["TRADIER_TOKEN", "POLYGON_API_KEY", "FINNHUB_API_KEY", "FMP_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    output_dir = tmp_path / "provider_diagnostics"
    code = main(
        [
            "provider-diagnostics",
            "--as-of",
            "2026-06-08",
            "--output-dir",
            str(output_dir),
            "--generated-at-utc",
            "2026-06-08T14:45:00+00:00",
        ]
    )
    assert code == 0
    payload = json.loads((output_dir / "provider_diagnostics_latest.json").read_text(encoding="utf-8"))
    text = (output_dir / "provider_diagnostics_latest.txt").read_text(encoding="utf-8")
    assert payload["overall_status"] == "WARN"
    assert payload["check_network"] is False
    assert payload["summary"]["configured_paid_providers"] == []
    assert set(payload["summary"]["missing_credentials"]) == {"TRADIER_TOKEN", "POLYGON_API_KEY", "FINNHUB_API_KEY", "FMP_API_KEY"}
    assert payload["summary"]["provider_chain"] == ["yahoo_chart"]
    assert payload["summary"]["yahoo_fallback_status"] == "NOT_TESTED"
    assert any(row["provider"] == "yahoo_chart" and row["network_status"] == "NOT_TESTED" for row in payload["rows"])
    assert "PROVIDER DIAGNOSTICS" in text
    assert "MISSING_CREDENTIALS: TRADIER_TOKEN, POLYGON_API_KEY, FINNHUB_API_KEY, FMP_API_KEY" in text
    assert (output_dir / "provider_diagnostics_latest.md").exists()
    assert (output_dir / "provider_diagnostics_latest.html").exists()
    assert (output_dir / "provider_diagnostics_latest.csv").exists()

    monkeypatch.setenv("FINNHUB_API_KEY", "dummy")
    configured_dir = tmp_path / "provider_diagnostics_configured"
    configured_code = main(
        [
            "provider-diagnostics",
            "--as-of",
            "2026-06-08",
            "--output-dir",
            str(configured_dir),
            "--generated-at-utc",
            "2026-06-08T14:45:00+00:00",
        ]
    )
    assert configured_code == 0
    configured = json.loads((configured_dir / "provider_diagnostics_latest.json").read_text(encoding="utf-8"))
    assert configured["overall_status"] == "PASS"
    assert configured["summary"]["configured_paid_providers"] == ["finnhub_quote"]
    assert configured["summary"]["provider_chain"] == ["finnhub_quote", "yahoo_chart"]
    assert "FINNHUB_API_KEY" not in configured["summary"]["missing_credentials"]


def test_cli_scenario_matrix_compares_generated_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECH_COMMAND_CENTER_DISABLE_PROGRESS", "1")
    fixture_dir = Path("research_ai_supply_chain/tech_market_command_center/fixtures")
    scenarios = {
        "strong_repair": "strong_repair_open.json",
        "failed_repair": "failed_repair_open.json",
        "space_rotation": "space_rotation_open.json",
        "data_outage": "premarket_partial_outage.json",
    }
    for scenario_name, fixture_name in scenarios.items():
        code = main(
            [
                "run",
                "--as-of",
                "2026-06-08",
                "--session",
                "monday_open",
                "--provider",
                "fixture",
                "--fixture-path",
                str(fixture_dir / fixture_name),
                "--generated-at-utc",
                "2026-06-08T14:45:00+00:00",
                "--output-dir",
                str(tmp_path / scenario_name),
                "--no-paid-placeholder",
                "--skip-history",
            ]
        )
        assert code == 0

    matrix_dir = tmp_path / "matrix"
    code = main(["scenario-matrix", "--input-dir", str(tmp_path), "--output-dir", str(matrix_dir)])
    assert code == 0
    matrix = json.loads((matrix_dir / "scenario_matrix_latest.json").read_text(encoding="utf-8"))
    rows = {row["scenario"]: row for row in matrix["rows"]}
    assert matrix["schema_version"] == 2
    assert matrix["matrix_preflight"]["status"] == "OK_LIVE"
    assert matrix["matrix_preflight"]["all_inputs_live_trusted"] is True
    assert matrix["matrix_preflight"]["data_quality_drill_count"] == 1
    assert matrix["matrix_preflight"]["data_quality_drill_flag_counts"] == {"BLOCK_DATA": 1}
    assert rows["data_outage"]["scenario_role"] == "data_quality_drill"
    assert rows["data_outage"]["preflight_permission"] == "DATA_BLOCKED"
    assert rows["data_outage"]["matrix_preflight_flag"] == "BLOCK_DATA"
    assert rows["data_outage"]["avgo_gate"] == "BLOCKED_DATA"
    assert rows["strong_repair"]["avgo_gate"] == "READY_SMALL"
    assert rows["strong_repair"]["preflight_permission"] == "LIVE_ACTIONS_ALLOWED"
    assert rows["strong_repair"]["matrix_preflight_flag"] == "OK_LIVE"
    assert rows["strong_repair"]["payload_age_gap_minutes"] == 0
    assert rows["failed_repair"]["delever_gate"] == "ACTIVE"
    assert rows["failed_repair"]["matrix_preflight_flag"] == "OK_LIVE"
    assert rows["space_rotation"]["space_gate"] == "READY_SMALL"
    assert rows["space_rotation"]["matrix_preflight_flag"] == "OK_LIVE"
    assert rows["strong_repair"]["dominant_blocker"] == "No core blocker; repair path is active."
    assert "DELEVERAGE" in rows["failed_repair"]["dominant_blocker"]
    assert "AVGO_LINEAR_ADD" in rows["space_rotation"]["dominant_blocker"]
    assert "SPACE_ROTATION_ADD" in rows["space_rotation"]["dominant_driver"]
    assert (matrix_dir / "scenario_matrix_latest.csv").exists()
    assert (matrix_dir / "scenario_matrix_latest.md").exists()
    assert (matrix_dir / "position_deltas_latest.json").exists()
    assert (matrix_dir / "position_deltas_latest.csv").exists()
    assert (matrix_dir / "position_deltas_latest.md").exists()
    assert (matrix_dir / "position_deltas_latest.html").exists()
    assert (matrix_dir / "risk_tape_latest.json").exists()
    assert (matrix_dir / "risk_tape_latest.csv").exists()
    assert (matrix_dir / "risk_tape_latest.md").exists()
    assert (matrix_dir / "risk_tape_latest.html").exists()
    assert (matrix_dir / "monday_open_brief_latest.md").exists()
    assert (matrix_dir / "monday_open_brief_latest.html").exists()
    deltas = json.loads((matrix_dir / "position_deltas_latest.json").read_text(encoding="utf-8"))["rows"]
    strong_tqqq = [row for row in deltas if row["scenario"] == "strong_repair" and row["symbol"] == "TQQQ"][0]
    failed_soxl = [row for row in deltas if row["scenario"] == "failed_repair" and row["symbol"] == "SOXL"][0]
    space_orbx = [row for row in deltas if row["scenario"] == "space_rotation" and row["symbol"] == "ORBX"][0]
    assert strong_tqqq["recommended_action"] == "ADD_PROBE"
    assert strong_tqqq["delta_high_usd"] > 0
    assert failed_soxl["recommended_action"] == "REDUCE_OR_HEDGE"
    assert failed_soxl["delta_low_usd"] < 0
    assert space_orbx["recommended_action"] == "ADD_PROBE"
    assert space_orbx["delta_high_usd"] > 0
    tape = json.loads((matrix_dir / "risk_tape_latest.json").read_text(encoding="utf-8"))
    assert tape["matrix_preflight"]["status"] == "OK_LIVE"
    assert tape["rows"][0]["scenario"] == "failed_repair"
    assert tape["rows"][0]["symbol"] == "SOXL"
    assert tape["rows"][0]["action"] == "REDUCE_OR_HEDGE"
    assert tape["rows"][0]["input_flag"] == "OK_LIVE"
    assert tape["rows"][0]["delta_low_usd"] < 0
    matrix_md = (matrix_dir / "scenario_matrix_latest.md").read_text(encoding="utf-8")
    assert "Main Blocker" in matrix_md
    assert "Main Driver" in matrix_md
    assert "Preflight" in matrix_md
    assert "OK_LIVE" in matrix_md
    brief_md = (matrix_dir / "monday_open_brief_latest.md").read_text(encoding="utf-8")
    assert "Monday Open Brief" in brief_md
    assert "Data Quality Summary" in brief_md
    assert "Scenario Decision Grid" in brief_md
    assert "Input Flag" in brief_md
    assert "data_quality_drill" in brief_md
    assert "BLOCK_DATA" in brief_md
    assert "### scenario_data_outage" not in brief_md
    assert "| scenario_data_outage | SOXL |" not in brief_md
    assert "Monday Open Risk Tape" in brief_md
    assert "Position Delta Grid" in brief_md
    assert "Path Playbooks" in brief_md
    assert "Hard Rules" in brief_md
    assert "failed_repair" in brief_md
    assert "space_rotation" in brief_md
    assert "strong_repair" in brief_md
    brief_html = (matrix_dir / "monday_open_brief_latest.html").read_text(encoding="utf-8")
    assert "Monday Open Brief" in brief_html
    assert "Data Quality Summary" in brief_html
    assert "Scenario Decision Grid" in brief_html
    assert "data_quality_drill" in brief_html
    assert "BLOCK_DATA" in brief_html
    assert "<h3>scenario_data_outage</h3>" not in brief_html
    assert "Monday Open Risk Tape" in brief_html
    assert "Position Delta Grid" in brief_html
    assert "REDUCE_OR_HEDGE" in brief_html
    assert "Path Playbooks" in brief_html
    assert "Print-ready" in brief_html
    matrix_html = (matrix_dir / "scenario_matrix_latest.html").read_text(encoding="utf-8")
    assert "Tech Market Scenario Matrix" in matrix_html
    assert "Preflight" in matrix_html
    assert "Main Blocker" in matrix_html


def test_cli_open_pack_refreshes_full_monday_bundle(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECH_COMMAND_CENTER_DISABLE_PROGRESS", "1")
    for key in ["TRADIER_TOKEN", "POLYGON_API_KEY", "FINNHUB_API_KEY", "FMP_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    output_root = tmp_path / "pack"
    code = main(
        [
            "open-pack",
            "--as-of",
            "2026-06-08",
            "--session",
            "monday_open",
            "--offline-fixture",
            "--output-root",
            str(output_root),
            "--no-paid-placeholder",
            "--skip-history",
            "--generated-at-utc",
            "2026-06-08T14:45:00+00:00",
            "--validate",
        ]
    )
    assert code == 0

    expected_paths = [
        output_root / "latest" / "snapshot_latest.json",
        output_root / "latest" / "report_latest.md",
        output_root / "provider_diagnostics" / "provider_diagnostics_latest.json",
        output_root / "provider_diagnostics" / "provider_diagnostics_latest.txt",
        output_root / "provider_diagnostics" / "provider_diagnostics_latest.md",
        output_root / "provider_diagnostics" / "provider_diagnostics_latest.csv",
        output_root / "provider_diagnostics" / "provider_diagnostics_latest.html",
        output_root / "provider_smoke" / "provider_smoke_latest.json",
        output_root / "provider_smoke" / "provider_smoke_latest.txt",
        output_root / "provider_smoke" / "provider_smoke_latest.md",
        output_root / "provider_smoke" / "provider_smoke_latest.csv",
        output_root / "provider_smoke" / "provider_smoke_latest.html",
        output_root / "scenario_strong_repair" / "snapshot_latest.json",
        output_root / "scenario_failed_repair" / "snapshot_latest.json",
        output_root / "scenario_space_rotation" / "snapshot_latest.json",
        output_root / "scenario_data_outage" / "snapshot_latest.json",
        output_root / "scenario_matrix" / "scenario_matrix_latest.json",
        output_root / "scenario_matrix" / "position_deltas_latest.json",
        output_root / "scenario_matrix" / "risk_tape_latest.json",
        output_root / "scenario_matrix" / "operator_checklist_latest.json",
        output_root / "scenario_matrix" / "operator_checklist_latest.csv",
        output_root / "scenario_matrix" / "operator_checklist_latest.md",
        output_root / "scenario_matrix" / "operator_checklist_latest.html",
        output_root / "scenario_matrix" / "live_readiness_latest.json",
        output_root / "scenario_matrix" / "live_readiness_latest.csv",
        output_root / "scenario_matrix" / "live_readiness_latest.md",
        output_root / "scenario_matrix" / "live_readiness_latest.html",
        output_root / "scenario_matrix" / "pre_open_runbook_latest.json",
        output_root / "scenario_matrix" / "pre_open_runbook_latest.txt",
        output_root / "scenario_matrix" / "pre_open_runbook_latest.md",
        output_root / "scenario_matrix" / "pre_open_runbook_latest.html",
        output_root / "scenario_matrix" / "pre_open_runbook_locked_drill.json",
        output_root / "scenario_matrix" / "pre_open_runbook_locked_drill.txt",
        output_root / "scenario_matrix" / "pre_open_runbook_locked_drill.md",
        output_root / "scenario_matrix" / "pre_open_runbook_locked_drill.html",
        output_root / "scenario_matrix" / "runbook_validation_latest.json",
        output_root / "scenario_matrix" / "runbook_validation_latest.txt",
        output_root / "scenario_matrix" / "runbook_validation_latest.md",
        output_root / "scenario_matrix" / "runbook_validation_latest.html",
        output_root / "scenario_matrix" / "runbook_validation_latest.csv",
        output_root / "scenario_matrix" / "operator_manifest_latest.json",
        output_root / "scenario_matrix" / "operator_manifest_latest.txt",
        output_root / "scenario_matrix" / "operator_manifest_latest.md",
        output_root / "scenario_matrix" / "operator_manifest_latest.html",
        output_root / "scenario_matrix" / "operator_manifest_validation_latest.json",
        output_root / "scenario_matrix" / "operator_manifest_validation_latest.txt",
        output_root / "scenario_matrix" / "operator_manifest_validation_latest.md",
        output_root / "scenario_matrix" / "operator_manifest_validation_latest.html",
        output_root / "scenario_matrix" / "operator_manifest_validation_latest.csv",
        output_root / "scenario_matrix" / "operator_status_latest.json",
        output_root / "scenario_matrix" / "operator_status_latest.txt",
        output_root / "scenario_matrix" / "operator_status_latest.md",
        output_root / "scenario_matrix" / "operator_status_latest.html",
        output_root / "scenario_matrix" / "monday_open_brief_latest.md",
        output_root / "scenario_matrix" / "monday_open_brief_latest.html",
    ]
    for path in expected_paths:
        assert path.exists()

    latest = json.loads((output_root / "latest" / "snapshot_latest.json").read_text(encoding="utf-8"))
    matrix = json.loads((output_root / "scenario_matrix" / "scenario_matrix_latest.json").read_text(encoding="utf-8"))
    tape = json.loads((output_root / "scenario_matrix" / "risk_tape_latest.json").read_text(encoding="utf-8"))
    checklist = json.loads((output_root / "scenario_matrix" / "operator_checklist_latest.json").read_text(encoding="utf-8"))
    readiness = json.loads((output_root / "scenario_matrix" / "live_readiness_latest.json").read_text(encoding="utf-8"))
    runbook = json.loads((output_root / "scenario_matrix" / "pre_open_runbook_latest.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "scenario_matrix" / "operator_manifest_latest.json").read_text(encoding="utf-8"))
    operator_status = json.loads((output_root / "scenario_matrix" / "operator_status_latest.json").read_text(encoding="utf-8"))
    provider_diagnostics = json.loads((output_root / "provider_diagnostics" / "provider_diagnostics_latest.json").read_text(encoding="utf-8"))
    provider_smoke = json.loads((output_root / "provider_smoke" / "provider_smoke_latest.json").read_text(encoding="utf-8"))
    scenario_names = {row["scenario"] for row in matrix["rows"]}

    assert provider_diagnostics["overall_status"] == "WARN"
    assert provider_diagnostics["exit_code"] == 0
    assert provider_diagnostics["check_network"] is False
    assert provider_diagnostics["summary"]["configured_paid_providers"] == []
    assert provider_diagnostics["summary"]["provider_chain"] == ["yahoo_chart"]
    assert provider_diagnostics["summary"]["yahoo_fallback_status"] == "NOT_TESTED"
    assert provider_smoke["overall_status"] == "PASS"
    assert provider_smoke["exit_code"] == 0
    assert provider_smoke["summary"]["preflight_permission"] == "LIVE_ACTIONS_ALLOWED"
    assert provider_smoke["summary"]["missing_required"] == []
    assert provider_smoke["summary"]["coverage_summary"]["mode"] == "FIXTURE_ONLY"
    assert provider_smoke["summary"]["coverage_summary"]["required_fixture_symbols"] == ["AVGO", "NQ=F", "NVDA", "QQQ", "SOXL"]
    assert latest["preflight"]["action_permission"] == "LIVE_ACTIONS_ALLOWED"
    assert matrix["matrix_preflight"]["status"] == "OK_LIVE"
    assert matrix["matrix_preflight"]["data_quality_drill_count"] == 1
    assert matrix["matrix_preflight"]["data_quality_drill_flag_counts"] == {"BLOCK_DATA": 1}
    assert scenario_names == {"scenario_strong_repair", "scenario_failed_repair", "scenario_space_rotation", "scenario_data_outage"}
    outage_row = next(row for row in matrix["rows"] if row["scenario"] == "scenario_data_outage")
    assert outage_row["scenario_role"] == "data_quality_drill"
    assert outage_row["preflight_permission"] == "DATA_BLOCKED"
    assert outage_row["matrix_preflight_flag"] == "BLOCK_DATA"
    assert outage_row["avgo_gate"] == "BLOCKED_DATA"
    assert tape["matrix_preflight"]["status"] == "OK_LIVE"
    assert tape["rows"][0]["scenario"] == "scenario_failed_repair"
    assert tape["rows"][0]["symbol"] == "SOXL"
    assert tape["rows"][0]["action"] == "REDUCE_OR_HEDGE"
    assert tape["rows"][0]["input_trusted"] is True
    assert checklist["active_path"] in scenario_names | {"mixed_or_unclear"}
    assert checklist["action_mode"] in {"DATA_ONLY", "DEFENSE_FIRST", "WAIT_CONFIRMATION", "SPACE_ROTATION_PROBE", "SMALL_REPAIR_PROBE"}
    assert checklist["timestamp_guard"]["status"] == "OK"
    assert checklist["timestamp_guard"]["actionable"] is True
    assert checklist["confidence"] >= 0
    assert checklist["path_similarity"]
    assert checklist["wait_for"]
    assert any(row["symbol"] == "AVGO" for row in checklist["invalidation_levels"])
    assert readiness["overall_status"] == "PASS"
    assert readiness["actionable"] is True
    assert readiness["summary"]["timestamp_guard_status"] == "OK"
    assert readiness["summary"]["required_pass_count"] == readiness["summary"]["required_count"]
    assert any(row["category"] == "required_symbol" and row["item"] == "AVGO" for row in readiness["rows"])
    assert runbook["execution_status"] == "READY"
    assert runbook["overall_status"] == "PASS"
    assert runbook["actionable"] is True
    assert runbook["warnings"] == []
    assert runbook["risk_tape_preflight"]["status"] == "OK_LIVE"
    assert runbook["operator_one_line"] == checklist["one_line"]
    assert len(runbook["top_risk_tape"]) == 3
    assert "TECH MARKET PRE-OPEN RUNBOOK" in (output_root / "scenario_matrix" / "pre_open_runbook_latest.txt").read_text(encoding="utf-8")
    assert manifest["manifest_status"] == "PASS"
    assert manifest["actionable"] is True
    assert "--validate" in manifest["command_line"]
    assert manifest["validation"]["status"] == "PASS"
    assert manifest["validation"]["latest_execution_status"] == "READY"
    assert manifest["validation"]["drill_execution_status"] == "LOCKED_READ_ONLY"
    assert manifest["provider_diagnostics"]["status"] == "WARN"
    assert manifest["provider_diagnostics"]["exit_code"] == 0
    assert manifest["provider_diagnostics"]["configured_paid_providers"] == []
    assert manifest["provider_diagnostics"]["provider_chain"] == ["yahoo_chart"]
    assert manifest["provider_diagnostics"]["yahoo_fallback_status"] == "NOT_TESTED"
    assert manifest["provider_smoke"]["status"] == "PASS"
    assert manifest["provider_smoke"]["exit_code"] == 0
    assert manifest["provider_smoke"]["missing_required"] == []
    assert manifest["provider_smoke"]["source_coverage"]["mode"] == "FIXTURE_ONLY"
    assert manifest["provider_smoke"]["source_coverage"]["required_fixture_count"] == 5
    assert manifest["provider_smoke"]["recovery_status"] == "CLEAR"
    assert manifest["source_summary"]["mode"] == "FIXTURE_DRILL"
    assert manifest["source_summary"]["uses_fixture"] is True
    assert "offline_fixture" in manifest["source_summary"]["providers_seen"]
    assert manifest["recovery"]["status"] == "CLEAR"
    assert manifest["recovery"]["primary_blocker"] == "none"
    assert manifest["recovery"]["minimum_rerun_command"].endswith("--validate")
    assert "--provider auto" in manifest["recovery"]["live_rerun_command"]
    assert "--offline-fixture" not in manifest["recovery"]["live_rerun_command"]
    assert "--smoke-fixture-path" not in manifest["recovery"]["live_rerun_command"]
    assert "--generated-at-utc" not in manifest["recovery"]["live_rerun_command"]
    assert "--no-paid-placeholder" not in manifest["recovery"]["live_rerun_command"]
    assert manifest["recovery"]["live_rerun_command"].endswith("--validate")
    assert manifest["results"]["provider_smoke"] == 0
    assert manifest["results"]["operator_manifest_validation"] == 0
    assert manifest["results"]["provider_diagnostics"] == 0
    assert manifest["summary_line"] == "SUMMARY: PASS | actionable=True | validation=PASS | diagnostics=WARN | smoke=PASS | latest=READY | drill=LOCKED_READ_ONLY | manifest_validation=PASS"
    assert "pre_open_runbook" in manifest["paths"]
    assert "provider_diagnostics" in manifest["paths"]
    assert "provider_smoke" in manifest["paths"]
    assert "SUMMARY: PASS" in (output_root / "scenario_matrix" / "operator_manifest_latest.txt").read_text(encoding="utf-8")
    assert "OPERATOR MANIFEST" in (output_root / "scenario_matrix" / "operator_manifest_latest.txt").read_text(encoding="utf-8")
    assert "RECOVERY: CLEAR" in (output_root / "scenario_matrix" / "operator_manifest_latest.txt").read_text(encoding="utf-8")
    assert "PROVIDER_DIAGNOSTICS: WARN" in (output_root / "scenario_matrix" / "operator_manifest_latest.txt").read_text(encoding="utf-8")
    assert "SOURCE_COVERAGE: mode=FIXTURE_ONLY" in (output_root / "scenario_matrix" / "operator_manifest_latest.txt").read_text(encoding="utf-8")
    assert "SOURCE_MODE: FIXTURE_DRILL" in (output_root / "scenario_matrix" / "operator_manifest_latest.txt").read_text(encoding="utf-8")
    assert "LIVE_RERUN:" in (output_root / "scenario_matrix" / "operator_manifest_latest.txt").read_text(encoding="utf-8")
    assert "## Recovery" in (output_root / "scenario_matrix" / "operator_manifest_latest.md").read_text(encoding="utf-8")
    assert operator_status["overall_status"] == "PASS"
    assert operator_status["exit_code"] == 0
    assert operator_status["actionable"] is True
    assert operator_status["source_summary"]["mode"] == "FIXTURE_DRILL"
    assert operator_status["source_coverage"]["mode"] == "FIXTURE_ONLY"
    assert operator_status["source_coverage"]["required_fixture_count"] == 5
    assert operator_status["provider_diagnostics"]["status"] == "WARN"
    assert operator_status["provider_diagnostics"]["provider_chain"] == ["yahoo_chart"]
    assert operator_status["recovery"]["primary_blocker"] == "none"
    assert operator_status["freshness_summary"]["provider_smoke_status"] == "PASS"
    assert operator_status["freshness_summary"]["oldest_required_age_minutes"] == 10.0
    assert any(
        row["symbol"] == "AVGO"
        and row["verdict"] == "PASS"
        and row["market_time_utc"] == "2026-06-08T14:35:00+00:00"
        for row in operator_status["freshness_summary"]["required_symbols"]
    )
    assert operator_status["top_risk_tape"]
    operator_status_text = (output_root / "scenario_matrix" / "operator_status_latest.txt").read_text(encoding="utf-8")
    assert "OPERATOR STATUS" in operator_status_text
    assert "SOURCE_MODE: mode=FIXTURE_DRILL" in operator_status_text
    assert "SOURCE_COVERAGE: mode=FIXTURE_ONLY" in operator_status_text
    assert "PROVIDER_DIAGNOSTICS: status=WARN" in operator_status_text
    assert "LIVE_RERUN:" in operator_status_text
    assert "FRESHNESS: provider=fixture | smoke=PASS" in operator_status_text
    assert "STALENESS: status=CURRENT" in operator_status_text
    assert "FEED_AVGO: PASS status=ok age=10.0m source=market_hours_fixture tier=FIXTURE market_time=2026-06-08T14:35:00+00:00" in operator_status_text
    assert "RISK_TAPE_1" in operator_status_text
    assert "Feed Freshness" in (output_root / "scenario_matrix" / "operator_status_latest.md").read_text(encoding="utf-8")
    assert "Feed Freshness" in (output_root / "scenario_matrix" / "operator_status_latest.html").read_text(encoding="utf-8")
    manifest_validation = json.loads((output_root / "scenario_matrix" / "operator_manifest_validation_latest.json").read_text(encoding="utf-8"))
    assert manifest_validation["overall_status"] == "PASS"
    assert manifest_validation["exit_code"] == 0
    assert any(row["section"] == "manifest" and row["item"] == "status_pass" and row["verdict"] == "PASS" for row in manifest_validation["rows"])
    assert any(row["section"] == "provider_diagnostics" and row["item"] == "status_ok" and row["verdict"] == "PASS" for row in manifest_validation["rows"])
    assert any(row["section"] == "provider_smoke" and row["item"] == "status_ok" and row["verdict"] == "PASS" for row in manifest_validation["rows"])
    assert any(row["section"] == "provider_smoke" and row["item"] == "source_coverage_present" and row["verdict"] == "PASS" for row in manifest_validation["rows"])
    assert any(row["section"] == "source" and row["item"] == "mode_present" and row["verdict"] == "PASS" for row in manifest_validation["rows"])
    assert not any(row["section"] == "source" and row["item"] == "live_source_required" for row in manifest_validation["rows"])
    assert any(row["section"] == "paths" and row["item"] == "provider_diagnostics" and row["verdict"] == "PASS" for row in manifest_validation["rows"])
    assert any(row["section"] == "paths" and row["item"] == "provider_smoke" and row["verdict"] == "PASS" for row in manifest_validation["rows"])
    manifest_code = main(["validate-manifest", "--output-root", str(output_root)])
    assert manifest_code == 0
    operator_status_code = main(["operator-status", "--output-root", str(output_root)])
    assert operator_status_code == 0
    stale_status_code = main(
        [
            "operator-status",
            "--output-root",
            str(output_root),
            "--reference-time-utc",
            "2026-06-08T15:30:00+00:00",
            "--max-age-minutes",
            "15",
        ]
    )
    assert stale_status_code == 1
    stale_operator_status = json.loads((output_root / "scenario_matrix" / "operator_status_latest.json").read_text(encoding="utf-8"))
    stale_operator_text = (output_root / "scenario_matrix" / "operator_status_latest.txt").read_text(encoding="utf-8")
    assert stale_operator_status["overall_status"] == "STALE"
    assert stale_operator_status["actionable"] is False
    assert stale_operator_status["staleness"]["status"] == "STALE"
    assert stale_operator_status["recovery"]["status"] == "LOCKED_READ_ONLY"
    assert stale_operator_status["recovery"]["primary_blocker"] == "operator_status_stale (age 45.0m > 15.0m)"
    assert "SUMMARY: STALE" in stale_operator_text
    assert "STALENESS: status=STALE" in stale_operator_text
    assert "NO_LEVERAGED_ADDS" in stale_operator_text

    code = main(["pre-open-runbook", "--output-root", str(output_root), "--limit", "2", "--locked-drill"])
    assert code == 0
    refreshed_runbook = json.loads((output_root / "scenario_matrix" / "pre_open_runbook_latest.json").read_text(encoding="utf-8"))
    assert len(refreshed_runbook["top_risk_tape"]) == 2
    locked_drill = json.loads((output_root / "scenario_matrix" / "pre_open_runbook_locked_drill.json").read_text(encoding="utf-8"))
    locked_drill_text = (output_root / "scenario_matrix" / "pre_open_runbook_locked_drill.txt").read_text(encoding="utf-8")
    assert locked_drill["drill_mode"] is True
    assert locked_drill["execution_status"] == "LOCKED_READ_ONLY"
    assert locked_drill["operator_lockout"] is True
    assert locked_drill["actionable"] is False
    assert locked_drill["timestamp_guard"]["status"] == "STALE_INPUTS_DRILL"
    assert locked_drill["readiness_summary"]["timestamp_guard_status"] == "STALE_INPUTS_DRILL"
    assert locked_drill["risk_tape_preflight"]["status"] == "DRILL_LOCKED_READ_ONLY"
    assert locked_drill["risk_tape_preflight"]["all_inputs_live_trusted"] is False
    assert "DRILL: LOCKED_READ_ONLY training artifact" in locked_drill_text
    assert "WARNING: LOCKED_READ_ONLY" in locked_drill_text
    validate_code = main(["validate-runbook", "--output-root", str(output_root)])
    assert validate_code == 0
    validation = json.loads((output_root / "scenario_matrix" / "runbook_validation_latest.json").read_text(encoding="utf-8"))
    assert validation["overall_status"] == "PASS"
    assert validation["exit_code"] == 0
    assert any(row["section"] == "latest" and row["item"] == "execution_status_ready" and row["verdict"] == "PASS" for row in validation["rows"])
    assert any(
        row["section"] == "locked_drill" and row["item"] == "execution_status_locked" and row["verdict"] == "PASS"
        for row in validation["rows"]
    )
    broken_latest = json.loads(json.dumps(refreshed_runbook))
    broken_latest["execution_status"] = "LOCKED_READ_ONLY"
    broken_validation = build_runbook_validation(broken_latest, locked_drill)
    assert broken_validation["overall_status"] == "FAIL"
    assert broken_validation["exit_code"] == 1

    stale_latest = json.loads(json.dumps(latest))
    stale_latest["snapshot"]["generated_at_utc"] = "2026-06-08T15:10:00+00:00"
    deltas = json.loads((output_root / "scenario_matrix" / "position_deltas_latest.json").read_text(encoding="utf-8"))["rows"]
    stale_checklist = build_operator_checklist(stale_latest, matrix["rows"], tape["rows"], deltas)
    assert stale_checklist["timestamp_guard"]["status"] == "STALE_INPUTS"
    assert stale_checklist["timestamp_guard"]["actionable"] is False
    assert stale_checklist["action_mode"] == "DATA_ONLY"
    assert "只读参考" in stale_checklist["immediate_action"]
    stale_readiness = build_live_readiness(stale_latest, stale_checklist)
    stale_runbook = build_pre_open_runbook(stale_readiness, stale_checklist, tape, limit=2)
    assert stale_runbook["execution_status"] == "LOCKED_READ_ONLY"
    assert stale_runbook["operator_lockout"] is True
    assert stale_runbook["actionable"] is False
    assert any("Timestamp guard STALE_INPUTS" in warning for warning in stale_runbook["warnings"])
    assert "WARNING: LOCKED_READ_ONLY" in render_terminal(stale_runbook)

    brief = (output_root / "scenario_matrix" / "monday_open_brief_latest.md").read_text(encoding="utf-8")
    assert "Operator Checklist" in brief
    assert "Feed Health" in brief
    assert "Provider smoke: `PASS`" in brief
    assert "| AVGO | PASS |" in brief
    assert "Active path" in brief
    assert "Timestamp guard" in brief
    assert "Live Readiness" in brief
    assert "Data Quality Summary" in brief
    assert "scenario_data_outage" in brief
    assert "data_quality_drill" in brief
    assert "BLOCK_DATA" in brief
    assert "Monday Open Risk Tape" in brief
    assert "scenario_failed_repair" in brief
    assert "REDUCE_OR_HEDGE" in brief
    brief_html = (output_root / "scenario_matrix" / "monday_open_brief_latest.html").read_text(encoding="utf-8")
    assert "Feed Health" in brief_html
    assert "Provider Smoke" in brief_html
    assert "AVGO" in brief_html
    assert "PASS" in brief_html


def test_cli_open_pack_fails_fast_when_provider_smoke_blocks(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECH_COMMAND_CENTER_DISABLE_PROGRESS", "1")
    fixture_dir = Path("research_ai_supply_chain/tech_market_command_center/fixtures")
    output_root = tmp_path / "pack_smoke_fail"
    code = main(
        [
            "open-pack",
            "--as-of",
            "2026-06-08",
            "--session",
            "monday_open",
            "--offline-fixture",
            "--smoke-fixture-path",
            str(fixture_dir / "premarket_partial_outage.json"),
            "--output-root",
            str(output_root),
            "--no-paid-placeholder",
            "--skip-history",
            "--generated-at-utc",
            "2026-06-08T14:45:00+00:00",
            "--validate",
        ]
    )
    assert code == 1

    latest = json.loads((output_root / "latest" / "snapshot_latest.json").read_text(encoding="utf-8"))
    provider_smoke = json.loads((output_root / "provider_smoke" / "provider_smoke_latest.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "scenario_matrix" / "operator_manifest_latest.json").read_text(encoding="utf-8"))
    manifest_validation = json.loads((output_root / "scenario_matrix" / "operator_manifest_validation_latest.json").read_text(encoding="utf-8"))
    operator_status = json.loads((output_root / "scenario_matrix" / "operator_status_latest.json").read_text(encoding="utf-8"))
    brief = (output_root / "scenario_matrix" / "monday_open_brief_latest.md").read_text(encoding="utf-8")

    assert latest["preflight"]["action_permission"] == "LIVE_ACTIONS_ALLOWED"
    assert provider_smoke["overall_status"] == "FAIL"
    assert provider_smoke["exit_code"] == 1
    assert provider_smoke["summary"]["missing_required"] == ["AVGO"]
    assert provider_smoke["operator_recovery"]["status"] == "LOCKED_READ_ONLY"
    assert provider_smoke["operator_recovery"]["lockout"] is True
    assert "NO_LEVERAGED_ADDS" in provider_smoke["operator_recovery"]["prohibited_actions"]
    assert any(row["symbol"] == "AVGO" and row["verdict"] == "BLOCK" for row in provider_smoke["rows"])
    assert manifest["open_pack_status"] == "failed"
    assert manifest["manifest_status"] == "FAIL"
    assert manifest["actionable"] is False
    assert manifest["provider_smoke"]["status"] == "FAIL"
    assert manifest["provider_smoke"]["missing_required"] == ["AVGO"]
    assert manifest["provider_smoke"]["recovery_status"] == "LOCKED_READ_ONLY"
    assert manifest["provider_smoke"]["lockout"] is True
    assert manifest["results"]["provider_smoke"] == 1
    assert manifest["results"]["operator_manifest_validation"] == 1
    assert manifest["source_summary"]["mode"] == "FIXTURE_DRILL"
    assert "--smoke-fixture-path" not in manifest["recovery"]["live_rerun_command"]
    assert "--offline-fixture" not in manifest["recovery"]["live_rerun_command"]
    assert "--generated-at-utc" not in manifest["recovery"]["live_rerun_command"]
    assert manifest["recovery"]["status"] == "LOCKED_READ_ONLY"
    assert manifest["recovery"]["primary_blocker"] == "provider_smoke_failed (missing required: AVGO)"
    assert "NO_LEVERAGED_ADDS" in manifest["recovery"]["prohibited_actions"]
    assert "Treat the full open pack as read-only" in manifest["recovery"]["next_steps"][0]
    assert "--smoke-fixture-path" not in manifest["recovery"]["minimum_rerun_command"]
    assert manifest["recovery"]["minimum_rerun_command"].endswith("--validate")
    assert "smoke=FAIL" in manifest["summary_line"]
    manifest_text = (output_root / "scenario_matrix" / "operator_manifest_latest.txt").read_text(encoding="utf-8")
    manifest_md = (output_root / "scenario_matrix" / "operator_manifest_latest.md").read_text(encoding="utf-8")
    manifest_html = (output_root / "scenario_matrix" / "operator_manifest_latest.html").read_text(encoding="utf-8")
    assert "RECOVERY: LOCKED_READ_ONLY" in manifest_text
    assert "RERUN:" in manifest_text
    assert "provider_smoke_failed (missing required: AVGO)" in manifest_md
    assert "Minimum rerun command" in manifest_html
    assert operator_status["overall_status"] == "FAIL"
    assert operator_status["exit_code"] == 1
    assert operator_status["actionable"] is False
    assert operator_status["source_summary"]["mode"] == "FIXTURE_DRILL"
    assert operator_status["recovery"]["primary_blocker"] == "provider_smoke_failed (missing required: AVGO)"
    assert "--smoke-fixture-path" not in operator_status["recovery"]["minimum_rerun_command"]
    assert operator_status["freshness_summary"]["provider_smoke_status"] == "FAIL"
    assert operator_status["freshness_summary"]["missing_required"] == ["AVGO"]
    assert any(row["symbol"] == "AVGO" and row["verdict"] == "BLOCK" for row in operator_status["freshness_summary"]["required_symbols"])
    operator_status_text = (output_root / "scenario_matrix" / "operator_status_latest.txt").read_text(encoding="utf-8")
    assert "OPERATOR STATUS" in operator_status_text
    assert "SOURCE_MODE: mode=FIXTURE_DRILL" in operator_status_text
    assert "LIVE_RERUN:" in operator_status_text
    assert "FRESHNESS: provider=fixture | smoke=FAIL" in operator_status_text
    assert "missing=AVGO" in operator_status_text
    assert "FEED_AVGO: BLOCK" in operator_status_text
    assert "RECOVERY: LOCKED_READ_ONLY" in operator_status_text
    assert "provider_smoke_failed (missing required: AVGO)" in operator_status_text
    assert manifest_validation["overall_status"] == "FAIL"
    assert manifest_validation["exit_code"] == 1
    assert any(
        row["section"] == "provider_smoke" and row["item"] == "status_ok" and row["verdict"] == "BLOCK"
        for row in manifest_validation["rows"]
    )
    assert any(
        row["section"] == "provider_smoke" and row["item"] == "required_symbols_present" and row["verdict"] == "BLOCK"
        for row in manifest_validation["rows"]
    )
    assert any(
        row["section"] == "recovery" and row["item"] == "instructions_present" and row["verdict"] == "PASS"
        for row in manifest_validation["rows"]
    )
    assert "Feed Health" in brief
    assert "Provider smoke: `FAIL`" in brief
    assert "Missing required: `AVGO`" in brief
    assert "Recovery status: `LOCKED_READ_ONLY`" in brief
    assert "Prohibited actions: `NO_ACTIVE_ADDS, NO_LEVERAGED_ADDS, NO_AVGO_REPAIR_ADD, NO_ORDER_FROM_RISK_TAPE`" in brief
    assert "Treat the full open pack as read-only" in brief
    assert "| AVGO | BLOCK | missing |" in brief
    brief_html = (output_root / "scenario_matrix" / "monday_open_brief_latest.html").read_text(encoding="utf-8")
    assert "Recovery Plan" in brief_html
    assert "LOCKED_READ_ONLY" in brief_html
    status_code = main(["operator-status", "--output-root", str(output_root)])
    assert status_code == 1


def test_cli_open_pack_requires_live_source_when_requested(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TECH_COMMAND_CENTER_DISABLE_PROGRESS", "1")
    output_root = tmp_path / "pack_require_live_source"
    code = main(
        [
            "open-pack",
            "--as-of",
            "2026-06-08",
            "--session",
            "monday_open",
            "--offline-fixture",
            "--output-root",
            str(output_root),
            "--no-paid-placeholder",
            "--skip-history",
            "--generated-at-utc",
            "2026-06-08T14:45:00+00:00",
            "--require-live-source",
        ]
    )
    assert code == 1

    manifest = json.loads((output_root / "scenario_matrix" / "operator_manifest_latest.json").read_text(encoding="utf-8"))
    validation = json.loads((output_root / "scenario_matrix" / "operator_manifest_validation_latest.json").read_text(encoding="utf-8"))
    status = json.loads((output_root / "scenario_matrix" / "operator_status_latest.json").read_text(encoding="utf-8"))
    validation_text = (output_root / "scenario_matrix" / "operator_manifest_validation_latest.txt").read_text(encoding="utf-8")

    assert manifest["source_summary"]["mode"] == "FIXTURE_DRILL"
    assert manifest["results"]["operator_manifest_validation"] == 1
    assert "--require-live-source" in manifest["command_line"]
    assert "--validate" in manifest["command_line"]
    assert "--offline-fixture" not in manifest["recovery"]["live_rerun_command"]
    assert validation["overall_status"] == "FAIL"
    assert validation["require_live_source"] is True
    assert manifest["recovery"]["primary_blocker"] == "live_source_required_but_fixture"
    assert any(
        row["section"] == "source" and row["item"] == "live_source_required" and row["verdict"] == "BLOCK"
        for row in validation["rows"]
    )
    assert "BLOCK: source.live_source_required" in validation_text
    assert status["overall_status"] == "FAIL"
    assert status["actionable"] is False


def test_operator_manifest_failure_modes(tmp_path: Path):
    matrix_dir = tmp_path / "scenario_matrix"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "pre_open_runbook_latest.json").write_text(
        json.dumps({"schema_version": 2, "execution_status": "READY", "actionable": True, "operator_lockout": False}),
        encoding="utf-8",
    )

    missing_validation = build_operator_manifest(
        tmp_path,
        "cmd --validate",
        {"latest": 0, "scenario_matrix": 0, "operator_checklist": 0, "runbook_validation": 0},
        "ok",
    )
    assert missing_validation["manifest_status"] == "FAIL"
    assert missing_validation["actionable"] is False
    assert missing_validation["validation"]["status"] == "NOT_RUN"
    assert missing_validation["locked_drill"]["exists"] is False
    assert "SUMMARY: FAIL" in missing_validation["summary_line"]

    smoke_dir = tmp_path / "provider_smoke"
    smoke_dir.mkdir()
    (smoke_dir / "provider_smoke_latest.json").write_text(
        json.dumps(
            {
                "overall_status": "FAIL",
                "exit_code": 1,
                "summary": {
                    "preflight_permission": "DATA_BLOCKED",
                    "data_quality": "DATA_QUALITY_RED",
                    "missing_required": ["AVGO"],
                    "stale_required": [],
                },
            }
        ),
        encoding="utf-8",
    )
    smoke_failed = build_operator_manifest(tmp_path, "cmd", {"latest": 0, "provider_smoke": 1}, "ok")
    assert smoke_failed["manifest_status"] == "FAIL"
    assert smoke_failed["actionable"] is False
    assert smoke_failed["provider_smoke"]["status"] == "FAIL"
    assert smoke_failed["recovery"]["status"] == "LOCKED_READ_ONLY"
    assert smoke_failed["recovery"]["next_steps"]
    smoke_validation = build_manifest_validation(smoke_failed)
    assert smoke_validation["overall_status"] == "FAIL"
    assert any(
        row["section"] == "provider_smoke" and row["item"] == "status_ok" and row["verdict"] == "BLOCK"
        for row in smoke_validation["rows"]
    )

    (matrix_dir / "pre_open_runbook_latest.json").write_text(
        json.dumps({"schema_version": 2, "execution_status": "LOCKED_READ_ONLY", "actionable": False, "operator_lockout": True}),
        encoding="utf-8",
    )
    stale_latest = build_operator_manifest(tmp_path, "cmd", {"latest": 0, "scenario_matrix": 0, "operator_checklist": 0}, "ok")
    assert stale_latest["manifest_status"] == "FAIL"
    assert stale_latest["latest_runbook"]["execution_status"] == "LOCKED_READ_ONLY"
    assert stale_latest["actionable"] is False
    stale_manifest_validation = build_manifest_validation(stale_latest)
    assert stale_manifest_validation["overall_status"] == "FAIL"
    assert stale_manifest_validation["exit_code"] == 1
    assert any(row["section"] == "manifest" and row["item"] == "status_pass" and row["verdict"] == "BLOCK" for row in stale_manifest_validation["rows"])


def test_scenario_matrix_flags_older_preflight_payloads():
    base_payload = {
        "snapshot": {"generated_at_utc": "2026-06-08T14:45:00+00:00", "quotes": {}, "data_quality": "DATA_QUALITY_YELLOW"},
        "scores": {},
        "probabilities": {},
        "execution_gates": [],
        "sector_breadth": [],
        "executive_summary": {},
        "primary_strategy": "selective_offense",
        "preflight": {
            "action_permission": "LIVE_ACTIONS_ALLOWED",
            "status": "PASS",
            "trusted_for_position_changes": True,
            "blockers": [],
            "warnings": [],
        },
    }
    old_payload = json.loads(json.dumps(base_payload))
    old_payload["snapshot"]["generated_at_utc"] = "2026-06-08T14:20:00+00:00"
    rows = {row["scenario"]: row for row in build_matrix_rows([("fresh", base_payload), ("old", old_payload)])}
    assert rows["fresh"]["matrix_preflight_flag"] == "OK_LIVE"
    assert rows["old"]["matrix_preflight_flag"] == "STALE_VS_MATRIX"
    assert rows["old"]["payload_age_gap_minutes"] == 25.0


def synthetic_histories():
    dates = [f"2026-01-{day:02d}" for day in range(1, 31)]
    out = {}
    for symbol, base in {"AVGO": 100.0, "QQQ": 100.0, "SMH": 100.0, "NVDA": 100.0}.items():
        bars = []
        value = base
        for idx, day in enumerate(dates):
            if idx == 10 and symbol == "AVGO":
                value *= 0.86
            elif idx == 10 and symbol == "QQQ":
                value *= 0.993
            elif idx == 10 and symbol == "SMH":
                value *= 0.975
            elif idx == 10 and symbol == "NVDA":
                value *= 1.004
            else:
                value *= 1.003
            bars.append({"date": day, "open": value, "high": value * 1.01, "low": value * 0.99, "close": value, "volume": 1000})
        out[symbol] = bars
    return out
