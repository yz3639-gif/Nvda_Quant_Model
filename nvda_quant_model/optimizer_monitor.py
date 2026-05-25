from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nvda_quant_model.config import PROJECT_ROOT, REPO_ROOT


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    module: str
    screen_name: str
    output_dir: Path
    log_name: str
    extra_args: tuple[str, ...]

    @property
    def state_path(self) -> Path:
        filename = "long_run_state.json" if self.name == "strict" else "high_sample_state.json"
        return self.output_dir / filename

    @property
    def log_path(self) -> Path:
        return self.output_dir / self.log_name


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def optimizer_specs(args: argparse.Namespace) -> list[OptimizerSpec]:
    strict_dir = Path(args.strict_output_dir)
    high_dir = Path(args.high_sample_output_dir)
    return [
        OptimizerSpec(
            name="strict",
            module="nvda_quant_model.long_run_optimizer",
            screen_name="nvda_optimizer_repaired",
            output_dir=strict_dir,
            log_name="optimizer.log",
            extra_args=(
                "--hours",
                str(args.hours),
                "--resume",
                "--checkpoint-every",
                "25",
                "--progress-seconds",
                "300",
                "--price-override",
                str(args.price_override),
            ),
        ),
        OptimizerSpec(
            name="high_sample",
            module="nvda_quant_model.high_sample_optimizer",
            screen_name="nvda_high_sample_repaired",
            output_dir=high_dir,
            log_name="high_sample_optimizer.log",
            extra_args=(
                "--hours",
                str(args.hours),
                "--resume",
                "--checkpoint-every",
                "25",
                "--progress-seconds",
                "300",
                "--price-override",
                str(args.price_override),
                "--min-trades",
                "60",
                "--min-active-days",
                "90",
                "--target-trades",
                "90",
                "--target-active-days",
                "130",
            ),
        ),
    ]


def process_table() -> str:
    result = subprocess.run(["ps", "aux"], check=False, capture_output=True, text=True)
    return result.stdout


def _output_dir_needles(output_dir: Path) -> set[str]:
    needles = {str(output_dir)}
    try:
        needles.add(str(output_dir.relative_to(REPO_ROOT)))
    except ValueError:
        pass
    return needles


def optimizer_process_count(spec: OptimizerSpec, ps_output: str) -> int:
    needles = _output_dir_needles(spec.output_dir)
    count = 0
    for line in ps_output.splitlines():
        parts = line.split(None, 10)
        command = parts[10] if len(parts) >= 11 else line
        executable = command.split(maxsplit=1)[0] if command.strip() else ""
        is_python_process = executable.startswith("python") or executable.endswith("/python3.13")
        if is_python_process and spec.module in command and any(needle in command for needle in needles):
            count += 1
    return count


def is_optimizer_running(spec: OptimizerSpec, ps_output: str) -> bool:
    return optimizer_process_count(spec, ps_output) > 0


def start_optimizer(spec: OptimizerSpec) -> dict[str, Any]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    cmd_parts = [
        "cd",
        shlex.quote(str(REPO_ROOT)),
        "&&",
        "python3.13",
        "-u",
        "-m",
        spec.module,
        *spec.extra_args,
        "--output-dir",
        shlex.quote(str(spec.output_dir)),
        ">>",
        shlex.quote(str(spec.log_path)),
        "2>&1",
    ]
    shell_cmd = " ".join(cmd_parts)
    result = subprocess.run(
        ["screen", "-dmS", spec.screen_name, "/bin/zsh", "-lc", shell_cmd],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "name": spec.name,
        "action": "restart",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def validation_contains_label(validation_payload: dict[str, Any], label: str | None) -> bool:
    if not label:
        return False
    rows = validation_payload.get("top_rows", [])
    if not isinstance(rows, list):
        return False
    return any(isinstance(row, dict) and row.get("label") == label for row in rows)


def needs_high_sample_validation(high_state: dict[str, Any], validation_payload: dict[str, Any]) -> bool:
    label = high_state.get("best_qualified_label")
    return bool(label) and not validation_contains_label(validation_payload, str(label))


def run_high_sample_validation(args: argparse.Namespace) -> dict[str, Any]:
    validation_dir = Path(args.validation_output_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python3.13",
        "-m",
        "nvda_quant_model.high_sample_validation",
        "--high-sample-results",
        str(Path(args.high_sample_output_dir) / "high_sample_results.csv"),
        "--baseline-results",
        str(Path(args.strict_output_dir) / "long_run_results.csv"),
        "--price-override",
        str(args.price_override),
        "--output-dir",
        str(validation_dir),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=1200)
    return {
        "action": "high_sample_validation",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def summarize_state(spec: OptimizerSpec, running: bool) -> dict[str, Any]:
    state = read_json(spec.state_path)
    return {
        "name": spec.name,
        "running": running,
        "state_path": spec.state_path,
        "updated_at": state.get("updated_at"),
        "evaluated": state.get("evaluated"),
        "skipped_errors": state.get("skipped_errors"),
        "best_score": state.get("best_score"),
        "best_label": state.get("best_label"),
        "best_qualified_score": state.get("best_qualified_score"),
        "best_qualified_label": state.get("best_qualified_label"),
        "deadline_utc": state.get("deadline_utc"),
    }


def run_monitor(args: argparse.Namespace) -> dict[str, Any]:
    specs = optimizer_specs(args)
    ps_output = process_table()
    actions: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []

    for spec in specs:
        process_count = optimizer_process_count(spec, ps_output)
        running = process_count > 0
        if not running and args.restart_missing:
            action = start_optimizer(spec)
            actions.append(action)
            running = action["returncode"] == 0
            process_count = 1 if running else 0
        row = summarize_state(spec, running)
        row["process_count"] = process_count
        row["duplicate_processes"] = max(0, process_count - 1)
        optimizer_rows.append(row)

    validation_dir = Path(args.validation_output_dir)
    validation_payload = read_json(validation_dir / "high_sample_validation.json")
    high_state = read_json(Path(args.high_sample_output_dir) / "high_sample_state.json")
    if args.validate_new_high_sample and needs_high_sample_validation(high_state, validation_payload):
        actions.append(run_high_sample_validation(args))
        validation_payload = read_json(validation_dir / "high_sample_validation.json")

    payload = {
        "updated_at": _now(),
        "optimizers": optimizer_rows,
        "actions": actions,
        "validation_counts": validation_payload.get("promotion_counts", {}),
        "validation_report": validation_payload.get("report"),
        "needs_notification": bool(actions),
    }
    output_dir = Path(args.monitor_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monitor_state.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor repaired NVDA optimizers and validation flow")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--price-override", type=float, default=215.34)
    parser.add_argument("--restart-missing", action="store_true")
    parser.add_argument("--validate-new-high-sample", action="store_true")
    parser.add_argument(
        "--strict-output-dir",
        default=str(PROJECT_ROOT / "outputs" / "long_run_optimizer_repaired"),
    )
    parser.add_argument(
        "--high-sample-output-dir",
        default=str(PROJECT_ROOT / "outputs" / "high_sample_optimizer_repaired"),
    )
    parser.add_argument(
        "--validation-output-dir",
        default=str(PROJECT_ROOT / "outputs" / "high_sample_validation_repaired"),
    )
    parser.add_argument(
        "--monitor-output-dir",
        default=str(PROJECT_ROOT / "outputs" / "optimizer_monitor_repaired"),
    )
    return parser.parse_args()


def main() -> None:
    run_monitor(parse_args())


if __name__ == "__main__":
    main()
