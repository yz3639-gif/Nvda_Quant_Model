"""Immutable offline shadow forecasts; no downloads, orders, or backdated evidence.

CLI: python -m nvda_quant_model.research.shadow {register,forecast,outcome} --help
An injected clock is allowed only in a separate simulation namespace. Source
metadata is caller-attested: this recorder verifies timing/hash consistency,
not whether a vendor or a supplied CSV is authentic market data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nvda_quant_model.data.data_validation import purge_immature_labels, validate_ohlcv
from nvda_quant_model.data.session_calendar import CALENDAR_VERSION, session_dates, session_schedule
from nvda_quant_model.research.experiments import FEATURES, FittedPredictor, research_frame
from nvda_quant_model.research.provenance import dependencies, safe_json, sha256

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ShadowConfig:
    model_id: str
    family: str
    parameter: float | int | None
    seed: int = 42
    train_sessions: int = 504
    probability_threshold: float = 0.55
    max_exposure: float = 1.0

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.model_id):
            raise ValueError("model_id must be a short filesystem-safe identifier")
        if self.family not in {"rule", "logistic", "boosting"}:
            raise ValueError("family must be rule, logistic, or boosting")
        if self.family == "rule" and self.parameter is not None:
            raise ValueError("rule parameter must be null")
        if self.family == "logistic" and (not isinstance(self.parameter, (int, float)) or
                                         not np.isfinite(self.parameter) or self.parameter <= 0):
            raise ValueError("logistic parameter must be a positive fixed C")
        if self.family == "boosting" and (type(self.parameter) is not int or not 1 <= self.parameter <= 6):
            raise ValueError("boosting parameter must be a fixed integer depth in [1,6]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(self.train_sessions) is not int or self.train_sessions < 30:
            raise ValueError("train_sessions must be an integer >=30")
        if not 0.5 <= self.probability_threshold < 1 or not 0 < self.max_exposure <= 1:
            raise ValueError("invalid probability threshold or max exposure")


def _utc(value):
    instant = pd.Timestamp(value)
    if instant.tzinfo is None or pd.isna(instant):
        raise ValueError("Timestamps must include an explicit timezone")
    return instant.tz_convert("UTC")


def _clock(now, simulation):
    if now is not None and not simulation:
        raise ValueError("Injected now requires simulation=True; backdated records are not prospective evidence")
    return _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")


def _encoded(value):
    return json.dumps(safe_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _record(value):
    return {**value, "record_sha256": _digest(value)}


def _read(path):
    record = json.loads(Path(path).read_text())
    digest = record.pop("record_sha256", None)
    if digest != _digest(record):
        raise ValueError(f"Immutable record integrity check failed: {path}")
    return {**record, "record_sha256": digest}


def _create_bytes(path, data):
    """Atomic publication with O_EXCL-like semantics; never replace a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".shadow-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic and fails if destination exists.
    finally:
        Path(temporary).unlink(missing_ok=True)


def _publish(path, record):
    try:
        _create_bytes(path, _encoded(record) + b"\n")
    except FileExistsError:
        if _read(path) != safe_json(record):
            raise FileExistsError(f"Immutable shadow record already exists with different content: {path}") from None
    return Path(path)


def _code():
    names = ["research/shadow.py", "research/experiments.py", "data/data_validation.py", "data/session_calendar.py"]
    files = {name: sha256(ROOT / "nvda_quant_model" / name) for name in names}
    return {"sha256": _digest(files), "files": files}


def _market_clock(now):
    center = now.tz_localize(None).normalize()
    schedule = session_schedule(session_dates(center - pd.Timedelta(days=20), center + pd.Timedelta(days=20)))
    closed = schedule.loc[schedule.market_close <= now]
    if closed.empty:
        raise ValueError("No latest closed XNYS session in calendar range")
    session = closed.index[-1]
    following = schedule.loc[schedule.index > session]
    return session, schedule.loc[session], following.iloc[0], following.iloc[1]


def register_model(config: ShadowConfig, store_dir: Path, *, simulation=False, now=None) -> Path:
    """Freeze one named configuration before forecasting; never select models here."""
    stamp = _clock(now, simulation)
    mode = "simulation" if simulation else "prospective"
    path = Path(store_dir) / mode / config.model_id / "registration.json"
    frozen = {"schema_version": 1, "kind": "shadow_model_registration", "mode": mode,
              "config": asdict(config), "config_sha256": _digest(asdict(config)), "code": _code(),
              "dependencies": dependencies(), "selection": "fixed_supplied_configuration_no_search"}
    if path.exists():
        old = _read(path)
        if any(old.get(key) != safe_json(value) for key, value in frozen.items()):
            raise FileExistsError("Model registration is immutable; use a different model_id for a changed specification")
        return path
    return _publish(path, _record({**frozen, "registered_at": stamp.isoformat(),
                                  "clock": "injected_simulation" if now is not None else "actual_utc_clock"}))


def _registration(path, simulation, now):
    registration = _read(path)
    mode = "simulation" if simulation else "prospective"
    if registration.get("kind") != "shadow_model_registration" or registration["mode"] != mode:
        raise ValueError("Registration mode does not match requested evidence mode")
    if _utc(registration["registered_at"]) > now:
        raise ValueError("Registration did not exist at the requested decision time")
    if registration["code"] != _code() or registration["dependencies"] != dependencies():
        raise ValueError("Registered model code/dependencies changed; register a new model_id")
    return registration, ShadowConfig(**registration["config"])


def _supplied_prices(prices_path, provenance_path, now, simulation):
    raw_bytes = Path(prices_path).read_bytes()
    input_hash = hashlib.sha256(raw_bytes).hexdigest()
    provenance_bytes = Path(provenance_path).read_bytes()
    source = json.loads(provenance_bytes)
    if source.get("prices_sha256") != input_hash:
        raise ValueError("Source provenance does not bind the supplied prices_sha256")
    if source.get("data_kind") not in {"observed_market", "synthetic"}:
        raise ValueError("Declare source data_kind as observed_market or synthetic")
    if source["data_kind"] == "synthetic" and not simulation:
        raise ValueError("Synthetic data cannot enter prospective evidence; use simulation mode")
    if not source.get("source") or not source.get("price_adjustment"):
        raise ValueError("Source and price_adjustment provenance are required")
    available = _utc(source["available_at"])
    retrieved = _utc(source["retrieved_at"])
    if not available <= retrieved <= now:
        raise ValueError("Data available_at <= retrieved_at <= now is required")
    # Parse the same bytes whose hash was checked, avoiding a second file read.
    from io import BytesIO
    prices = pd.read_csv(BytesIO(raw_bytes), index_col=0, parse_dates=True)
    prices = validate_ohlcv(prices, strict_calendar=True)
    if prices.empty:
        raise ValueError("Supplied prices are empty")
    last_close = session_schedule(prices.index[-1:]).market_close.iloc[0]
    if available < last_close:
        raise ValueError("Full daily OHLCV cannot be available before its final session close")
    return prices, source, raw_bytes, provenance_bytes, input_hash


def _blob(directory, data, suffix):
    digest = hashlib.sha256(data).hexdigest()
    path = directory / "blobs" / (digest + suffix)
    if path.exists():
        if sha256(path) != digest:
            raise ValueError("Content-addressed input blob was modified")
    else:
        try:
            _create_bytes(path, data)
        except FileExistsError:
            if sha256(path) != digest:
                raise ValueError("Content-addressed input blob conflict") from None
    return str(path.resolve())


def _check_snapshots(forecast):
    for key, identity_key in [("prices_snapshot", "prices_sha256"), ("provenance_snapshot", "provenance_sha256")]:
        if sha256(Path(forecast[key])) != forecast["identity"][identity_key]:
            raise ValueError("Immutable forecast input snapshot was modified")


def record_forecast(prices_path: Path, registration_path: Path, provenance_path: Path,
                    *, simulation=False, now=None) -> Path:
    """Fit only mature labels, predict the latest closed bar, then freeze a record.

    Prospective mode always reads the real clock. Simulation obeys the same
    session/maturity gates but is explicitly excluded from prospective evidence.
    Same session/model/inputs are idempotent; changed inputs/config cannot replace
    the original forecast, even if they would produce the same probability.
    """
    fit_at = _clock(now, simulation)
    registration, config = _registration(registration_path, simulation, fit_at)
    prices, source, raw_bytes, provenance_bytes, input_hash = _supplied_prices(prices_path, provenance_path, fit_at, simulation)
    session, closed, entry, exit_session = _market_clock(fit_at)
    if prices.index[-1] != session:
        raise ValueError(f"Stale or future last price session: expected latest closed XNYS session {session.date()}")
    if fit_at >= entry.market_open:
        raise ValueError("Intended next-session open has passed; retroactive forecasting is forbidden")
    directory = Path(registration_path).parent
    path = directory / "forecasts" / f"{session.date()}.json"
    identity = {"registration_sha256": registration["record_sha256"], "prices_sha256": input_hash,
                "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(), "session": str(session.date())}
    if path.exists():
        old = _read(path)
        if old["identity"] != identity:
            raise FileExistsError("Forecast already exists for this session/model with different data; historical records cannot be overwritten")
        _check_snapshots(old)
        return path
    frame = research_frame(prices)
    if frame.empty or frame.index[-1] != session:
        raise ValueError("Insufficient feature history for the latest session")
    train = purge_immature_labels(frame, fit_at).dropna(subset=["target_direction", "target_return"]).tail(config.train_sessions)
    if len(train) < config.train_sessions:
        raise ValueError(f"Need {config.train_sessions} mature training rows; only {len(train)} available")
    prediction_row = frame.iloc[-1:]
    with threadpool_limits(limits=1):
        model = FittedPredictor(config.family, config.parameter, config.seed).fit(train, fit_at)
        probability = float(model.predict(prediction_row)[0])
    created_at = _clock(now, simulation)
    if created_at >= entry.market_open:
        raise ValueError("Fitting finished after the intended next-session open; record rejected")
    if created_at < fit_at:
        raise ValueError("Clock moved backwards during fitting")
    weight = config.max_exposure if probability >= config.probability_threshold else 0.0
    result = {"schema_version": 1, "kind": "shadow_forecast", "mode": registration["mode"],
              "evidence_status": "simulation_not_prospective" if simulation else "prospective_shadow_pending_outcome",
              "identity": identity, "model": asdict(config), "code": registration["code"],
              "registration_path": str(Path(registration_path).resolve()),
              "available_at": source["available_at"], "fit_at": fit_at.isoformat(),
              "created_at": created_at.isoformat(), "decision_at": created_at.isoformat(),
              "session_close_at": closed.market_close.isoformat(), "fill_at": entry.market_open.isoformat(),
              "entry_session": str(entry.name.date()), "exit_session": str(exit_session.name.date()),
              "label_end_at": exit_session.market_open.isoformat(), "calendar": CALENDAR_VERSION,
              "clock": "injected_simulation" if now is not None else "actual_utc_clock",
              "prob_up": probability, "prob_down": 1-probability, "target_weight": weight,
              "decision": "long" if weight else "flat", "probability_status": "raw_uncalibrated_fixed_model",
              "prediction_target": "next_session_open_to_following_session_open",
              "training": {"n": len(train), "first_session": str(train.index.min().date()),
                           "last_session": str(train.index.max().date()),
                           "max_label_end_at": pd.to_datetime(train.label_end_at, utc=True).max().isoformat(),
                           "max_feature_available_at": pd.to_datetime(train.available_at, utc=True).max().isoformat()},
              "features": prediction_row[FEATURES].iloc[0].to_dict(),
              "transformed_features": model.preprocessor.transform(prediction_row)[FEATURES].iloc[0].to_dict(),
              "preprocessing": {"lower": model.preprocessor.lower_.to_dict(), "upper": model.preprocessor.upper_.to_dict(),
                                "medians": model.preprocessor.medians_.to_dict()},
              "data_provenance": source, "provenance_verification": "caller_attested_timing_and_hash_checked_not_vendor_authenticated",
              "dependencies": registration["dependencies"], "orders_sent": False,
              "limitations": ["Prospective recording does not establish an untouched model-selection holdout.",
                              "Daily OHLC adjustment provenance is supplied by the caller.",
                              "An intended next-open target is a hypothetical shadow decision, not an executed trade."]}
    result["prices_snapshot"] = _blob(directory, raw_bytes, ".csv")
    result["provenance_snapshot"] = _blob(directory, provenance_bytes, ".json")
    # Blob persistence may take time. Recheck immediately before publication;
    # a late attempt may leave immutable input blobs but never a late forecast.
    publication_at = _clock(now, simulation)
    if publication_at >= entry.market_open or publication_at < created_at:
        raise ValueError("Recording missed the next-session open or clock moved backwards; forecast rejected")
    result["created_at"] = result["decision_at"] = publication_at.isoformat()
    return _publish(path, _record(result))


def attach_outcome(forecast_path: Path, prices_path: Path, provenance_path: Path,
                   *, simulation=False, now=None) -> Path:
    """Attach a separate immutable mature open-to-open label; never edit forecast."""
    stamp = _clock(now, simulation)
    forecast = _read(forecast_path)
    if forecast.get("kind") != "shadow_forecast" or forecast["mode"] != ("simulation" if simulation else "prospective"):
        raise ValueError("Forecast evidence mode mismatch")
    _check_snapshots(forecast)
    if stamp < _utc(forecast["label_end_at"]):
        raise ValueError("Outcome label has not matured")
    prices, source, raw_bytes, provenance_bytes, input_hash = _supplied_prices(prices_path, provenance_path, stamp, simulation)
    if _utc(source["available_at"]) < _utc(forecast["label_end_at"]):
        raise ValueError("Outcome source predates label maturity")
    entry_date, exit_date = pd.Timestamp(forecast["entry_session"]), pd.Timestamp(forecast["exit_session"])
    if entry_date not in prices.index or exit_date not in prices.index:
        raise ValueError("Outcome data must contain the entry and following-session opens")
    opening, ending = float(prices.loc[entry_date, "Open"]), float(prices.loc[exit_date, "Open"])
    realized_return = ending / opening - 1
    label = int(realized_return > 0)
    directory = Path(forecast_path).parent.parent
    path = directory / "outcomes" / Path(forecast_path).name
    identity = {"forecast_sha256": forecast["record_sha256"], "prices_sha256": input_hash,
                "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest()}
    if path.exists():
        old = _read(path)
        if old["identity"] != identity:
            raise FileExistsError("Outcome is immutable; changed source data requires a separately documented correction")
        return path
    result = {"schema_version": 1, "kind": "shadow_outcome", "mode": forecast["mode"],
              "evidence_status": "simulation_not_prospective" if simulation else "mature_prospective_shadow_label",
              "identity": identity, "forecast_path": str(Path(forecast_path).resolve()), "created_at": stamp.isoformat(),
              "available_at": source["available_at"], "label_end_at": forecast["label_end_at"],
              "entry_open": opening, "exit_open": ending, "target_return": realized_return, "target_direction": label,
              "prob_up": forecast["prob_up"], "brier_score": (forecast["prob_up"]-label)**2,
              "log_loss": -float(np.log(forecast["prob_up"] if label else 1-forecast["prob_up"])),
              "hypothetical_gross_strategy_return": realized_return * forecast["target_weight"],
              "return_status": "gross_shadow_label_not_realized_trading_pnl_no_costs",
              "data_provenance": source, "orders_sent": False,
              "prices_snapshot": _blob(directory, raw_bytes, ".csv"),
              "provenance_snapshot": _blob(directory, provenance_bytes, ".json")}
    return _publish(path, _record(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="Freeze a supplied model config JSON")
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--store", type=Path, required=True)
    register.add_argument("--simulation", action="store_true")
    forecast = commands.add_parser("forecast", help="Record a timely shadow forecast from supplied daily OHLCV")
    forecast.add_argument("--registration", type=Path, required=True)
    outcome = commands.add_parser("outcome", help="Attach a mature open-to-open outcome")
    outcome.add_argument("--forecast", type=Path, required=True)
    for command in [forecast, outcome]:
        command.add_argument("--prices", type=Path, required=True)
        command.add_argument("--provenance", type=Path, required=True)
        command.add_argument("--simulation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "register":
        path = register_model(ShadowConfig(**json.loads(args.config.read_text())), args.store, simulation=args.simulation)
    elif args.command == "forecast":
        path = record_forecast(args.prices, args.registration, args.provenance, simulation=args.simulation)
    else:
        path = attach_outcome(args.forecast, args.prices, args.provenance, simulation=args.simulation)
    print(path)


if __name__ == "__main__":
    main()
