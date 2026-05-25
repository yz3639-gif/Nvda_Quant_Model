from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from nvda_quant_model.event_overlay.event_classifier import classify_events
from nvda_quant_model.event_overlay.feature_builder import build_event_feature_frame
from nvda_quant_model.event_overlay.labels import build_event_labels
from nvda_quant_model.event_overlay.overlay_model import score_event_overlay, train_event_overlay_model


def _direction_metrics(y: pd.Series, prob_up: pd.Series, pred: pd.Series) -> dict[str, float]:
    y_up = (y == "bullish").astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "brier_score": float(brier_score_loss(y_up, prob_up.clip(0.0, 1.0))),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_up, prob_up)) if y_up.nunique() > 1 else float("nan")
    except ValueError:
        metrics["auc"] = float("nan")
    try:
        proba = pd.DataFrame({"bearish": 1.0 - prob_up, "bullish": prob_up, "neutral": 0.0})
        metrics["log_loss"] = float(log_loss(y, proba, labels=["bearish", "bullish", "neutral"]))
    except ValueError:
        metrics["log_loss"] = float("nan")
    return metrics


def _coverage_metrics(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    overlay: pd.DataFrame | None,
    horizon: int,
    market_data: pd.DataFrame,
    external_data: pd.DataFrame | None,
) -> dict[str, float]:
    features = build_event_feature_frame(frame, market_data, external_data)
    realized = labels[f"future_{horizon}d_return"].astype(float).reset_index(drop=True)
    base_move = features["pre_realized_vol_20d"].mask(
        features["pre_realized_vol_20d"].eq(0.0),
        features["pre_realized_vol_5d"],
    ).reset_index(drop=True)
    base_move = (base_move * np.sqrt(horizon)).clip(lower=0.005)
    multiplier = pd.Series(1.0, index=realized.index)
    tail_prob = pd.Series(0.0, index=realized.index)
    if overlay is not None and not overlay.empty:
        multiplier = overlay["event_sigma_multiplier"].reset_index(drop=True).clip(1.0, 1.30)
        tail_prob = overlay["downside_tail_probability"].reset_index(drop=True).clip(0.0, 1.0)
    expected_move = base_move * multiplier
    valid = realized.notna()
    if not valid.any():
        return {"expected_move_coverage": float("nan"), "var_95_coverage": float("nan"), "downside_tail_recall": float("nan")}
    tail_truth = labels["downside_tail_risk"].astype(int).reset_index(drop=True)
    tail_hits = tail_truth == 1
    recall = float(((tail_prob >= 0.5) & tail_hits).sum() / tail_hits.sum()) if tail_hits.sum() else float("nan")
    return {
        "expected_move_coverage": float((realized[valid].abs() <= expected_move[valid]).mean()),
        "var_95_coverage": float((realized[valid] >= -1.65 * expected_move[valid]).mean()),
        "downside_tail_recall": recall,
    }


def walk_forward_event_overlay_validation(
    events: pd.DataFrame,
    market_data: pd.DataFrame,
    external_data: pd.DataFrame | None = None,
    min_train_size: int = 120,
    test_size: int = 30,
    step: int = 30,
    target_horizon: int = 1,
    model_type: str = "logistic",
    text_method: str = "tfidf",
    random_state: int = 7,
) -> dict[str, pd.DataFrame | str]:
    """Time-based validation comparing price-only core proxy vs event overlay."""

    classified = classify_events(events).reset_index(drop=True)
    if len(classified) <= min_train_size + test_size:
        raise ValueError("Not enough events for walk-forward validation.")
    rows: list[dict[str, float | int | str]] = []
    group_rows: list[dict[str, float | int | str]] = []
    split = 0
    for start in range(min_train_size, len(classified) - test_size + 1, step):
        train = classified.iloc[:start].reset_index(drop=True)
        test = classified.iloc[start : start + test_size].reset_index(drop=True)
        labels = build_event_labels(test, market_data, horizons=(1, 3, 5))
        target = f"future_{target_horizon}d_direction"
        y = labels[target].astype(str)
        valid = y != "unknown"
        if not valid.any():
            continue

        base_model, _ = train_event_overlay_model(
            train,
            market_data,
            external_data,
            target_horizon=target_horizon,
            model_type=model_type,
            text_method="none",
            include_event_context=False,
            random_state=random_state + split,
        )
        event_model, _ = train_event_overlay_model(
            train,
            market_data,
            external_data,
            target_horizon=target_horizon,
            model_type=model_type,
            text_method=text_method,  # type: ignore[arg-type]
            include_event_context=True,
            random_state=random_state + split,
        )
        for model_name, model in [("core_only", base_model), ("core_event_overlay", event_model)]:
            overlay = score_event_overlay(model, test, market_data, external_data)
            prob_up = overlay["event_bullish_score"].reset_index(drop=True)
            pred = np.where(
                overlay["event_bullish_score"] > overlay["event_bearish_score"],
                "bullish",
                np.where(overlay["event_bearish_score"] > overlay["event_bullish_score"], "bearish", "neutral"),
            )
            metrics = _direction_metrics(y[valid].reset_index(drop=True), prob_up[valid].reset_index(drop=True), pd.Series(pred)[valid].reset_index(drop=True))
            metrics.update(
                _coverage_metrics(
                    test,
                    labels,
                    overlay if model_name == "core_event_overlay" else None,
                    target_horizon,
                    market_data,
                    external_data,
                )
            )
            rows.append({"split": split, "model": model_name, "train_size": len(train), "test_size": int(valid.sum()), **metrics})

            if model_name == "core_event_overlay":
                grouped = test.assign(
                    source_quality_bucket=pd.cut(test["source_quality"], bins=[0.0, 0.7, 0.85, 1.0], labels=["low", "medium", "high"], include_lowest=True),
                    earnings_flag=np.where(test["event_type"].eq("earnings"), "earnings", "non_earnings"),
                )
                feature_context = build_event_feature_frame(test, market_data, external_data)
                grouped["market_regime"] = np.select(
                    [
                        feature_context["market_regime_high_vol"].eq(1.0),
                        feature_context["market_regime_uptrend"].eq(1.0),
                        feature_context["market_regime_downtrend"].eq(1.0),
                    ],
                    ["high_vol", "uptrend", "downtrend"],
                    default="neutral",
                )
                for column in ["event_type", "source_quality_bucket", "earnings_flag", "market_regime"]:
                    for value, idx in grouped.groupby(column, observed=True).groups.items():
                        mask = pd.Series(grouped.index.isin(idx))
                        if mask.sum() >= 2:
                            subset_y = y[mask].reset_index(drop=True)
                            subset_p = prob_up[mask].reset_index(drop=True)
                            subset_pred = pd.Series(pred)[mask].reset_index(drop=True)
                            group_metric = _direction_metrics(subset_y, subset_p, subset_pred)
                            group_rows.append({"split": split, "group": column, "value": str(value), "count": int(mask.sum()), **group_metric})
        split += 1

    windows = pd.DataFrame(rows)
    summary = windows.groupby("model", as_index=False).mean(numeric_only=True) if not windows.empty else pd.DataFrame()
    groups = pd.DataFrame(group_rows)
    group_summary = groups.groupby(["group", "value"], as_index=False).mean(numeric_only=True) if not groups.empty else pd.DataFrame()
    return {
        "summary": summary,
        "windows": windows,
        "group_summary": group_summary,
        "principle": "Event overlay adjusts distribution, not trading decision.",
    }
