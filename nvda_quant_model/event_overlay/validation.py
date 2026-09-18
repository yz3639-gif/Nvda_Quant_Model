from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from nvda_quant_model.event_overlay.event_classifier import classify_events
from nvda_quant_model.event_overlay.feature_builder import build_event_feature_frame
from nvda_quant_model.event_overlay.labels import build_event_labels
from nvda_quant_model.event_overlay.overlay_model import score_event_overlay, train_event_overlay_model
from nvda_quant_model.data.data_validation import purge_immature_labels


def _direction_metrics(y: pd.Series, prob_up: pd.Series, pred: pd.Series, prob_down: pd.Series | None = None) -> dict[str, float]:
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
        bearish = 1.0 - prob_up if prob_down is None else prob_down
        proba = pd.DataFrame({"bearish": bearish, "bullish": prob_up, "neutral": (1.0 - prob_up - bearish).clip(lower=0)})
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
    tail_truth = labels["downside_tail_risk"].astype(float).reset_index(drop=True)
    tail_hits = tail_truth == 1
    recall = float(((tail_prob >= 0.5) & tail_hits).sum() / tail_hits.sum()) if tail_hits.sum() else float("nan")
    return {
        "expected_move_coverage": float((realized[valid].abs() <= expected_move[valid]).mean()),
        "var_95_coverage": float((realized[valid] >= -1.65 * expected_move[valid]).mean()),
        "downside_tail_recall": recall,
    }


def event_validation_splits(events: pd.DataFrame, min_train_size: int, test_size: int, step: int):
    """Yield contiguous temporal folds with no timestamp/duplicate cluster split.

    Boundaries crossing any cluster span are forbidden, including transitive
    overlaps. Sizes are minimum row counts and can expand to preserve groups.
    """
    if min(min_train_size, test_size, step) <= 0:
        raise ValueError("Fold sizes must be positive")
    classified = classify_events(events).reset_index(drop=True)
    n = len(classified)
    forbidden = set()
    for column in ("available_at", "cluster_id"):
        for _, positions in classified.groupby(column, dropna=False).groups.items():
            lo, hi = min(positions), max(positions)
            forbidden.update(range(lo + 1, hi + 1))
    boundaries = [i for i in range(n + 1) if i not in forbidden]
    cursor = min_train_size
    while cursor < n:
        starts = [i for i in boundaries if i >= cursor and i < n]
        if not starts:
            break
        start = starts[0]
        ends = [i for i in boundaries if i >= start + test_size]
        if not ends:
            break
        end = ends[0]
        yield classified.iloc[:start].copy(), classified.iloc[start:end].copy()
        cursor = max(start + step, end)


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
    assignments = []
    for train, test in event_validation_splits(classified, min_train_size, test_size, step):
        fit_at = test["available_at"].min()
        train_labels = build_event_labels(train, market_data, horizons=tuple(sorted({1, 3, 5, target_horizon})))
        train = train.join(train_labels.set_index("event_id")[["label_end_at"]], on="event_id")
        train = purge_immature_labels(train, fit_at).drop(columns=["label_end_at", "fit_at"]).reset_index(drop=True)
        test = test.reset_index(drop=True)
        if len(train) < 5:
            continue
        for role, subset in (("train", train), ("test", test)):
            for _, event in subset.iterrows():
                assignments.append({"split": split, "event_id": event["event_id"], "cluster_id": event["cluster_id"], "available_at": event["available_at"], "role": role, "fit_at": fit_at})
        labels = build_event_labels(test, market_data, horizons=tuple(sorted({1, 3, 5, target_horizon})))
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
            fit_at=fit_at,
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
            fit_at=fit_at,
        )
        for model_name, model in [("core_only", base_model), ("core_event_overlay", event_model)]:
            overlay = score_event_overlay(model, test, market_data, external_data)
            prob_up = overlay["event_bullish_score"].reset_index(drop=True)
            prob_down = overlay["event_bearish_score"].reset_index(drop=True)
            direction_probabilities = pd.DataFrame({"bearish": prob_down, "bullish": prob_up,
                "neutral": (1 - prob_up - prob_down).clip(lower=0)})
            pred = direction_probabilities.idxmax(axis=1).to_numpy()
            metrics = _direction_metrics(y[valid].reset_index(drop=True), prob_up[valid].reset_index(drop=True), pd.Series(pred)[valid].reset_index(drop=True), prob_down[valid].reset_index(drop=True))
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
            rows.append({"fit_at": str(fit_at), "split": split, "model": model_name, "train_size": len(train), "test_size": int(valid.sum()), **metrics})

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
                        mask = pd.Series(grouped.index.isin(idx)) & valid
                        if mask.sum() >= 2:
                            subset_y = y[mask].reset_index(drop=True)
                            subset_p = prob_up[mask].reset_index(drop=True)
                            subset_pred = pd.Series(pred)[mask].reset_index(drop=True)
                            group_metric = _direction_metrics(subset_y, subset_p, subset_pred, prob_down[mask].reset_index(drop=True))
                            group_rows.append({"split": split, "group": column, "value": str(value), "count": int(mask.sum()), **group_metric})
        split += 1

    windows = pd.DataFrame(rows)
    summary = windows.groupby("model", as_index=False).mean(numeric_only=True) if not windows.empty else pd.DataFrame()
    groups = pd.DataFrame(group_rows)
    group_summary = groups.groupby(["group", "value"], as_index=False).mean(numeric_only=True) if not groups.empty else pd.DataFrame()
    return {
        "fold_assignments": pd.DataFrame(assignments),
        "summary": summary,
        "windows": windows,
        "group_summary": group_summary,
        "principle": "Event overlay adjusts distribution, not trading decision.",
    }
