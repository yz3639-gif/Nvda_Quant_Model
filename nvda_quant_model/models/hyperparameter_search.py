from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

from nvda_quant_model.config import PROJECT_ROOT, StrategyConfig
from nvda_quant_model.models.ml_model import SklearnDirectionReturnModel
from nvda_quant_model.pipeline import prepare_model_inputs


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj)!r} is not JSON serializable")


def _spaces(random_state: int) -> dict[str, tuple[dict[str, list[Any]], Callable[[dict[str, Any]], SklearnDirectionReturnModel]]]:
    spaces: dict[str, tuple[dict[str, list[Any]], Callable[[dict[str, Any]], SklearnDirectionReturnModel]]] = {
        "ElasticNet_Logit": (
            {
                "classifier__C": [0.15, 0.25, 0.5, 0.8, 1.2],
                "regressor__alpha": [0.0002, 0.0005, 0.001, 0.002],
                "regressor__l1_ratio": [0.25, 0.5, 0.75],
                "top_k": [8, 10, 12, 16],
            },
            lambda p: SklearnDirectionReturnModel(
                "ElasticNet_Logit",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=p["classifier__C"],
                    random_state=random_state,
                    max_iter=1500,
                ),
                ElasticNet(
                    alpha=p["regressor__alpha"],
                    l1_ratio=p["regressor__l1_ratio"],
                    random_state=random_state,
                    max_iter=6000,
                ),
                top_k=p["top_k"],
                random_state=random_state,
                scale=True,
            ),
        ),
        "RandomForest": (
            {
                "classifier__n_estimators": [160, 240, 320, 420],
                "classifier__max_depth": [3, 4, 5, 6],
                "classifier__min_samples_leaf": [6, 10, 14, 18],
                "regressor__n_estimators": [160, 240, 320, 420],
                "regressor__max_depth": [3, 4, 5, 6],
                "regressor__min_samples_leaf": [6, 10, 14, 18],
                "top_k": [8, 10, 12, 16],
            },
            lambda p: SklearnDirectionReturnModel(
                "RandomForest",
                RandomForestClassifier(
                    n_estimators=p["classifier__n_estimators"],
                    max_depth=p["classifier__max_depth"],
                    min_samples_leaf=p["classifier__min_samples_leaf"],
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                ),
                RandomForestRegressor(
                    n_estimators=p["regressor__n_estimators"],
                    max_depth=p["regressor__max_depth"],
                    min_samples_leaf=p["regressor__min_samples_leaf"],
                    random_state=random_state,
                    n_jobs=1,
                ),
                top_k=p["top_k"],
                random_state=random_state,
            ),
        ),
        "HistGradientBoosting": (
            {
                "classifier__max_iter": [80, 120, 160, 220],
                "classifier__learning_rate": [0.02, 0.035, 0.05, 0.08],
                "classifier__max_leaf_nodes": [5, 8, 12, 16],
                "classifier__l2_regularization": [0.05, 0.1, 0.2, 0.4],
                "regressor__max_iter": [80, 120, 160, 220],
                "regressor__learning_rate": [0.02, 0.035, 0.05, 0.08],
                "regressor__max_leaf_nodes": [5, 8, 12, 16],
                "regressor__l2_regularization": [0.05, 0.1, 0.2, 0.4],
                "top_k": [8, 10, 12, 16],
            },
            lambda p: SklearnDirectionReturnModel(
                "HistGradientBoosting",
                HistGradientBoostingClassifier(
                    max_iter=p["classifier__max_iter"],
                    learning_rate=p["classifier__learning_rate"],
                    max_leaf_nodes=p["classifier__max_leaf_nodes"],
                    l2_regularization=p["classifier__l2_regularization"],
                    random_state=random_state,
                ),
                HistGradientBoostingRegressor(
                    max_iter=p["regressor__max_iter"],
                    learning_rate=p["regressor__learning_rate"],
                    max_leaf_nodes=p["regressor__max_leaf_nodes"],
                    l2_regularization=p["regressor__l2_regularization"],
                    random_state=random_state,
                ),
                top_k=p["top_k"],
                random_state=random_state,
            ),
        ),
    }

    try:  # pragma: no cover - optional dependency
        from xgboost import XGBClassifier, XGBRegressor

        spaces["XGBoost"] = (
            {
                "classifier__n_estimators": [90, 140, 180, 240],
                "classifier__max_depth": [2, 3, 4],
                "classifier__learning_rate": [0.02, 0.035, 0.05, 0.08],
                "classifier__subsample": [0.70, 0.85, 1.0],
                "classifier__colsample_bytree": [0.70, 0.85, 1.0],
                "classifier__reg_lambda": [0.25, 0.5, 1.0],
                "classifier__reg_alpha": [0.0, 0.05, 0.1],
                "regressor__n_estimators": [90, 140, 180, 240],
                "regressor__max_depth": [2, 3, 4],
                "regressor__learning_rate": [0.02, 0.035, 0.05, 0.08],
                "regressor__subsample": [0.70, 0.85, 1.0],
                "regressor__colsample_bytree": [0.70, 0.85, 1.0],
                "regressor__reg_lambda": [0.25, 0.5, 1.0],
                "regressor__reg_alpha": [0.0, 0.05, 0.1],
                "top_k": [8, 10, 12, 16],
            },
            lambda p: SklearnDirectionReturnModel(
                "XGBoost",
                XGBClassifier(
                    n_estimators=p["classifier__n_estimators"],
                    max_depth=p["classifier__max_depth"],
                    learning_rate=p["classifier__learning_rate"],
                    subsample=p["classifier__subsample"],
                    colsample_bytree=p["classifier__colsample_bytree"],
                    reg_lambda=p["classifier__reg_lambda"],
                    reg_alpha=p["classifier__reg_alpha"],
                    eval_metric="logloss",
                    random_state=random_state,
                ),
                XGBRegressor(
                    n_estimators=p["regressor__n_estimators"],
                    max_depth=p["regressor__max_depth"],
                    learning_rate=p["regressor__learning_rate"],
                    subsample=p["regressor__subsample"],
                    colsample_bytree=p["regressor__colsample_bytree"],
                    reg_lambda=p["regressor__reg_lambda"],
                    reg_alpha=p["regressor__reg_alpha"],
                    random_state=random_state,
                ),
                top_k=p["top_k"],
                random_state=random_state,
            ),
        )
    except Exception:
        pass
    return spaces


def _split_params(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "classifier": {key.replace("classifier__", ""): value for key, value in params.items() if key.startswith("classifier__")},
        "regressor": {key.replace("regressor__", ""): value for key, value in params.items() if key.startswith("regressor__")},
        "top_k": params.get("top_k"),
    }


def search_hyperparameters(
    data: pd.DataFrame,
    feature_columns: list[str],
    trials: int = 20,
    splits: int = 4,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame]:
    trainable = data.dropna(subset=["target_return", "target_direction"])
    if len(trainable) < max(120, splits * 35):
        raise ValueError("Not enough rows for time-series hyperparameter search")

    rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, dict[str, Any]]] = {}
    tscv = TimeSeriesSplit(n_splits=splits)
    for model_name, (space, factory) in _spaces(random_state).items():
        sampler = ParameterSampler(space, n_iter=trials, random_state=random_state)
        best_score = -np.inf
        for trial_idx, params in enumerate(sampler, start=1):
            fold_scores = []
            fold_acc = []
            fold_log_loss = []
            fold_weight = []
            for train_idx, val_idx in tscv.split(trainable):
                train = trainable.iloc[train_idx]
                val = trainable.iloc[val_idx]
                score = factory(params).validation_score(train, val, feature_columns)
                fold_acc.append(score.accuracy)
                fold_log_loss.append(score.log_loss)
                fold_weight.append(score.weight)
                fold_scores.append(score.accuracy + 0.08 * score.weight - 0.12 * score.log_loss)
            composite = float(np.mean(fold_scores) - 0.25 * np.std(fold_scores, ddof=0))
            row = {
                "model": model_name,
                "trial": trial_idx,
                "score": composite,
                "accuracy_mean": float(np.mean(fold_acc)),
                "accuracy_std": float(np.std(fold_acc, ddof=0)),
                "log_loss_mean": float(np.mean(fold_log_loss)),
                "weight_mean": float(np.mean(fold_weight)),
                "params": json.dumps(params, ensure_ascii=False, default=_json_default),
            }
            rows.append(row)
            if composite > best_score:
                best_score = composite
                split = _split_params(params)
                best_params[model_name] = {
                    "classifier": split["classifier"],
                    "regressor": split["regressor"],
                }
                if split["top_k"] is not None:
                    best_params[model_name]["top_k"] = {"value": split["top_k"]}
    return best_params, pd.DataFrame(rows).sort_values("score", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-series hyperparameter search for NVDA ML candidate models")
    parser.add_argument("--start", default="auto")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--lookback-months", type=int, default=24)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-peer-events", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "hyperparameter_search"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = StrategyConfig(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        lookback_months=args.lookback_months,
        top_k_features=args.top_k,
        include_peer_events=not args.no_peer_events,
        random_state=args.random_state,
    )
    inputs = prepare_model_inputs(config, force_refresh=args.force_refresh)
    best_params, rows = search_hyperparameters(
        inputs.trainable,
        inputs.feature_columns,
        trials=args.trials,
        splits=args.splits,
        random_state=args.random_state,
    )
    rows.to_csv(output_dir / "hyperparameter_trials.csv", index=False)
    payload = {
        "config": asdict(inputs.config),
        "data_status": inputs.data_status,
        "model_params": best_params,
        "best_trials": rows.groupby("model").head(1).to_dict(orient="records"),
    }
    (output_dir / "best_model_params.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(payload["best_trials"], indent=2, ensure_ascii=False, default=_json_default))
    print(f"Saved: {output_dir / 'best_model_params.json'}")


if __name__ == "__main__":
    main()
