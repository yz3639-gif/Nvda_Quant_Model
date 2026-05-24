from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd


class SMARsiBaseline:
    name = "SMA_RSI_Baseline"

    def fit(self, data: pd.DataFrame, feature_columns: list[str] | None = None) -> "SMARsiBaseline":
        returns = data["target_return"].dropna()
        self.base_abs_return_ = float(returns.abs().median()) if not returns.empty else 0.01
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        score = pd.Series(0.0, index=data.index)
        if "price_20ma_ratio" in data:
            score += np.tanh((data["price_20ma_ratio"] - 1.0) * 8.0)
        if "20d_return" in data:
            score += np.tanh(data["20d_return"] * 6.0)
        if "rsi_14" in data:
            score += np.where(data["rsi_14"] < 35, 0.5, 0.0)
            score += np.where(data["rsi_14"] > 72, -0.5, 0.0)
        if "SMH_return" in data:
            score += np.tanh(data["SMH_return"] * 8.0)
        prob_up = pd.Series(1 / (1 + np.exp(-score)), index=data.index).clip(0.05, 0.95)
        expected_return = (prob_up - 0.5) * 2.0 * self.base_abs_return_
        return pd.DataFrame(
            {
                "prob_up": prob_up,
                "expected_return": expected_return,
            },
            index=data.index,
        )


@dataclass(frozen=True)
class RuleSpec:
    momentum_quantile: float
    volume_quantile: float
    filters: tuple[str, ...]

    @property
    def label(self) -> str:
        suffix = "+".join(self.filters) if self.filters else "none"
        return f"mom_q={self.momentum_quantile:.2f}|vol_q={self.volume_quantile:.2f}|filters={suffix}"


class VolumeMomentumRule:
    """Point-in-time adaptive rule model using train-window-only search.

    The rule family is intentionally small enough to avoid brute-force curve
    fitting, but it now adapts quantiles and risk filters inside each outer
    walk-forward training window. The selected rule is then refit on that
    training window and applied to the next unseen test window.
    """

    name = "Adaptive_Volume_Momentum_Rule"

    def __init__(self, quantile: float = 0.55, max_filters: int = 2):
        self.quantile = quantile
        self.max_filters = max_filters
        self.selected_spec_ = RuleSpec(quantile, quantile, tuple())

    def fit(self, data: pd.DataFrame, feature_columns: list[str] | None = None) -> "VolumeMomentumRule":
        self.selected_spec_ = self._select_spec(data)
        self.volume_threshold_ = float(data["volume_sma_ratio"].quantile(self.selected_spec_.volume_quantile))
        self.momentum_threshold_ = float(data["20d_return"].quantile(self.selected_spec_.momentum_quantile))
        self.filter_thresholds_ = self._fit_filter_thresholds(data, self.selected_spec_)
        condition = self._condition(data)
        active_returns = data.loc[condition, "target_return"].dropna()
        self.expected_active_return_ = float(active_returns.median()) if not active_returns.empty else 0.002
        self.prob_active_ = float(data.loc[condition, "target_direction"].mean()) if condition.any() else 0.55
        self.prob_active_ = float(np.clip(self.prob_active_, 0.58, 0.78))
        self.active_coverage_ = float(condition.mean()) if len(condition) else 0.0
        return self

    def _candidate_specs(self) -> list[RuleSpec]:
        quantiles = [0.55, 0.60, 0.65, 0.70]
        filters = [
            "rsi_lt_75",
            "smh_positive",
            "qqq_positive",
            "vix_not_spiking",
            "volatility_calm",
            "trend_above_60ma",
            "holiday_not_negative",
        ]
        filter_sets: list[tuple[str, ...]] = [tuple()]
        for size in range(1, self.max_filters + 1):
            filter_sets.extend(combinations(filters, size))
        return [
            RuleSpec(momentum_quantile=mq, volume_quantile=vq, filters=flt)
            for mq in quantiles
            for vq in quantiles
            for flt in filter_sets
        ]

    def _fit_filter_thresholds(self, data: pd.DataFrame, spec: RuleSpec) -> dict[str, float]:
        thresholds: dict[str, float] = {}
        if "vix_not_spiking" in spec.filters and "VIX_weekly_change" in data:
            thresholds["vix_not_spiking"] = float(data["VIX_weekly_change"].quantile(0.80))
        return thresholds

    def _filter_mask(self, data: pd.DataFrame, spec: RuleSpec, thresholds: dict[str, float] | None = None) -> pd.Series:
        thresholds = thresholds or {}
        mask = pd.Series(True, index=data.index)
        for name in spec.filters:
            if name == "rsi_lt_75" and "rsi_14" in data:
                mask &= data["rsi_14"] < 75
            elif name == "smh_positive" and "SMH_return" in data:
                mask &= data["SMH_return"] > 0
            elif name == "qqq_positive" and "QQQ_return" in data:
                mask &= data["QQQ_return"] > 0
            elif name == "vix_not_spiking" and "VIX_weekly_change" in data:
                threshold = thresholds.get(name, data["VIX_weekly_change"].quantile(0.80))
                mask &= data["VIX_weekly_change"] < threshold
            elif name == "volatility_calm" and {"volatility_20", "volatility_60"}.issubset(data.columns):
                mask &= data["volatility_20"] < data["volatility_60"] * 1.20
            elif name == "trend_above_60ma" and "price_60ma_ratio" in data:
                mask &= data["price_60ma_ratio"] > 1.0
            elif name == "holiday_not_negative" and {"pre_holiday_session", "pre_holiday_momentum"}.issubset(data.columns):
                mask &= ~((data["pre_holiday_session"] == 1) & (data["pre_holiday_momentum"] < 0))
        return mask.fillna(False)

    def _condition_for_spec(self, data: pd.DataFrame, spec: RuleSpec, thresholds: dict[str, float] | None = None) -> pd.Series:
        volume_threshold = float(data["volume_sma_ratio"].quantile(spec.volume_quantile))
        momentum_threshold = float(data["20d_return"].quantile(spec.momentum_quantile))
        return self._condition_with_thresholds(data, spec, volume_threshold, momentum_threshold, thresholds)

    def _condition_with_thresholds(
        self,
        data: pd.DataFrame,
        spec: RuleSpec,
        volume_threshold: float,
        momentum_threshold: float,
        thresholds: dict[str, float] | None = None,
    ) -> pd.Series:
        base = (data["volume_sma_ratio"] > volume_threshold) & (data["20d_return"] > momentum_threshold)
        return (base & self._filter_mask(data, spec, thresholds)).fillna(False)

    def _select_spec(self, data: pd.DataFrame) -> RuleSpec:
        if len(data) < 140:
            return RuleSpec(self.quantile, self.quantile, tuple())

        fold_size = max(42, min(63, len(data) // 4))
        fold_starts = list(range(len(data) - 3 * fold_size, len(data), fold_size))
        fold_starts = [start for start in fold_starts if start >= 84 and start + fold_size <= len(data)]
        if not fold_starts:
            return RuleSpec(self.quantile, self.quantile, tuple())

        best_spec = RuleSpec(self.quantile, self.quantile, tuple())
        best_score = -np.inf
        for spec in self._candidate_specs():
            active_returns: list[float] = []
            trade_counts = 0
            fold_scores = []
            for start in fold_starts:
                inner_train = data.iloc[:start]
                inner_val = data.iloc[start : start + fold_size]
                thresholds = self._fit_filter_thresholds(inner_train, spec)
                volume_threshold = float(inner_train["volume_sma_ratio"].quantile(spec.volume_quantile))
                momentum_threshold = float(inner_train["20d_return"].quantile(spec.momentum_quantile))
                condition = self._condition_with_thresholds(
                    inner_val,
                    spec,
                    volume_threshold,
                    momentum_threshold,
                    thresholds,
                )
                returns = inner_val.loc[condition, "target_return"].dropna()
                active_count = len(returns)
                if active_count < 3:
                    fold_scores.append(-0.25)
                    continue
                trade_counts += int((condition.astype(int).diff().fillna(condition.astype(int)) == 1).sum())
                active_returns.extend(returns.tolist())
                win_rate = float((returns > 0).mean())
                mean_return = float(returns.mean())
                volatility = float(returns.std(ddof=0))
                sharpe = mean_return / volatility * np.sqrt(252) if volatility else 0.0
                coverage = float(condition.mean())
                coverage_penalty = max(0.0, coverage - 0.22) * 0.35 + max(0.0, 0.04 - coverage) * 0.20
                fold_scores.append(sharpe + 3.0 * mean_return + 0.25 * win_rate - coverage_penalty)

            if len(active_returns) < 8:
                continue
            all_returns = pd.Series(active_returns)
            avg_score = float(np.mean(fold_scores))
            downside = abs(float(all_returns[all_returns < 0].sum()))
            upside = float(all_returns[all_returns > 0].sum())
            profit_factor = upside / downside if downside else 3.0
            stability_penalty = float(np.std(fold_scores))
            sparse_penalty = 0.20 if trade_counts < 4 else 0.0
            complexity_penalty = 0.05 * len(spec.filters)
            score = avg_score + 0.15 * min(profit_factor, 3.0) - 0.15 * stability_penalty - sparse_penalty - complexity_penalty
            if score > best_score:
                best_score = score
                best_spec = spec
        return best_spec

    def _condition(self, data: pd.DataFrame) -> pd.Series:
        return self._condition_with_thresholds(
            data,
            self.selected_spec_,
            self.volume_threshold_,
            self.momentum_threshold_,
            self.filter_thresholds_,
        )

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        condition = self._condition(data)
        prob_up = pd.Series(0.50, index=data.index)
        expected_return = pd.Series(0.0, index=data.index)
        prob_up.loc[condition] = self.prob_active_
        expected_return.loc[condition] = max(self.expected_active_return_, 0.001)
        return pd.DataFrame({"prob_up": prob_up, "expected_return": expected_return}, index=data.index)

    def validation_score(self, train: pd.DataFrame, val: pd.DataFrame, feature_columns: list[str] | None = None) -> SimpleNamespace:
        model = VolumeMomentumRule(self.quantile, self.max_filters).fit(train, feature_columns)
        pred = model.predict(val)
        active = pred["prob_up"] > 0.5
        if active.any():
            accuracy = float((val.loc[active, "target_direction"].astype(int) == 1).mean())
            coverage = float(active.mean())
            active_return = float(val.loc[active, "target_return"].mean())
        else:
            accuracy = 0.5
            coverage = 0.0
            active_return = 0.0
        weight = max(0.10, (accuracy - 0.5) * 8.0 + max(active_return, 0.0) * 30.0 + coverage)
        return SimpleNamespace(
            name=self.name,
            accuracy=accuracy,
            log_loss=1.0 - accuracy,
            weight=weight,
            selected_rule=model.selected_spec_.label,
            coverage=model.active_coverage_,
        )
