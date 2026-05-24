"""Correlation-shift network diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_shift(samples_by_event: dict[str, pd.DataFrame], prices: pd.DataFrame, assets: dict[str, str]) -> pd.DataFrame:
    returns = prices[list(dict.fromkeys([s for s in assets.values() if s in prices]))].pct_change(fill_method=None).dropna(how="all")
    rows = []
    for event_id, samples in samples_by_event.items():
        if samples.empty:
            continue
        deltas = {}
        pc_pre = []
        pc_post = []
        for _, sample in samples.iterrows():
            pos = _event_pos(returns.index, pd.Timestamp(sample["date"]))
            if pos is None or pos < 65 or pos + 25 >= len(returns):
                continue
            pre = returns.iloc[pos - 60 : pos - 20].dropna(axis=1, thresh=20)
            post = returns.iloc[pos + 1 : pos + 21].dropna(axis=1, thresh=10)
            common = pre.columns.intersection(post.columns)
            if len(common) < 3:
                continue
            c0 = pre[common].corr()
            c1 = post[common].corr()
            delta = c1 - c0
            vals = delta.where(np.triu(np.ones(delta.shape), 1).astype(bool)).stack()
            for pair, value in vals.items():
                deltas.setdefault(pair, []).append(float(value))
            pc_pre.append(_first_pc_share(pre[common]))
            pc_post.append(_first_pc_share(post[common]))
        for (a, b), vals in deltas.items():
            rows.append(
                {
                    "event_id": event_id,
                    "asset_a": a,
                    "asset_b": b,
                    "corr_delta_mean": float(np.nanmean(vals)),
                    "n": len(vals),
                    "pc1_pre_mean": float(np.nanmean(pc_pre)) if pc_pre else np.nan,
                    "pc1_post_mean": float(np.nanmean(pc_post)) if pc_post else np.nan,
                    "systemic_pc1_delta": (float(np.nanmean(pc_post)) - float(np.nanmean(pc_pre))) if pc_pre and pc_post else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _first_pc_share(frame: pd.DataFrame) -> float:
    clean = frame.dropna(axis=1, thresh=max(5, int(len(frame) * 0.7))).dropna()
    if clean.shape[1] < 2 or clean.shape[0] < 5:
        return np.nan
    x = (clean - clean.mean()) / clean.std(ddof=1).replace(0, np.nan)
    x = x.dropna(axis=1).dropna()
    if x.shape[1] < 2:
        return np.nan
    cov = np.cov(x.to_numpy(), rowvar=False)
    vals = np.linalg.eigvalsh(cov)
    return float(vals[-1] / vals.sum()) if vals.sum() > 0 else np.nan


def _event_pos(index: pd.Index, event_date: pd.Timestamp) -> int | None:
    idx = pd.DatetimeIndex(index)
    locs = np.where(idx >= event_date.normalize())[0]
    return int(locs[0]) if len(locs) else None
