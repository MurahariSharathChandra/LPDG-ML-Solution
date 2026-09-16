from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd

from .config import ANOMALY_SIGMA, BASELINE_DAYS, RECENT_DAYS, TELEMETRY_METRICS


def _stats(window: pd.DataFrame) -> pd.DataFrame:
    return window.groupby("gateway_id")[TELEMETRY_METRICS].agg(["mean", "std"])


def baseline_rank_week(frame: pd.DataFrame, monday: date) -> pd.DataFrame:
    """Reference ranking matching the supplied 3-sigma idea, with deterministic ties."""
    end = pd.Timestamp(monday, tz="UTC")
    window = frame[(frame["ts"] >= end - timedelta(days=BASELINE_DAYS)) & (frame["ts"] < end)]
    if window.empty:
        return pd.DataFrame(columns=["gateway_id", "score", "reason"])

    stats = _stats(window)
    recent = window[window["ts"] >= end - timedelta(days=RECENT_DAYS)].copy()
    flags = pd.Series(0, index=recent.index, dtype=int)
    first_metric = pd.Series("", index=recent.index, dtype=object)

    for metric in TELEMETRY_METRICS:
        means = recent["gateway_id"].map(stats[(metric, "mean")])
        stds = recent["gateway_id"].map(stats[(metric, "std")]).replace(0, np.nan)
        exceeded = ((recent[metric] - means) > ANOMALY_SIGMA * stds).fillna(False)
        flags += exceeded.astype(int)
        first_metric = first_metric.mask(exceeded & (first_metric == ""), metric)

    recent["flagged"] = flags
    recent["first_metric"] = first_metric
    ranked = recent.groupby("gateway_id").agg(
        flagged_hours=("flagged", "sum"),
        first_metric=("first_metric", lambda s: next((v for v in s if v), "")),
    ).reset_index()

    ranked["score"] = ranked["flagged_hours"].astype(float)
    if ranked.empty:
        ranked["reason"] = pd.Series(dtype=str)
        return ranked

    ranked["reason"] = ranked.apply(
        lambda r: (
            f"{int(r.flagged_hours)} hour(s) beyond 3 sigma of this gateway's own "
            f"28-day baseline in the last 7 days; first breach on {r.first_metric or 'telemetry'}"
        ), axis=1,
    )
    return ranked.sort_values(["score", "gateway_id"], ascending=[False, True]).reset_index(drop=True)


def build_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for monday in __import__("src.config", fromlist=["SCORED_WEEKS"]).SCORED_WEEKS:
        ranked = baseline_rank_week(frame, monday)
        if len(ranked) < 15:
            raise ValueError(f"Only {len(ranked)} gateways have usable telemetry before {monday}")
        for rank, row in enumerate(ranked.head(15).itertuples(index=False), 1):
            rows.append({
                "week_start": monday.isoformat(),
                "rank": rank,
                "gateway_id": row.gateway_id,
                "score": float(row.score),
                "reason": row.reason[:300],
            })
    return pd.DataFrame(rows)
