from datetime import date, datetime, timedelta, timezone
import pandas as pd

from src.ranking import baseline_rank_week


def make_frame():
    rows = []
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    gateways = [f"{i:012X}" for i in range(20)]
    for g in gateways:
        for h in range(90 * 24):
            ts = start + timedelta(hours=h)
            rows.append({
                "gateway_id": g,
                "ts": ts,
                "offline_duration_sec": 0.0,
                "disconnection_cnt": 0.0,
                "reboot_cnt": 0.0,
            })
    # Inject a clear anomaly into gateway 000...001 during the latest week.
    for row in rows:
        if row["gateway_id"] == "000000000001" and row["ts"] == datetime(2026, 1, 30, tzinfo=timezone.utc):
            row["offline_duration_sec"] = 100.0
            row["disconnection_cnt"] = 10.0
            row["reboot_cnt"] = 5.0
    return pd.DataFrame(rows)


def test_ranking_returns_expected_columns_and_anomaly():
    result = baseline_rank_week(make_frame(), date(2026, 2, 2))
    assert {"gateway_id", "score", "reason"}.issubset(result.columns)
    assert result.iloc[0].gateway_id == "000000000001"
    assert result.iloc[0].score > 0


def test_build_predictions_produces_120_rows():
    from src.ranking import build_predictions
    frame = make_frame()
    out = build_predictions(frame)
    assert len(out) == 120
    assert out.groupby("week_start").size().eq(15).all()
    assert out.groupby("week_start")["rank"].apply(lambda s: sorted(s.tolist()) == list(range(1, 16))).all()
    assert out.groupby("week_start")["gateway_id"].nunique().eq(15).all()
