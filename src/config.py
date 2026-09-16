from __future__ import annotations

from datetime import date, timedelta

SCORED_WEEKS = [date(2026, 2, 2) + timedelta(days=7 * i) for i in range(8)]
VISITS_PER_WEEK = 15
BASELINE_DAYS = 28
RECENT_DAYS = 7
ANOMALY_SIGMA = 3.0

TELEMETRY_METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]
