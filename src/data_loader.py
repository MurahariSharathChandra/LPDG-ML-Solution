from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import TELEMETRY_METRICS


def normalize_gateway_id(value: object) -> str:
    """Return the validator-compatible bare 12-hex gateway id."""
    text = str(value).strip().upper()
    return text.replace(":", "")


def load_telemetry(data_dir: Path) -> pd.DataFrame:
    """Load the monthly Parquet telemetry partitions without copying the full table twice."""
    telemetry_dir = data_dir / "telemetry"
    if not telemetry_dir.exists():
        raise FileNotFoundError(f"Telemetry directory not found: {telemetry_dir}")

    columns = ["gateway_id", "ts_utc", *TELEMETRY_METRICS]
    frame = pd.read_parquet(telemetry_dir, columns=columns)
    frame["gateway_id"] = frame["gateway_id"].map(normalize_gateway_id)
    frame["ts"] = pd.to_datetime(frame["ts_utc"], utc=True)
    return frame.drop(columns=["ts_utc"])


def load_supporting_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the smaller supporting tables."""
    return {
        "gateway_master": pd.read_csv(data_dir / "gateway_master.csv", encoding="latin1"),
        "field_visits": pd.read_csv(data_dir / "field_visits.csv", encoding="latin1"),
        "meter_read_success": pd.read_csv(data_dir / "meter_read_success.csv"),
        "engineer_review": pd.read_excel(data_dir / "engineer_review_2026-02.xlsx"),
    }
