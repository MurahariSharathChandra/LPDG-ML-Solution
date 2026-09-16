"""Historical pre-visit evidence analysis for the LPDG gateway challenge."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
ENCODING = "cp1252"

OUT_FIXED = "Fehler behoben"
OUT_NO_ERROR = "Kein Fehler gefunden"
OUT_NO_ACCESS = "Kein Zugang"


def normalize_gateway_id(value):
    """Normalize MAC/gateway IDs for matching across source files."""
    if pd.isna(value):
        return np.nan
    return str(value).strip().upper().replace(":", "").replace("-", "").replace(" ", "")


def load_telemetry():
    files = sorted((DATA / "telemetry").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No telemetry Parquet files found under data/telemetry")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "gateway_id" not in df.columns or "DateDt" not in df.columns:
        raise KeyError("Telemetry must contain gateway_id and DateDt")
    df["gateway_key"] = df["gateway_id"].map(normalize_gateway_id)
    df["DateDt"] = pd.to_datetime(df["DateDt"], errors="coerce").dt.normalize()
    return df.dropna(subset=["gateway_key", "DateDt"]).copy()


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_telemetry_features(t):
    wanted = [
        "rx_nr_pkts", "rx_crc_bad", "tx_success", "tx_busy", "tx_override",
        "number_of_messages", "reboot_cnt", "reboot_duration_sec",
        "r_cnt_power_cycle", "offline_duration_sec", "disconnection_cnt",
        "online_duration_mins", "rssi_good", "rssi_normal", "rssi_bad",
        "rscp_rsrp_good", "rscp_rsrp_normal", "rscp_rsrp_bad",
        "ecio_rsrq_good", "ecio_rsrq_normal", "ecio_rsrq_bad",
    ]
    t = safe_numeric(t, wanted)

    if {"rx_nr_pkts", "rx_crc_bad"}.issubset(t.columns):
        t["crc_bad_rate"] = np.where(
            t["rx_nr_pkts"] > 0, t["rx_crc_bad"] / t["rx_nr_pkts"], np.nan
        )
    if {"tx_success", "tx_busy", "tx_override"}.issubset(t.columns):
        total = t["tx_success"] + t["tx_busy"] + t["tx_override"]
        t["tx_busy_rate"] = np.where(total > 0, t["tx_busy"] / total, np.nan)
    for good, normal, bad, out in [
        ("rssi_good", "rssi_normal", "rssi_bad", "rssi_bad_share"),
        ("rscp_rsrp_good", "rscp_rsrp_normal", "rscp_rsrp_bad", "rsrp_bad_share"),
        ("ecio_rsrq_good", "ecio_rsrq_normal", "ecio_rsrq_bad", "ecio_bad_share"),
    ]:
        if {good, normal, bad}.issubset(t.columns):
            denom = t[good] + t[normal] + t[bad]
            t[out] = np.where(denom > 0, t[bad] / denom, np.nan)
    return t


def aggregate_previsit(t, visits):
    """Build 7-day and 28-day features strictly before each visit date."""
    numeric = [
        "rx_nr_pkts", "rx_crc_bad", "crc_bad_rate", "tx_success", "tx_busy",
        "tx_override", "tx_busy_rate", "number_of_messages", "reboot_cnt",
        "reboot_duration_sec", "r_cnt_power_cycle", "offline_duration_sec",
        "disconnection_cnt", "online_duration_mins", "rssi_bad_share",
        "rsrp_bad_share", "ecio_bad_share",
    ]
    numeric = [c for c in numeric if c in t.columns]
    daily = (
        t.groupby(["gateway_key", "DateDt"], as_index=False)[numeric]
        .mean()
        .sort_values(["gateway_key", "DateDt"])
        .reset_index(drop=True)
    )

    required = ["visit_id", "gateway_id", "requested_on", "outcome"]
    missing = [c for c in required if c not in visits.columns]
    if missing:
        raise KeyError(f"Field visits missing columns: {missing}")

    v = visits.copy()
    if "reason_reported" not in v.columns:
        v["reason_reported"] = np.nan
    v["requested_on"] = pd.to_datetime(v["requested_on"], errors="coerce")
    v["gateway_key"] = v["gateway_id"].map(normalize_gateway_id)
    v = v.dropna(subset=["requested_on", "gateway_key"]).copy()
    v["cutoff_date"] = v["requested_on"].dt.normalize()

    sum_cols = {
        "reboot_cnt", "reboot_duration_sec", "r_cnt_power_cycle",
        "offline_duration_sec", "disconnection_cnt", "number_of_messages",
    }
    rows = []

    for r in v[["visit_id", "gateway_id", "gateway_key", "requested_on", "cutoff_date", "outcome", "reason_reported"]].itertuples(index=False):
        g = daily[(daily.gateway_key == r.gateway_key) & (daily.DateDt < r.cutoff_date)]
        row = {
            "visit_id": r.visit_id,
            "gateway_id": r.gateway_id,
            "requested_on": r.requested_on,
            "outcome": r.outcome,
            "reason_reported": r.reason_reported,
            "cutoff_date": r.cutoff_date,
            "telemetry_days_7d": 0,
            "telemetry_days_28d": 0,
            "last_telemetry_date": pd.NaT,
        }
        if g.empty:
            rows.append(row)
            continue

        g7 = g[(g.DateDt >= r.cutoff_date - pd.Timedelta(days=7)) & (g.DateDt < r.cutoff_date)]
        g28 = g[(g.DateDt >= r.cutoff_date - pd.Timedelta(days=28)) & (g.DateDt < r.cutoff_date)]
        row["telemetry_days_7d"] = len(g7)
        row["telemetry_days_28d"] = len(g28)
        row["last_telemetry_date"] = g.DateDt.max()

        for c in numeric:
            row[f"{c}_r7d"] = g7[c].mean() if not g7.empty else np.nan
            row[f"{c}_r28d"] = g28[c].mean() if not g28.empty else np.nan
            if c in sum_cols:
                row[f"{c}_r7d_sum"] = g7[c].sum(min_count=1) if not g7.empty else np.nan
                row[f"{c}_r28d_sum"] = g28[c].sum(min_count=1) if not g28.empty else np.nan
        rows.append(row)

    features = pd.DataFrame(rows)
    features["DateDt_match"] = features["telemetry_days_28d"] > 0
    return features


def add_meter_features(features, meter):
    required = ["gateway_id", "week_start", "meters_expected", "meters_read"]
    missing = [c for c in required if c not in meter.columns]
    if missing:
        raise KeyError(f"Meter-read file missing columns: {missing}")

    m = meter.copy()
    m["gateway_key"] = m["gateway_id"].map(normalize_gateway_id)
    m["week_start"] = pd.to_datetime(m["week_start"], errors="coerce").dt.normalize()
    m["meters_expected"] = pd.to_numeric(m["meters_expected"], errors="coerce")
    m["meters_read"] = pd.to_numeric(m["meters_read"], errors="coerce")
    m["read_rate"] = np.where(m["meters_expected"] > 0, m["meters_read"] / m["meters_expected"], np.nan)
    m = m.dropna(subset=["gateway_key", "week_start"]).sort_values(["week_start", "gateway_key"]).reset_index(drop=True)

    out = features.copy()
    out["gateway_key"] = out["gateway_id"].map(normalize_gateway_id)
    out["requested_on"] = pd.to_datetime(out["requested_on"], errors="coerce")
    out["requested_week"] = out["requested_on"].dt.normalize() - pd.to_timedelta(out["requested_on"].dt.weekday, unit="D")
    out = out.sort_values(["requested_week", "gateway_key"]).reset_index(drop=True)

    right = m[["gateway_key", "week_start", "meters_expected", "meters_read", "read_rate"]]
    out = pd.merge_asof(
        out, right, left_on="requested_week", right_on="week_start", by="gateway_key",
        direction="backward", allow_exact_matches=False,
    )

    hist = m[["gateway_key", "week_start", "read_rate"]].copy()
    hist["read_rate_4w"] = hist.groupby("gateway_key", sort=False)["read_rate"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    hist = hist[["gateway_key", "week_start", "read_rate_4w"]]
    out = out.sort_values(["requested_week", "gateway_key"]).reset_index(drop=True)
    out = pd.merge_asof(
        out, hist.sort_values(["week_start", "gateway_key"]),
        left_on="requested_week", right_on="week_start", by="gateway_key",
        direction="backward", allow_exact_matches=False,
    )
    out = out.sort_values("visit_id").reset_index(drop=True)
    return out.drop(columns=["gateway_key"], errors="ignore")


def write_report(features, telemetry, visits):
    usable = features[features.outcome.isin([OUT_FIXED, OUT_NO_ERROR])].copy()
    matched = int(features["DateDt_match"].sum())
    no_match = int((~features["DateDt_match"]).sum())
    lines = [
        "# Historical Visit Evidence Analysis", "",
        "Exploratory analysis only. `reason_reported` is not used as a predictive feature.", "",
        "## Coverage", "",
        f"- Telemetry rows: **{len(telemetry):,}**",
        f"- Telemetry dates: **{telemetry.DateDt.min().date()} to {telemetry.DateDt.max().date()}**",
        f"- Historical visits: **{len(visits):,}**",
        f"- Visits with pre-request telemetry in 28 days: **{matched:,}**",
        f"- Visits without pre-request telemetry in 28 days: **{no_match:,}**", "",
        "## Historical outcome counts", "", features.outcome.value_counts(dropna=False).to_string(), "",
    ]
    if not usable.empty:
        usable["is_fixed"] = (usable.outcome == OUT_FIXED).astype(int)
        candidates = [c for c in usable.columns if any(k in c for k in (
            "reboot_cnt", "disconnection_cnt", "offline_duration_sec", "crc_bad_rate",
            "tx_busy_rate", "rssi_bad_share", "rsrp_bad_share", "ecio_bad_share", "read_rate"
        )) and pd.api.types.is_numeric_dtype(usable[c])]
        rows = []
        for c in candidates:
            a = usable.loc[usable.is_fixed == 1, c].dropna()
            b = usable.loc[usable.is_fixed == 0, c].dropna()
            if len(a) >= 5 and len(b) >= 5:
                rows.append({"feature": c, "fixed_n": len(a), "no_error_n": len(b),
                             "fixed_median": a.median(), "no_error_median": b.median(),
                             "median_diff": a.median() - b.median()})
        lines += ["## Pre-visit feature comparison", ""]
        if rows:
            comp = pd.DataFrame(rows)
            comp["abs_diff"] = comp.median_diff.abs()
            lines.append(comp.sort_values("abs_diff", ascending=False).head(30).round(4).to_string(index=False))
        else:
            lines.append("No sufficiently populated numeric features were available.")
        lines.append("")
    lines += [
        "## Interpretation guardrails", "",
        "- Gateway IDs are normalized before matching across telemetry, field visits, and meter-read data.",
        "- Telemetry features use only dates strictly before `requested_on`.",
        "- 7-day and 28-day windows are calendar windows immediately before the visit date.",
        "- `reason_reported` is descriptive historical information and is not used as a prediction feature.",
        "- `Kein Zugang` is retained as an operational outcome but is not treated as a confirmed gateway fault.",
        "- Outcome differences are evidence for further testing, not proof of causation.",
        "- Final decisions should be evaluated using the challenge's asymmetric visit and missed-fault costs.",
    ]
    (REPORTS / "historical_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading telemetry...")
    telemetry = add_telemetry_features(load_telemetry())
    print(f"Telemetry: {len(telemetry):,} rows")
    daily = telemetry_daily = (
        telemetry.groupby(["gateway_key", "DateDt"], as_index=False)[[
            c for c in telemetry.columns if c in []
        ]].mean()
        if False else None
    )
    # Reuse the feature calculation path directly; aggregate_previsit performs daily aggregation.
    visits = pd.read_csv(DATA / "field_visits.csv", encoding=ENCODING)
    meter = pd.read_csv(DATA / "meter_read_success.csv", encoding=ENCODING)
    print("Building pre-visit telemetry features...")
    features = aggregate_previsit(telemetry, visits)
    print("Adding meter-read features...")
    features = add_meter_features(features, meter)
    feature_file = REPORTS / "historical_visit_features.csv"
    features.to_csv(feature_file, index=False)
    write_report(features, telemetry, visits)
    print(f"Visits with pre-request telemetry: {int(features.DateDt_match.sum()):,}")
    print(f"Visits without pre-request telemetry: {int((~features.DateDt_match).sum()):,}")
    print(f"Wrote: {feature_file}")
    print(f"Wrote: {REPORTS / 'historical_analysis.md'}")


if __name__ == "__main__":
    main()
