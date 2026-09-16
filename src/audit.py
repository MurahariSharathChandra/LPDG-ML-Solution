from __future__ import annotations

from pathlib import Path
import pandas as pd


def audit(data_dir: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    master = pd.read_csv(data_dir / 'gateway_master.csv', encoding='latin1')
    meter = pd.read_csv(data_dir / 'meter_read_success.csv')
    visits = pd.read_csv(data_dir / 'field_visits.csv', encoding='latin1')
    review = pd.read_excel(data_dir / 'engineer_review_2026-02.xlsx')

    result['gateway_master_rows'] = len(master)
    result['gateway_master_unique_gateways'] = master.gateway_id.astype(str).str.replace(':','',regex=False).str.upper().nunique()
    result['meter_rows'] = len(meter)
    result['meter_unique_gateways'] = meter.gateway_id.nunique()
    result['meter_weeks'] = meter.week_start.nunique()
    result['field_visit_rows'] = len(visits)
    result['field_visit_unique_gateways'] = visits.gateway_id.astype(str).str.replace(':','',regex=False).str.upper().nunique()
    result['field_visit_outcomes'] = visits.outcome.value_counts().to_dict()
    result['engineer_review_rows'] = len(review)
    result['engineer_review_categories'] = review['Kategorie'].value_counts().to_dict()

    sample_path = data_dir / 'telemetry_sample_2025-08.csv'
    if sample_path.exists():
        sample = pd.read_csv(sample_path, encoding='latin1', decimal=',')
        result['telemetry_sample_rows'] = len(sample)
        result['telemetry_sample_gateways'] = sample.gateway_id.nunique()
        result['telemetry_sample_columns'] = len(sample.columns)
        result['telemetry_sample_start'] = sample.ts_utc.min()
        result['telemetry_sample_end'] = sample.ts_utc.max()
        result['telemetry_sample_missing_rate'] = float(sample.isna().mean().mean())
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    args = parser.parse_args()
    for k, v in audit(args.data).items():
        print(f'{k}: {v}')
