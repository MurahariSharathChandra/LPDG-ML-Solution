#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_telemetry
from src.ranking import build_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank gateways for weekly field visits.")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()

    telemetry = load_telemetry(args.data)
    predictions = build_predictions(telemetry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out, index=False)
    print(f"wrote {args.out} — {len(predictions)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
