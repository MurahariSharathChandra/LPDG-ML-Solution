from pathlib import Path

import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "reports" / "final_model.joblib"
FEATURE_FILE = ROOT / "reports" / "final_features.txt"
INPUT_FILE = ROOT / "reports" / "historical_visit_features.csv"
OUTPUT_FILE = ROOT / "reports" / "prediction_results.csv"

THRESHOLD = 0.45


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("=" * 60)
print("LPDG FAULT PREDICTION")
print("=" * 60)

print("\nLoading final model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# 2. LOAD FEATURE LIST
# ============================================================

print("\nLoading final feature list...")

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    features = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Features required: {len(features)}")


# ============================================================
# 3. LOAD INPUT DATA
# ============================================================

print("\nLoading historical visit features...")

df = pd.read_csv(INPUT_FILE)

print(f"Input rows: {len(df)}")


# ============================================================
# 4. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:
    print("\nERROR: Missing required features:")
    for feature in missing_features:
        print(" -", feature)

    raise RuntimeError(
        "Input data does not contain all required model features."
    )


# ============================================================
# 5. PREPARE MODEL INPUT
# ============================================================

X = df[features].copy()

print(
    f"\nPreparing {len(features)} model features..."
)


# ============================================================
# 6. GENERATE PROBABILITIES
# ============================================================

print("Generating fault probabilities...")

probabilities = model.predict_proba(X)[:, 1]


# ============================================================
# 7. APPLY FINAL THRESHOLD
# ============================================================

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# 8. CREATE RESULTS
# ============================================================

results = pd.DataFrame()

if "visit_id" in df.columns:
    results["visit_id"] = df["visit_id"]

if "gateway_id" in df.columns:
    results["gateway_id"] = df["gateway_id"]

if "requested_on" in df.columns:
    results["requested_on"] = df["requested_on"]

results["fault_probability"] = probabilities
results["predicted_fault"] = predictions

results["prediction"] = results[
    "predicted_fault"
].map({
    0: "No Fault",
    1: "Fault Likely",
})


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 10. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

print(
    f"Total predictions : {len(results)}"
)

print(
    f"Fault likely      : {(predictions == 1).sum()}"
)

print(
    f"No fault          : {(predictions == 0).sum()}"
)

print(
    f"Threshold         : {THRESHOLD}"
)


print("\nSAMPLE PREDICTIONS")

display_columns = [
    column
    for column in [
        "visit_id",
        "gateway_id",
        "fault_probability",
        "prediction",
    ]
    if column in results.columns
]

print(
    results[display_columns]
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)