from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "reports" / "historical_visit_features.csv"
OUTPUT_FILE = ROOT / "reports" / "baseline_predictions.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading historical visit features...")

df = pd.read_csv(INPUT_FILE)

print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")


# ============================================================
# 2. KEEP ONLY USABLE OUTCOMES
# ============================================================

# We model confirmed operational outcomes only:
#
# Fehler behoben       -> 1
# Kein Fehler gefunden -> 0
#
# Kein Zugang is excluded because the historical analysis
# explicitly does not treat it as a confirmed gateway fault.

VALID_OUTCOMES = [
    "Fehler behoben",
    "Kein Fehler gefunden",
]

df = df[df["outcome"].isin(VALID_OUTCOMES)].copy()

print(f"After outcome filtering: {len(df)} rows")


# ============================================================
# 3. KEEP VISITS WITH TELEMETRY
# ============================================================

# The historical analysis found that only 314 visits had
# pre-request telemetry in the 28-day window.
#
# telemetry_days_28d > 0 means telemetry existed before visit.

if "telemetry_days_28d" in df.columns:
    df = df[df["telemetry_days_28d"] > 0].copy()

print(f"After telemetry filtering: {len(df)} rows")


# ============================================================
# 4. CREATE TARGET
# ============================================================

df["target"] = (
    df["outcome"] == "Fehler behoben"
).astype(int)

print("\nTARGET DISTRIBUTION:")
print(
    df["target"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Kein Fehler gefunden",
        1: "Fehler behoben",
    })
)


# ============================================================
# 5. REMOVE NON-PREDICTIVE / IDENTIFIER COLUMNS
# ============================================================

# These columns should not be used as model features.

EXCLUDE_COLUMNS = {
    "visit_id",
    "gateway_id",
    "requested_on",
    "outcome",
    "reason_reported",
    "cutoff_date",
    "DateDt_match",
    "target",
}


feature_columns = [
    c
    for c in df.columns
    if c not in EXCLUDE_COLUMNS
]


# Keep numeric features only.
numeric_features = df[feature_columns].select_dtypes(
    include=["number"]
).columns.tolist()


print(f"\nNumeric candidate features: {len(numeric_features)}")


# ============================================================
# 6. REMOVE FEATURES WITH TOO MUCH MISSING DATA
# ============================================================

# A feature must have data for at least 50% of the modeling
# rows.

minimum_non_null = len(df) * 0.50

usable_features = [
    c
    for c in numeric_features
    if df[c].notna().sum() >= minimum_non_null
]


print(
    f"Features with >=50% data: {len(usable_features)}"
)


# ============================================================
# 7. REMOVE CONSTANT FEATURES
# ============================================================

usable_features = [
    c
    for c in usable_features
    if df[c].nunique(dropna=True) > 1
]


print(
    f"Features after removing constants: "
    f"{len(usable_features)}"
)


if len(usable_features) == 0:
    raise RuntimeError(
        "No usable numeric features were found."
    )


# ============================================================
# 8. BUILD X AND y
# ============================================================

X = df[usable_features].copy()
y = df["target"].copy()


print("\nFINAL MODEL DATA:")
print(f"Rows: {len(X)}")
print(f"Features: {len(X.columns)}")


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


print("\nTRAIN / TEST:")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")


# ============================================================
# 10. BUILD BASELINE MODEL
# ============================================================

# Median imputation handles missing telemetry values.
#
# Random Forest is used as the first baseline because:
# - it handles nonlinear relationships
# - it works well with mixed telemetry measurements
# - it does not require feature scaling

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\nTraining Random Forest baseline...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 12. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 13. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    y_prob,
)


print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
)

print("\nCONFUSION MATRIX")
print(
    "Rows    = Actual"
)
print(
    "Columns = Predicted"
)
print(
    "          No Fault   Fault"
)
print(
    f"No Fault   {cm[0, 0]:7d}   {cm[0, 1]:5d}"
)
print(
    f"Fault      {cm[1, 0]:7d}   {cm[1, 1]:5d}"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Kein Fehler gefunden",
            "Fehler behoben",
        ],
        zero_division=0,
    )
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

rf = model.named_steps["classifier"]

importances = pd.Series(
    rf.feature_importances_,
    index=usable_features,
).sort_values(
    ascending=False
)


print("\n" + "=" * 60)
print("TOP 20 FEATURE IMPORTANCES")
print("=" * 60)

print(
    importances.head(20).round(4).to_string()
)


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

results = df.loc[
    X_test.index,
    [
        "visit_id",
        "gateway_id",
        "requested_on",
        "outcome",
    ],
].copy()

results["actual"] = y_test
results["predicted"] = y_pred
results["fault_probability"] = y_prob

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nPredictions written to: {OUTPUT_FILE}"
)


# ============================================================
# 18. FINISHED
# ============================================================

print("\nBaseline modeling completed successfully.")
