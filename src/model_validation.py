from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "reports" / "historical_visit_features.csv"
OUTPUT_FILE = ROOT / "reports" / "validation_results.csv"

FINAL_THRESHOLD = 0.40


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading historical visit features...")

df = pd.read_csv(INPUT_FILE)

print(f"Original dataset: {df.shape}")


# ============================================================
# 2. KEEP VALID OUTCOMES
# ============================================================

VALID_OUTCOMES = [
    "Fehler behoben",
    "Kein Fehler gefunden",
]

df = df[df["outcome"].isin(VALID_OUTCOMES)].copy()


# ============================================================
# 3. KEEP VISITS WITH TELEMETRY
# ============================================================

if "telemetry_days_28d" in df.columns:
    df = df[df["telemetry_days_28d"] > 0].copy()


print(f"Modeling rows: {len(df)}")


# ============================================================
# 4. CREATE TARGET
# ============================================================

df["target"] = (
    df["outcome"] == "Fehler behoben"
).astype(int)


print("\nTARGET DISTRIBUTION")
print(df["target"].value_counts().sort_index())


# ============================================================
# 5. REMOVE NON-PREDICTIVE COLUMNS
# ============================================================

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


# ============================================================
# 6. NUMERIC FEATURES
# ============================================================

numeric_features = df[feature_columns].select_dtypes(
    include=["number"]
).columns.tolist()


# ============================================================
# 7. MISSING DATA FILTER
# ============================================================

minimum_non_null = len(df) * 0.50

usable_features = [
    c
    for c in numeric_features
    if df[c].notna().sum() >= minimum_non_null
]


# ============================================================
# 8. REMOVE CONSTANT FEATURES
# ============================================================

usable_features = [
    c
    for c in usable_features
    if df[c].nunique(dropna=True) > 1
]


print(f"Final usable features: {len(usable_features)}")


# ============================================================
# 9. BUILD X AND y
# ============================================================

X = df[usable_features].copy()
y = df["target"].copy()


# ============================================================
# 10. MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                max_features="sqrt",
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# 11. STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

print("\nStarting 5-fold cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# cross_val_predict gives out-of-fold probabilities.
# Every row is predicted by a model that did NOT train on it.

probabilities = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]


# ============================================================
# 12. EVALUATE MULTIPLE THRESHOLDS
# ============================================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]


rows = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    rows.append(
        {
            "threshold": threshold,
            "accuracy": accuracy_score(
                y,
                predictions,
            ),
            "precision": precision_score(
                y,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y,
                predictions,
                zero_division=0,
            ),
        }
    )


threshold_results = pd.DataFrame(rows)


# ============================================================
# 13. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y,
    probabilities,
)


# ============================================================
# 14. SELECT FINAL THRESHOLD
# ============================================================

final_predictions = (
    probabilities >= FINAL_THRESHOLD
).astype(int)


accuracy = accuracy_score(
    y,
    final_predictions,
)

precision = precision_score(
    y,
    final_predictions,
    zero_division=0,
)

recall = recall_score(
    y,
    final_predictions,
    zero_division=0,
)

f1 = f1_score(
    y,
    final_predictions,
    zero_division=0,
)


# ============================================================
# 15. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)

print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"Threshold: {FINAL_THRESHOLD:.2f}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    final_predictions,
)


print("\nCONFUSION MATRIX")
print("Rows    = Actual")
print("Columns = Predicted")
print()
print("             No Fault   Fault")
print(
    f"No Fault    {cm[0,0]:8d} {cm[0,1]:7d}"
)
print(
    f"Fault       {cm[1,0]:8d} {cm[1,1]:7d}"
)


# ============================================================
# 17. THRESHOLD COMPARISON
# ============================================================

print("\nTHRESHOLD COMPARISON")
print(
    threshold_results.round(4).to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

validation_output = df[
    [
        "visit_id",
        "gateway_id",
        "requested_on",
        "outcome",
        "target",
    ]
].copy()

validation_output[
    "fault_probability"
] = probabilities

validation_output[
    "predicted"
] = final_predictions

validation_output.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    f"\nValidation predictions written to:"
    f"\n{OUTPUT_FILE}"
)


# ============================================================
# 19. FINISHED
# ============================================================

print(
    "\nModel validation completed successfully."
)