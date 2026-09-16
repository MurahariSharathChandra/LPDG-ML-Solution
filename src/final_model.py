from pathlib import Path

import joblib
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

MODEL_FILE = ROOT / "reports" / "final_model.joblib"
FEATURE_FILE = ROOT / "reports" / "final_features.txt"
IMPORTANCE_FILE = ROOT / "reports" / "final_feature_importance.csv"
EVALUATION_FILE = ROOT / "reports" / "final_evaluation.csv"

FINAL_THRESHOLD = 0.45


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

df = df[
    df["outcome"].isin(VALID_OUTCOMES)
].copy()


# ============================================================
# 3. KEEP VISITS WITH TELEMETRY
# ============================================================

if "telemetry_days_28d" in df.columns:
    df = df[
        df["telemetry_days_28d"] > 0
    ].copy()


print(f"Final modeling rows: {len(df)}")


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
# 6. SELECT NUMERIC FEATURES
# ============================================================

numeric_features = df[
    feature_columns
].select_dtypes(
    include=["number"]
).columns.tolist()


# ============================================================
# 7. REMOVE FEATURES WITH TOO MUCH MISSING DATA
# ============================================================

minimum_non_null = len(df) * 0.50

usable_features = [
    c
    for c in numeric_features
    if df[c].notna().sum()
    >= minimum_non_null
]


# ============================================================
# 8. REMOVE CONSTANT FEATURES
# ============================================================

usable_features = [
    c
    for c in usable_features
    if df[c].nunique(dropna=True) > 1
]


print(
    f"\nFinal usable features: "
    f"{len(usable_features)}"
)


# ============================================================
# 9. BUILD DATA
# ============================================================

X = df[
    usable_features
].copy()

y = df["target"].copy()


print("\nFINAL MODEL DATA")
print(f"Rows     : {len(X)}")
print(f"Features : {len(X.columns)}")


# ============================================================
# 10. FINAL MODEL CONFIGURATION
# ============================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
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
# 11. FIVE-FOLD OUT-OF-FOLD EVALUATION
# ============================================================

print(
    "\nRunning final 5-fold evaluation..."
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


probabilities = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]


# ============================================================
# 12. APPLY FINAL THRESHOLD
# ============================================================

predictions = (
    probabilities >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# 13. FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    predictions,
)

precision = precision_score(
    y,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y,
    probabilities,
)


# ============================================================
# 14. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL VALIDATION RESULTS")
print("=" * 60)

print(f"Threshold : {FINAL_THRESHOLD:.2f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    predictions,
)


print("\nCONFUSION MATRIX")
print("Rows    = Actual")
print("Columns = Predicted")
print()
print("             No Fault   Fault")
print(
    f"No Fault    {cm[0, 0]:8d} "
    f"{cm[0, 1]:7d}"
)
print(
    f"Fault       {cm[1, 0]:8d} "
    f"{cm[1, 1]:7d}"
)


# ============================================================
# 16. SAVE EVALUATION RESULTS
# ============================================================

evaluation = pd.DataFrame(
    [
        {
            "threshold": FINAL_THRESHOLD,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "true_negative": cm[0, 0],
            "false_positive": cm[0, 1],
            "false_negative": cm[1, 0],
            "true_positive": cm[1, 1],
        }
    ]
)

evaluation.to_csv(
    EVALUATION_FILE,
    index=False,
)


# ============================================================
# 17. TRAIN FINAL MODEL ON ALL DATA
# ============================================================

print(
    "\nTraining final model on all "
    "available modeling data..."
)

model.fit(X, y)

print("Final model training completed.")


# ============================================================
# 18. FEATURE IMPORTANCE
# ============================================================

rf = model.named_steps[
    "classifier"
]

importances = pd.DataFrame(
    {
        "feature": usable_features,
        "importance": rf.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)


print("\n" + "=" * 60)
print("TOP 20 FINAL FEATURES")
print("=" * 60)

print(
    importances.head(20)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 19. SAVE FEATURE IMPORTANCE
# ============================================================

importances.to_csv(
    IMPORTANCE_FILE,
    index=False,
)


# ============================================================
# 20. SAVE FEATURE LIST
# ============================================================

with open(
    FEATURE_FILE,
    "w",
    encoding="utf-8",
) as f:

    for feature in usable_features:
        f.write(feature + "\n")


# ============================================================
# 21. SAVE FINAL MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE,
)


# ============================================================
# 22. FINISHED
# ============================================================

print(
    f"\nFinal model saved to:"
    f"\n{MODEL_FILE}"
)

print(
    f"\nFeature list saved to:"
    f"\n{FEATURE_FILE}"
)

print(
    f"\nFeature importance saved to:"
    f"\n{IMPORTANCE_FILE}"
)

print(
    f"\nEvaluation saved to:"
    f"\n{EVALUATION_FILE}"
)

print(
    "\nFINAL MODEL PIPELINE COMPLETED SUCCESSFULLY."
)