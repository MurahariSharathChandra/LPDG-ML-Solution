from pathlib import Path

import numpy as np
import pandas as pd

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

INPUT_FILE = ROOT / "reports" / "improved_predictions.csv"


# ============================================================
# 1. LOAD IMPROVED MODEL PREDICTIONS
# ============================================================

print("Loading improved model predictions...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns: {df.columns.tolist()}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "actual",
    "fault_probability",
]

for column in required_columns:
    if column not in df.columns:
        raise RuntimeError(
            f"Required column missing: {column}"
        )


# ============================================================
# 3. REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "actual",
        "fault_probability",
    ]
).copy()


y_true = df["actual"].astype(int)

y_prob = df["fault_probability"].astype(float)


# ============================================================
# 4. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_true,
    y_prob,
)

print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

print(f"ROC-AUC: {roc_auc:.4f}")


# ============================================================
# 5. TEST MULTIPLE THRESHOLDS
# ============================================================

thresholds = np.arange(
    0.20,
    0.81,
    0.05,
)

rows = []

for threshold in thresholds:

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    rows.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


results = pd.DataFrame(rows)


# ============================================================
# 6. DISPLAY ALL THRESHOLDS
# ============================================================

print("\nTHRESHOLD COMPARISON")

print(
    results.round(4).to_string(
        index=False
    )
)


# ============================================================
# 7. FIND BEST F1 THRESHOLD
# ============================================================

best_row = results.loc[
    results["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

print("\n" + "=" * 60)
print("BEST THRESHOLD BY F1-SCORE")
print("=" * 60)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Accuracy  : {best_row['accuracy']:.4f}"
)

print(
    f"Precision : {best_row['precision']:.4f}"
)

print(
    f"Recall    : {best_row['recall']:.4f}"
)

print(
    f"F1-Score  : {best_row['f1']:.4f}"
)


# ============================================================
# 8. CONFUSION MATRIX FOR BEST THRESHOLD
# ============================================================

best_predictions = (
    y_prob >= best_threshold
).astype(int)


cm = confusion_matrix(
    y_true,
    best_predictions,
)


print("\nCONFUSION MATRIX")
print("Rows    = Actual")
print("Columns = Predicted")

print(
    "             No Fault   Fault"
)

print(
    f"No Fault    {cm[0, 0]:8d}   {cm[0, 1]:5d}"
)

print(
    f"Fault       {cm[1, 0]:8d}   {cm[1, 1]:5d}"
)


# ============================================================
# 9. SAVE THRESHOLD RESULTS
# ============================================================

OUTPUT_FILE = (
    ROOT
    / "reports"
    / "threshold_results.csv"
)

results.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(
    f"\nThreshold results written to: "
    f"{OUTPUT_FILE}"
)


print(
    "\nThreshold tuning completed successfully."
)