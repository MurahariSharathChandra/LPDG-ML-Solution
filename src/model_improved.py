from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
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

INPUT_FILE = (
    ROOT / "reports" / "historical_visit_features.csv"
)

OUTPUT_FILE = (
    ROOT / "reports" / "improved_predictions.csv"
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading historical visit features...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Original dataset: "
    f"{df.shape[0]} rows, {df.shape[1]} columns"
)


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

print(
    f"After outcome filtering: {len(df)} rows"
)


# ============================================================
# 3. KEEP VISITS WITH TELEMETRY
# ============================================================

if "telemetry_days_28d" in df.columns:

    df = df[
        df["telemetry_days_28d"] > 0
    ].copy()

print(
    f"After telemetry filtering: {len(df)} rows"
)


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
)


# ============================================================
# 5. REMOVE NON-FEATURE COLUMNS
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
# 6. KEEP NUMERIC FEATURES
# ============================================================

numeric_features = (
    df[feature_columns]
    .select_dtypes(include=["number"])
    .columns
    .tolist()
)

print(
    f"\nNumeric candidate features: "
    f"{len(numeric_features)}"
)


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
    f"Final usable features: "
    f"{len(usable_features)}"
)


if not usable_features:
    raise RuntimeError(
        "No usable numeric features found."
    )


# ============================================================
# 9. BUILD X AND y
# ============================================================

X = df[usable_features].copy()

y = df["target"].copy()


print("\nFINAL MODEL DATA:")
print(f"Rows     : {len(X)}")
print(f"Features : {len(X.columns)}")


# ============================================================
# 10. TRAIN / TEST SPLIT
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
# 11. BUILD PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# 12. HYPERPARAMETER SEARCH
# ============================================================

print("\nStarting hyperparameter tuning...")

param_grid = {

    "classifier__n_estimators": [
        200,
        400,
    ],

    "classifier__max_depth": [
        None,
        10,
        20,
    ],

    "classifier__min_samples_leaf": [
        1,
        2,
        4,
    ],

    "classifier__max_features": [
        "sqrt",
        "log2",
    ],
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1,
)


grid_search.fit(
    X_train,
    y_train,
)


print("\nTUNING COMPLETED.")

print("\nBEST PARAMETERS:")

for key, value in (
    grid_search.best_params_.items()
):

    print(
        f"{key}: {value}"
    )


print(
    f"\nBEST CROSS-VALIDATION "
    f"ROC-AUC: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 13. BEST MODEL
# ============================================================

model = grid_search.best_estimator_


# ============================================================
# 14. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = (
    model.predict_proba(X_test)[:, 1]
)


# ============================================================
# 15. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

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
print("IMPROVED MODEL RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# 16. CONFUSION MATRIX
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
    f"No Fault   "
    f"{cm[0, 0]:7d}   "
    f"{cm[0, 1]:5d}"
)

print(
    f"Fault      "
    f"{cm[1, 0]:7d}   "
    f"{cm[1, 1]:5d}"
)


# ============================================================
# 17. CLASSIFICATION REPORT
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
# 18. FEATURE IMPORTANCE
# ============================================================

rf = (
    model
    .named_steps["classifier"]
)


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
    importances
    .head(20)
    .round(4)
    .to_string()
)


# ============================================================
# 19. SAVE PREDICTIONS
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
    f"\nPredictions written to: "
    f"{OUTPUT_FILE}"
)


# ============================================================
# 20. FINISHED
# ============================================================

print(
    "\nImproved modeling completed successfully."
)