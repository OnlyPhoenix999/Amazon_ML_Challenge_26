import sys
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_features_sample.tsv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t"
)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# PREPARE FEATURES
# ============================================================

# IDs are identifiers, NOT model features.
DROP_COLUMNS = [
    "s1_entity_id",
    "target_entity_id",
    "label",
]

X = df.drop(columns=DROP_COLUMNS)
y = df["label"]


# Make sure everything is numeric
X = X.apply(pd.to_numeric, errors="coerce")

X = X.fillna(0)


print(f"\nTraining features: {X.shape[1]}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")


# ============================================================
# LIGHTGBM
# ============================================================

print("\nTraining LightGBM...")


model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=300,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1,

    verbosity=-1,
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

print("Predicting validation set...")

probabilities = model.predict_proba(X_val)[:, 1]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\nSearching thresholds...")

best_threshold = 0.5
best_f05 = 0.0

results = []

for threshold in [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_val,
        predictions,
        beta=0.5,
        zero_division=0
    )

    results.append(
        (
            threshold,
            precision,
            recall,
            f05
        )
    )

    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("LIGHTGBM BASELINE")
print("========================================")

print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Best F0.5:      {best_f05:.4f}"
)

print("\nThreshold results:")
print(
    "Threshold | Precision | Recall | F0.5"
)

for threshold, precision, recall, f05 in results:

    print(
        f"{threshold:8.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f05:5.4f}"
    )


# ============================================================
# FINAL VALIDATION METRICS
# ============================================================

final_predictions = (
    probabilities >= best_threshold
).astype(int)


print("\nClassification report:")

print(
    classification_report(
        y_val,
        final_predictions,
        digits=4,
        zero_division=0
    )
)


print("Confusion matrix:")

print(
    confusion_matrix(
        y_val,
        final_predictions
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n========================================")
print("TOP FEATURE IMPORTANCE")
print("========================================")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.head(20).to_string(
        index=False
    )
)