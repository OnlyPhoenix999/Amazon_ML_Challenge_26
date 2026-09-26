from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "training_features_v4_augmented.tsv"
OUTPUT_DIR = ROOT / "experiments" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "entity_id"
TARGET_ID_COL = "target_entity_id"
LABEL_COL = "label"

RELATIONSHIP_FEATURES = [
    "name_address_mean",
    "name_address_min",
    "name_address_max",
    "name_address_gap",
    "name_dominance",
    "address_dominance",
]

SUFFIX_CONFLICT_FEATURES = [
    "suffix_both_present",
    "suffix_same",
    "suffix_conflict",
    "address_number_conflict",
]

DROP = {
    ID_COL,
    TARGET_ID_COL,
    LABEL_COL,
}

RANDOM_STATE = 42
TEST_SIZE = 0.20


def train_variant(name, feature_columns, X_train, y_train, X_val, val_df):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"Features: {len(feature_columns):,}")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    output = val_df[
        [ID_COL, TARGET_ID_COL, LABEL_COL]
    ].copy()

    output["probability"] = probabilities

    path = OUTPUT_DIR / f"xgboost_{name}_validation_predictions.tsv"

    output.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(f"Saved: {path}")


def main():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        dtype={
            "entity_id": str,
            "target_entity_id": str,
            "label": int,
        },
    )

    all_features = [
        c for c in df.columns
        if c not in DROP
    ]

    base_features = [
        c for c in all_features
        if c not in RELATIONSHIP_FEATURES
        and c not in SUFFIX_CONFLICT_FEATURES
    ]

    relationship_features = (
        base_features + RELATIONSHIP_FEATURES
    )

    suffix_features = (
        base_features + SUFFIX_CONFLICT_FEATURES
    )

    variants = {
        "base39": base_features,
        "relationship45": relationship_features,
        "suffixconflict43": suffix_features,
        "full49": all_features,
    }

    X = (
        df[all_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y = df[LABEL_COL].astype(int)
    groups = df[ID_COL]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    val_df = df.iloc[val_idx].copy()

    print(f"Training entities:   {df.iloc[train_idx][ID_COL].nunique():,}")
    print(f"Validation entities: {df.iloc[val_idx][ID_COL].nunique():,}")

    for name, feature_columns in variants.items():
        X_train = X.iloc[train_idx][feature_columns]
        X_val = X.iloc[val_idx][feature_columns]
        y_train = y.iloc[train_idx]

        train_variant(
            name,
            feature_columns,
            X_train,
            y_train,
            X_val,
            val_df,
        )


if __name__ == "__main__":
    main()
