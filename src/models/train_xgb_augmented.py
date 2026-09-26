from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_features_v4_augmented.tsv"
)

OUTPUT_PATH = (
    ROOT
    / "experiments"
    / "models"
    / "xgboost_augmented_validation_predictions.tsv"
)

ID_COL = "entity_id"
TARGET_ID_COL = "target_entity_id"
LABEL_COL = "label"

DROP_COLUMNS = [
    ID_COL,
    TARGET_ID_COL,
    LABEL_COL,
]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    print(f"Loading: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        dtype={
            "entity_id": str,
            "target_entity_id": str,
            "label": int,
        },
    )

    feature_columns = [
        c for c in df.columns
        if c not in DROP_COLUMNS
    ]

    X = (
        df[feature_columns]
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

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    validation_df = df.iloc[val_idx].copy()

    print("=" * 70)
    print("XGBOOST 49-FEATURE EXPERIMENT")
    print("=" * 70)
    print(f"Features:             {len(feature_columns):,}")
    print(f"Training rows:        {len(train_idx):,}")
    print(f"Validation rows:      {len(val_idx):,}")
    print(
        f"Training entities:   "
        f"{df.iloc[train_idx][ID_COL].nunique():,}"
    )
    print(
        f"Validation entities: "
        f"{df.iloc[val_idx][ID_COL].nunique():,}"
    )
    print(f"Training positives:   {y_train.sum():,}")
    print(f"Validation positives: {y_val.sum():,}")

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

    print()
    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train,
    )

    print("Predicting validation set...")

    probabilities = model.predict_proba(X_val)[:, 1]

    output = validation_df[
        [
            ID_COL,
            TARGET_ID_COL,
            LABEL_COL,
        ]
    ].copy()

    output["probability"] = probabilities

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
