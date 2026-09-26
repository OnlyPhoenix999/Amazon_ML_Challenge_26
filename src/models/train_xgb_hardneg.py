from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]

FULL_FEATURE_PATH = ROOT / "data" / "processed" / "training_features_v4.tsv"
HARDNEG_PATH = ROOT / "data" / "processed" / "training_features_v4_hardneg.tsv"

OUTPUT_DIR = ROOT / "experiments" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = OUTPUT_DIR / "xgboost_hardneg_validation_predictions.tsv"

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
    print("Loading original feature table...")
    full = pd.read_csv(
        FULL_FEATURE_PATH,
        sep="\t",
        dtype={
            "entity_id": str,
            "target_entity_id": str,
            "label": int,
        },
    )

    feature_columns = [
        c for c in full.columns
        if c not in DROP_COLUMNS
    ]

    X_full = (
        full[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y_full = full[LABEL_COL].astype(int)
    groups_full = full[ID_COL]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(
            X_full,
            y_full,
            groups=groups_full,
        )
    )

    validation_entities = set(
        full.iloc[val_idx][ID_COL].unique()
    )

    print(f"Validation entities: {len(validation_entities):,}")
    print(f"Validation rows:     {len(val_idx):,}")
    print()

    print("Loading hard-negative training table...")
    hardneg = pd.read_csv(
        HARDNEG_PATH,
        sep="\t",
        dtype={
            "entity_id": str,
            "target_entity_id": str,
            "label": int,
        },
    )

    # Safety check: validation entities must not be in training data.
    overlap = (
        set(hardneg[ID_COL].unique())
        & validation_entities
    )

    if overlap:
        raise RuntimeError(
            f"Validation leakage detected: "
            f"{len(overlap):,} validation entities appear in training."
        )

    X_train = (
        hardneg[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y_train = hardneg[LABEL_COL].astype(int)

    validation_df = full.iloc[val_idx].copy()

    X_val = (
        validation_df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y_val = validation_df[LABEL_COL].astype(int)

    print("=" * 70)
    print("XGBOOST HARD-NEGATIVE EXPERIMENT")
    print("=" * 70)
    print(f"Features:            {len(feature_columns):,}")
    print(f"Training rows:       {len(X_train):,}")
    print(f"Training positives:  {y_train.sum():,}")
    print(f"Training negatives:  {(y_train == 0).sum():,}")
    print(f"Validation rows:     {len(X_val):,}")
    print(f"Validation positives:{y_val.sum():,}")

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
    print("Training XGBoost on hard-negative dataset...")

    model.fit(
        X_train,
        y_train,
    )

    print("Predicting validation set...")

    probabilities = model.predict_proba(X_val)[:, 1]

    out = validation_df[
        [ID_COL, TARGET_ID_COL, LABEL_COL]
    ].copy()

    out["probability"] = probabilities

    out.to_csv(
        PRED_PATH,
        sep="\t",
        index=False,
    )

    print()
    print(f"Saved: {PRED_PATH}")


if __name__ == "__main__":
    main()
