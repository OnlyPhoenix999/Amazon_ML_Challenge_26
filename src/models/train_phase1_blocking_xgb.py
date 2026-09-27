from pathlib import Path

import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_features_phase1_blocking.tsv"
)

SPLIT_DIR = ROOT / "experiments" / "splits"
OUT_DIR = ROOT / "experiments" / "models"

SEED = 42

ID_COL = "entity_id"
TARGET_COL = "target_entity_id"
LABEL_COL = "label"

# These are the older augmentation columns in
# training_features_v4_augmented.tsv.
# We deliberately exclude them so Phase 1 is:
# 39 core + 7 blocking evidence = 46 features.
LEGACY_AUGMENTATION = {
    "name_address_mean",
    "name_address_min",
    "name_address_max",
    "name_address_gap",
    "name_dominance",
    "address_dominance",
    "suffix_both_present",
    "suffix_same",
    "suffix_conflict",
    "address_number_conflict",
}

BLOCKING_FEATURES = {
    "v4_shared_key_count",
    "v4_shared_name_key_count",
    "v4_shared_address_key_count",
    "v4_name_first_match",
    "v4_name_prefix2_match",
    "v4_address_number_match",
    "v4_address_first_match",
}


def load_ids(name):
    return set(
        pd.read_csv(
            SPLIT_DIR / name,
            sep="\t",
            dtype=str,
        )["entity_id"]
    )


def numeric_frame(df, features):
    return (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def main():

    train_ids = load_ids("train_entities.tsv")
    dev_ids = load_ids("dev_entities.tsv")
    holdout_ids = load_ids("holdout_entities.tsv")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        dtype={
            ID_COL: str,
            TARGET_COL: str,
            LABEL_COL: int,
        },
    )

    all_features = [
        c for c in df.columns
        if c not in {
            ID_COL,
            TARGET_COL,
            LABEL_COL,
        }
    ]

    # Remove the older 10-feature augmentation block.
    features = [
        c for c in all_features
        if c not in LEGACY_AUGMENTATION
    ]

    print(f"All available features : {len(all_features)}")
    print(f"Model features         : {len(features)}")

    assert len(features) == 46, (
        f"Expected 46 model features, got {len(features)}"
    )

    missing = [
        c for c in BLOCKING_FEATURES
        if c not in features
    ]

    assert not missing, (
        f"Missing blocking features: {missing}"
    )

    train_df = df[df[ID_COL].isin(train_ids)]
    dev_df = df[df[ID_COL].isin(dev_ids)]
    holdout_df = df[df[ID_COL].isin(holdout_ids)]

    print(f"Train rows   : {len(train_df):,}")
    print(f"Dev rows     : {len(dev_df):,}")
    print(f"Holdout rows : {len(holdout_df):,}")

    X_train = numeric_frame(train_df, features)
    y_train = train_df[LABEL_COL].astype(int)

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
        random_state=SEED,
        n_jobs=4,
    )

    print("\nTraining Phase 1 XGBoost...")
    model.fit(X_train, y_train)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, subset in [
        ("dev", dev_df),
        ("holdout", holdout_df),
    ]:

        X = numeric_frame(subset, features)

        out = subset[
            [ID_COL, TARGET_COL, LABEL_COL]
        ].copy()

        out["probability"] = model.predict_proba(X)[:, 1]

        path = (
            OUT_DIR
            / f"phase1_blocking_{split_name}_predictions.tsv"
        )

        out.to_csv(
            path,
            sep="\t",
            index=False,
        )

        print(f"Saved: {path}")

    print("\n" + "=" * 60)
    print("PHASE 1 TRAINING COMPLETE")
    print("=" * 60)
    print("Model features:", len(features))
    print("  39 core features")
    print("  7 blocking-evidence features")
    print("Legacy augmentation features excluded.")
    print("=" * 60)


if __name__ == "__main__":
    main()