from pathlib import Path

import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_features_address_intelligence.tsv"
)

SPLIT_DIR = ROOT / "experiments" / "splits"
OUT_DIR = ROOT / "experiments" / "models"

SEED = 42

ID_COL = "entity_id"
TARGET_COL = "target_entity_id"
LABEL_COL = "label"


ADDRESS_FEATURES = [
    "address_num_jaccard",
    "address_num_overlap_min",
    "address_num_exact_set",
    "address_num_count_diff",
    "address_compound_num_exact",
    "address_compound_num_jaccard",
    "address_alnum_token_jaccard",
    "address_alnum_token_overlap_min",
    "address_first_num_exact_normalized",
    "address_postal_overlap",
]


def load_ids(name):
    return set(
        pd.read_csv(
            SPLIT_DIR / name,
            sep="\t",
            dtype=str,
        )["entity_id"]
    )


def main():

    # -----------------------------------------------------
    # Load split IDs
    # -----------------------------------------------------

    train_ids = load_ids("train_entities.tsv")
    dev_ids = load_ids("dev_entities.tsv")
    holdout_ids = load_ids("holdout_entities.tsv")

    # -----------------------------------------------------
    # Load features
    # -----------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        dtype={
            ID_COL: str,
            TARGET_COL: str,
            LABEL_COL: int,
        },
    )

    # All columns except identifiers and label are model features.
    features = [
        c
        for c in df.columns
        if c not in {
            ID_COL,
            TARGET_COL,
            LABEL_COL,
        }
    ]

    print("Total model features:", len(features))

    # 39 original + 10 address intelligence = 49
    assert len(features) == 49, (
        f"Expected 49 features, got {len(features)}"
    )

    # Verify every new feature exists.
    missing_address_features = [
        c
        for c in ADDRESS_FEATURES
        if c not in features
    ]

    assert not missing_address_features, (
        f"Missing address features: {missing_address_features}"
    )

    print()
    print("Address intelligence features:")

    for c in ADDRESS_FEATURES:
        print("  ", c)

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    train_df = df[
        df[ID_COL].isin(train_ids)
    ]

    dev_df = df[
        df[ID_COL].isin(dev_ids)
    ]

    holdout_df = df[
        df[ID_COL].isin(holdout_ids)
    ]

    print()
    print("Train rows:", len(train_df))
    print("Dev rows:", len(dev_df))
    print("Holdout rows:", len(holdout_df))

    # -----------------------------------------------------
    # Training data
    # -----------------------------------------------------

    X_train = (
        train_df[features]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    y_train = train_df[LABEL_COL].astype(int)

    # -----------------------------------------------------
    # SAME MODEL SETTINGS AS BASELINE
    # -----------------------------------------------------

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

    print()
    print("=" * 60)
    print("Training 49-feature Address Intelligence XGBoost...")
    print("=" * 60)

    # IMPORTANT:
    # Model is trained ONLY on the 1,400 training entities.
    model.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Generate DEV + HOLDOUT predictions
    #
    # We are NOT training on either of these.
    # They are only passed through the trained model.
    # -----------------------------------------------------

    for split_name, subset in [
        ("dev", dev_df),
        ("holdout", holdout_df),
    ]:

        X = (
            subset[features]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .fillna(0)
        )

        out = subset[
            [
                ID_COL,
                TARGET_COL,
                LABEL_COL,
            ]
        ].copy()

        out["probability"] = (
            model.predict_proba(X)[:, 1]
        )

        path = (
            OUT_DIR
            / f"fresh_address_intelligence_{split_name}_predictions.tsv"
        )

        out.to_csv(
            path,
            sep="\t",
            index=False,
        )

        print(
            f"Saved {split_name.upper()} predictions: {path}"
        )

    # -----------------------------------------------------
    # Final information
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("ADDRESS INTELLIGENCE TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Features:",
        len(features)
    )

    print(
        "Train entities:",
        len(train_ids)
    )

    print(
        "Dev entities:",
        len(dev_ids)
    )

    print(
        "Holdout entities:",
        len(holdout_ids)
    )

    print()
    print(
        "Model trained ONLY on TRAIN entities."
    )

    print(
        "DEV and HOLDOUT predictions generated."
    )


if __name__ == "__main__":
    main()