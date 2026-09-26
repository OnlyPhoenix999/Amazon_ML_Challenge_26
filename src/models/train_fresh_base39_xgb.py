from pathlib import Path
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "training_features_v4_augmented.tsv"
SPLIT_DIR = ROOT / "experiments" / "splits"
OUT_DIR = ROOT / "experiments" / "models"

SEED = 42

ID_COL = "entity_id"
TARGET_COL = "target_entity_id"
LABEL_COL = "label"

DROP = {
    ID_COL,
    TARGET_COL,
    LABEL_COL,
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

def load_ids(name):
    return set(
        pd.read_csv(
            SPLIT_DIR / name,
            sep="\t",
            dtype=str,
        )["entity_id"]
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

    features = [
        c for c in df.columns
        if c not in DROP
    ]

    assert len(features) == 39, len(features)

    train_df = df[df[ID_COL].isin(train_ids)]
    dev_df = df[df[ID_COL].isin(dev_ids)]
    holdout_df = df[df[ID_COL].isin(holdout_ids)]

    X_train = (
        train_df[features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
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

    print("Training 39-feature XGBoost...")
    model.fit(X_train, y_train)

    for name, subset in [
        ("dev", dev_df),
        ("holdout", holdout_df),
    ]:
        X = (
            subset[features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        out = subset[
            [ID_COL, TARGET_COL, LABEL_COL]
        ].copy()

        out["probability"] = model.predict_proba(X)[:, 1]

        path = (
            OUT_DIR
            / f"fresh_base39_{name}_predictions.tsv"
        )

        out.to_csv(
            path,
            sep="\t",
            index=False,
        )

        print(f"Saved: {path}")

    print()
    print("39-feature count:", len(features))
    print("Train entities:", len(train_ids))
    print("Dev entities:", len(dev_ids))
    print("Holdout entities:", len(holdout_ids))


if __name__ == "__main__":
    main()

