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

DROP = {ID_COL, TARGET_ID_COL, LABEL_COL}

NEW_FEATURES = [
    "suffix_both_present",
    "suffix_same",
    "suffix_conflict",
    "address_number_conflict",
]

VARIANTS = {
    "suffix_both": ["suffix_both_present"],
    "suffix_same": ["suffix_same"],
    "suffix_conflict": ["suffix_conflict"],
    "address_number_conflict": ["address_number_conflict"],
    "suffix_conflict_number": [
        "suffix_conflict",
        "address_number_conflict",
    ],
}

RANDOM_STATE = 42


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
        if c not in NEW_FEATURES
    ]

    X = (
        df[all_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y = df[LABEL_COL].astype(int)
    groups = df[ID_COL]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(X, y, groups=groups)
    )

    val_df = df.iloc[val_idx].copy()
    y_train = y.iloc[train_idx]

    print(
        f"Training entities:   "
        f"{df.iloc[train_idx][ID_COL].nunique():,}"
    )
    print(
        f"Validation entities: "
        f"{df.iloc[val_idx][ID_COL].nunique():,}"
    )

    for name, additions in VARIANTS.items():
        feature_columns = base_features + additions

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
        print("=" * 60)
        print(name)
        print("=" * 60)
        print(f"Features: {len(feature_columns)}")

        model.fit(
            X.iloc[train_idx][feature_columns],
            y_train,
        )

        probabilities = model.predict_proba(
            X.iloc[val_idx][feature_columns]
        )[:, 1]

        output = val_df[
            [ID_COL, TARGET_ID_COL, LABEL_COL]
        ].copy()

        output["probability"] = probabilities

        path = (
            OUTPUT_DIR
            / f"xgboost_{name}_validation_predictions.tsv"
        )

        output.to_csv(
            path,
            sep="\t",
            index=False,
        )

        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
