from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_features_v4_augmented.tsv"
)

SPLIT_DIR = ROOT / "experiments" / "splits"
OUT_DIR = ROOT / "experiments" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

SEED = 42
CHUNK_SIZE = 250000

ID_COL = "entity_id"
TARGET_COL = "target_entity_id"
LABEL_COL = "label"

EXTRA_FEATURE = "address_number_conflict"

DROP_COLS = {
    ID_COL,
    TARGET_COL,
    LABEL_COL,
}

THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01),
    2,
)


def load_entities(path):
    return set(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
        )["entity_id"].astype(str)
    )


def load_ground_truth(entity_ids):
    entity_ids = set(entity_ids)

    gt = {
        entity_id: set()
        for entity_id in entity_ids
    }

    print(f"Scanning GT for {len(entity_ids):,} entities...")

    for chunk in pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.fillna("")

        filtered = chunk[
            chunk["source1_entity_id"].isin(entity_ids)
        ]

        for row in filtered.itertuples(index=False):
            values = {
                value.strip()
                for value in str(
                    row.matched_entity_ids
                ).split(",")
                if value.strip()
            }

            gt[row.source1_entity_id].update(values)

    return gt


def entity_f05(true_targets, predicted_targets):
    true_targets = set(true_targets)
    predicted_targets = set(predicted_targets)

    if not true_targets:
        return (
            1.0
            if not predicted_targets
            else 0.0
        )

    tp = len(
        true_targets & predicted_targets
    )

    fp = len(
        predicted_targets - true_targets
    )

    fn = len(
        true_targets - predicted_targets
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    if precision == 0.0 or recall == 0.0:
        return 0.0

    return (
        1.25
        * precision
        * recall
        / (0.25 * precision + recall)
    )


def evaluate_thresholds(predictions, gt):
    entity_ids = predictions[ID_COL].unique()

    rows = []

    for threshold in THRESHOLDS:
        scores = []
        prediction_count = 0

        for entity_id in entity_ids:
            group = predictions[
                predictions[ID_COL] == entity_id
            ]

            selected = set(
                group.loc[
                    group["probability"] >= threshold,
                    TARGET_COL,
                ]
            )

            prediction_count += len(selected)

            scores.append(
                entity_f05(
                    gt[entity_id],
                    selected,
                )
            )

        rows.append({
            "threshold": threshold,
            "entity_macro_f05": float(
                np.mean(scores)
            ),
            "avg_predictions_per_entity": (
                prediction_count / len(entity_ids)
            ),
        })

    return pd.DataFrame(rows)


def main():
    train_entities = load_entities(
        SPLIT_DIR / "train_entities.tsv"
    )

    dev_entities = load_entities(
        SPLIT_DIR / "dev_entities.tsv"
    )

    holdout_entities = load_entities(
        SPLIT_DIR / "holdout_entities.tsv"
    )

    assert not train_entities & dev_entities
    assert not train_entities & holdout_entities
    assert not dev_entities & holdout_entities

    print("=" * 70)
    print("FRESH SOURCE1 HOLDOUT EXPERIMENT")
    print("=" * 70)
    print(f"Train entities:    {len(train_entities):,}")
    print(f"Dev entities:      {len(dev_entities):,}")
    print(f"Holdout entities:  {len(holdout_entities):,}")

    print()
    print("Loading feature table...")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        dtype={
            ID_COL: str,
            TARGET_COL: str,
            LABEL_COL: int,
        },
    )

    # Exact 40-feature configuration.
    all_model_features = [
        column
        for column in df.columns
        if column not in DROP_COLS
    ]

    model_features = [
        column
        for column in all_model_features
        if column != "name_address_mean"
        and column != "name_address_min"
        and column != "name_address_max"
        and column != "name_address_gap"
        and column != "name_dominance"
        and column != "address_dominance"
        and column != "suffix_both_present"
        and column != "suffix_same"
        and column != "suffix_conflict"
    ]

    expected_extra = {
        EXTRA_FEATURE
    }

    if EXTRA_FEATURE not in model_features:
        raise RuntimeError(
            "address_number_conflict missing from "
            "augmented feature table."
        )

    print(
        f"Model features: {len(model_features):,}"
    )

    # Partition by the frozen S1 entity split.
    train_df = df[
        df[ID_COL].isin(train_entities)
    ].copy()

    dev_df = df[
        df[ID_COL].isin(dev_entities)
    ].copy()

    holdout_df = df[
        df[ID_COL].isin(holdout_entities)
    ].copy()

    print()
    print(
        f"Train rows:    {len(train_df):,}"
    )
    print(
        f"Dev rows:      {len(dev_df):,}"
    )
    print(
        f"Holdout rows:  {len(holdout_df):,}"
    )

    X_train = (
        train_df[model_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y_train = train_df[LABEL_COL].astype(int)

    X_dev = (
        dev_df[model_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    X_holdout = (
        holdout_df[model_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

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
    print("Training XGBoost...")

    model.fit(
        X_train,
        y_train,
    )

    print("Predicting development set...")

    dev_predictions = dev_df[
        [ID_COL, TARGET_COL, LABEL_COL]
    ].copy()

    dev_predictions["probability"] = (
        model.predict_proba(X_dev)[:, 1]
    )

    print("Predicting holdout set...")

    holdout_predictions = holdout_df[
        [ID_COL, TARGET_COL, LABEL_COL]
    ].copy()

    holdout_predictions["probability"] = (
        model.predict_proba(X_holdout)[:, 1]
    )

    dev_gt = load_ground_truth(dev_entities)
    holdout_gt = load_ground_truth(holdout_entities)

    print()
    print("=" * 70)
    print("THRESHOLD SELECTION ON DEV ONLY")
    print("=" * 70)

    dev_results = evaluate_thresholds(
        dev_predictions,
        dev_gt,
    )

    best_row = dev_results.loc[
        dev_results["entity_macro_f05"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print(
        f"Best dev threshold:       "
        f"{best_threshold:.2f}"
    )

    print(
        f"Dev Entity Macro F0.5:    "
        f"{best_row['entity_macro_f05']:.6f}"
    )

    print(
        f"Dev avg predictions/entity: "
        f"{best_row['avg_predictions_per_entity']:.3f}"
    )

    print()
    print("=" * 70)
    print("FINAL HOLDOUT EVALUATION")
    print("=" * 70)

    holdout_results = evaluate_thresholds(
        holdout_predictions,
        holdout_gt,
    )

    holdout_row = holdout_results[
        holdout_results["threshold"] == best_threshold
    ].iloc[0]

    print(
        f"Fixed threshold:          "
        f"{best_threshold:.2f}"
    )

    print(
        f"Holdout Entity Macro F0.5:"
        f" {holdout_row['entity_macro_f05']:.6f}"
    )

    print(
        f"Holdout avg predictions/"
        f"entity: {holdout_row['avg_predictions_per_entity']:.3f}"
    )

    dev_results.to_csv(
        OUT_DIR
        / "fresh_holdout_dev_thresholds.tsv",
        sep="\t",
        index=False,
    )

    holdout_results.to_csv(
        OUT_DIR
        / "fresh_holdout_thresholds.tsv",
        sep="\t",
        index=False,
    )

    dev_predictions.to_csv(
        OUT_DIR
        / "fresh_holdout_dev_predictions.tsv",
        sep="\t",
        index=False,
    )

    holdout_predictions.to_csv(
        OUT_DIR
        / "fresh_holdout_predictions.tsv",
        sep="\t",
        index=False,
    )

    print()
    print("Saved fresh-holdout experiment outputs.")


if __name__ == "__main__":
    main()
