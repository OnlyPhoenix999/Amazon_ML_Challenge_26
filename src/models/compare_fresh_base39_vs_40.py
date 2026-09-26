from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "training_features_v4_augmented.tsv"
SPLIT_DIR = ROOT / "experiments" / "splits"
GT_PATH = ROOT / "data" / "raw" / "dataset" / "train" / "train_ground_truth.tsv"
OUT_DIR = ROOT / "experiments" / "models"

SEED = 42
CHUNK_SIZE = 250000
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def load_entities(filename):
    return set(
        pd.read_csv(
            SPLIT_DIR / filename,
            sep="\t",
            dtype=str,
        )["entity_id"]
    )


def load_gt(entity_ids):
    entity_ids = set(entity_ids)
    gt = {x: set() for x in entity_ids}

    for chunk in pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.fillna("")
        chunk = chunk[
            chunk["source1_entity_id"].isin(entity_ids)
        ]

        for row in chunk.itertuples(index=False):
            gt[row.source1_entity_id].update(
                x.strip()
                for x in str(row.matched_entity_ids).split(",")
                if x.strip()
            )

    return gt


def f05(true_targets, predicted_targets):
    true_targets = set(true_targets)
    predicted_targets = set(predicted_targets)

    if not true_targets:
        return 1.0 if not predicted_targets else 0.0

    tp = len(true_targets & predicted_targets)
    fp = len(predicted_targets - true_targets)
    fn = len(true_targets - predicted_targets)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    if precision == 0.0 or recall == 0.0:
        return 0.0

    return 1.25 * precision * recall / (
        0.25 * precision + recall
    )


def threshold_search(predictions, gt):
    entities = predictions["entity_id"].unique()
    results = []

    for threshold in THRESHOLDS:
        scores = []
        count = 0

        for entity_id in entities:
            group = predictions[
                predictions["entity_id"] == entity_id
            ]

            selected = set(
                group.loc[
                    group["probability"] >= threshold,
                    "target_entity_id",
                ]
            )

            count += len(selected)

            scores.append(
                f05(
                    gt[entity_id],
                    selected,
                )
            )

        results.append({
            "threshold": threshold,
            "macro_f05": float(np.mean(scores)),
            "avg_predictions_per_entity": (
                count / len(entities)
            ),
        })

    result = pd.DataFrame(results)

    best = result.loc[
        result["macro_f05"].idxmax()
    ]

    return (
        float(best["threshold"]),
        float(best["macro_f05"]),
        float(best["avg_predictions_per_entity"]),
    )


def train_and_predict(df, features, train_ids, eval_ids):
    train_df = df[df["entity_id"].isin(train_ids)]
    eval_df = df[df["entity_id"].isin(eval_ids)]

    X_train = (
        train_df[features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    y_train = train_df["label"].astype(int)

    X_eval = (
        eval_df[features]
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

    model.fit(X_train, y_train)

    output = eval_df[
        ["entity_id", "target_entity_id", "label"]
    ].copy()

    output["probability"] = model.predict_proba(X_eval)[:, 1]

    return output


def main():
    train_ids = load_entities("train_entities.tsv")
    dev_ids = load_entities("dev_entities.tsv")
    holdout_ids = load_entities("holdout_entities.tsv")

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
        if c not in {
            "entity_id",
            "target_entity_id",
            "label",
        }
    ]

    base39 = [
        c for c in all_features
        if c != "address_number_conflict"
    ]

    features40 = all_features

    print("Training base39...")
    base_dev = train_and_predict(
        df,
        base39,
        train_ids,
        dev_ids,
    )

    print("Training base39 holdout...")
    base_holdout = train_and_predict(
        df,
        base39,
        train_ids,
        holdout_ids,
    )

    print("Training feature40...")
    f40_dev = train_and_predict(
        df,
        features40,
        train_ids,
        dev_ids,
    )

    print("Training feature40 holdout...")
    f40_holdout = train_and_predict(
        df,
        features40,
        train_ids,
        holdout_ids,
    )

    dev_gt = load_gt(dev_ids)
    holdout_gt = load_gt(holdout_ids)

    base_threshold, base_dev_score, _ = threshold_search(
        base_dev,
        dev_gt,
    )

    f40_threshold, f40_dev_score, _ = threshold_search(
        f40_dev,
        dev_gt,
    )

    base_holdout_score = float(np.mean([
        f05(
            holdout_gt[e],
            set(
                g.loc[
                    g["probability"] >= base_threshold,
                    "target_entity_id",
                ]
            ),
        )
        for e, g in base_holdout.groupby("entity_id")
    ]))

    f40_holdout_score = float(np.mean([
        f05(
            holdout_gt[e],
            set(
                g.loc[
                    g["probability"] >= f40_threshold,
                    "target_entity_id",
                ]
            ),
        )
        for e, g in f40_holdout.groupby("entity_id")
    ]))

    print()
    print("=" * 70)
    print("FRESH HOLDOUT COMPARISON")
    print("=" * 70)

    print(
        f"Base39 | "
        f"dev_threshold={base_threshold:.2f} | "
        f"dev_f05={base_dev_score:.6f} | "
        f"holdout_f05={base_holdout_score:.6f}"
    )

    print(
        f"Feature40 | "
        f"dev_threshold={f40_threshold:.2f} | "
        f"dev_f05={f40_dev_score:.6f} | "
        f"holdout_f05={f40_holdout_score:.6f}"
    )


if __name__ == "__main__":
    main()
