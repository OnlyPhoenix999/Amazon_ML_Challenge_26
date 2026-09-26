from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "data" / "raw" / "dataset" / "train" / "train_ground_truth.tsv"
PRED_DIR = ROOT / "experiments" / "models"

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
CHUNK_SIZE = 250000


def load_gt(ids):
    ids = set(ids)
    gt = {x: set() for x in ids}

    for chunk in pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.fillna("")
        chunk = chunk[
            chunk["source1_entity_id"].isin(ids)
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

    if precision == 0 or recall == 0:
        return 0.0

    return 1.25 * precision * recall / (
        0.25 * precision + recall
    )


def evaluate(path, gt):
    pred = pd.read_csv(path, sep="\t", dtype=str)
    pred["probability"] = pred["probability"].astype(float)

    entities = pred["entity_id"].unique()

    best = (-1, None, None)

    for threshold in THRESHOLDS:
        scores = []
        count = 0

        for entity_id in entities:
            group = pred[pred["entity_id"] == entity_id]

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

        score = float(np.mean(scores))
        avg_predictions = count / len(entities)

        if score > best[0]:
            best = (
                score,
                threshold,
                avg_predictions,
            )

    return best


def main():
    names = [
        "suffix_both",
        "suffix_same",
        "suffix_conflict",
        "address_number_conflict",
        "suffix_conflict_number",
    ]

    first_path = (
        PRED_DIR
        / f"xgboost_{names[0]}_validation_predictions.tsv"
    )

    ids = pd.read_csv(
        first_path,
        sep="\t",
        dtype=str,
        usecols=["entity_id"],
    )["entity_id"].unique()

    gt = load_gt(ids)

    print("=" * 70)
    print("SUFFIX / ADDRESS-CONFLICT ABLATION")
    print("=" * 70)

    for name in names:
        path = (
            PRED_DIR
            / f"xgboost_{name}_validation_predictions.tsv"
        )

        score, threshold, avg = evaluate(path, gt)

        print(
            f"{name:28s} | "
            f"threshold={threshold:.2f} | "
            f"Macro F0.5={score:.6f} | "
            f"avg_pred/entity={avg:.3f}"
        )


if __name__ == "__main__":
    main()
