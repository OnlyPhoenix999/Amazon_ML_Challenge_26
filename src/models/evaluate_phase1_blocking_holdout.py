from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PRED_PATH = (
    ROOT
    / "experiments"
    / "models"
    / "phase1_blocking_holdout_predictions.tsv"
)

GT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

SPLIT_PATH = (
    ROOT
    / "experiments"
    / "splits"
    / "holdout_entities.tsv"
)

THRESHOLD = 0.77


def entity_f05(true_set, pred_set):
    true_set = set(true_set)
    pred_set = set(pred_set)

    if not true_set:
        return 1.0 if not pred_set else 0.0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

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


def load_gt(entity_ids):
    entity_ids = set(entity_ids)
    gt = {x: set() for x in entity_ids}

    for chunk in pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        chunksize=250_000,
    ):
        chunk = chunk.fillna("")

        chunk = chunk[
            chunk["source1_entity_id"].isin(entity_ids)
        ]

        for row in chunk.itertuples(index=False):
            value = str(
                row.matched_entity_ids
            ).strip()

            if value:
                gt[row.source1_entity_id].update(
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                )

    return gt


def main():

    holdout_ids = set(
        pd.read_csv(
            SPLIT_PATH,
            sep="\t",
            dtype=str,
        )["entity_id"]
    )

    pred = pd.read_csv(
        PRED_PATH,
        sep="\t",
        dtype=str,
    )

    pred["probability"] = (
        pred["probability"]
        .astype(float)
    )

    if set(pred["entity_id"]) != holdout_ids:
        raise RuntimeError(
            "Prediction entities do not match holdout split."
        )

    gt = load_gt(holdout_ids)

    scores = []
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for entity_id in holdout_ids:

        group = pred[
            pred["entity_id"] == entity_id
        ]

        selected = set(
            group.loc[
                group["probability"] >= THRESHOLD,
                "target_entity_id",
            ]
        )

        true_set = gt[entity_id]

        total_pred += len(selected)
        total_tp += len(true_set & selected)
        total_fp += len(selected - true_set)
        total_fn += len(true_set - selected)

        scores.append(
            entity_f05(
                true_set,
                selected,
            )
        )

    macro_f05 = float(np.mean(scores))

    precision = (
        total_tp / total_pred
        if total_pred
        else 1.0
    )

    total_true = total_tp + total_fn

    recall = (
        total_tp / total_true
        if total_true
        else 0.0
    )

    print("=" * 70)
    print("PHASE 1 — HOLDOUT")
    print("=" * 70)
    print(f"Entities:              {len(holdout_ids):,}")
    print(f"Fixed threshold:       {THRESHOLD:.2f}")
    print(f"Macro F0.5:            {macro_f05:.6f}")
    print(f"Global precision:      {precision:.6f}")
    print(f"Global recall:         {recall:.6f}")
    print(f"Total TP:              {total_tp:,}")
    print(f"Total FP:              {total_fp:,}")
    print(f"Total FN:              {total_fn:,}")
    print(
        f"Avg predictions/entity:"
        f" {total_pred / len(holdout_ids):.3f}"
    )


if __name__ == "__main__":
    main()
