from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "experiments" / "models"
SPLIT_DIR = ROOT / "experiments" / "splits"

GT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

CHUNK_SIZE = 250000


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

    gt = {
        x: set()
        for x in entity_ids
    }

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
                for x in str(
                    row.matched_entity_ids
                ).split(",")
                if x.strip()
            )

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

    tp = len(true_targets & predicted_targets)
    fp = len(predicted_targets - true_targets)
    fn = len(true_targets - predicted_targets)

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
        / (
            0.25 * precision
            + recall
        )
    )


def evaluate_rule(
    predictions,
    gt,
    threshold,
    max_k=None,
):

    scores = []

    prediction_count = 0

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for entity_id, group in predictions.groupby(
        "entity_id",
        sort=False,
    ):

        group = group.sort_values(
            "probability",
            ascending=False,
        )

        group = group[
            group["probability"] >= threshold
        ]

        if max_k is not None:
            group = group.head(max_k)

        selected = set(
            group["target_entity_id"]
        )

        truth = gt[entity_id]

        tp_total += len(truth & selected)
        fp_total += len(selected - truth)
        fn_total += len(truth - selected)

        prediction_count += len(selected)

        scores.append(
            entity_f05(
                truth,
                selected,
            )
        )

    precision = (
        tp_total / (tp_total + fp_total)
        if tp_total + fp_total
        else 0.0
    )

    recall = (
        tp_total / (tp_total + fn_total)
        if tp_total + fn_total
        else 0.0
    )

    return {
        "macro_f05": float(np.mean(scores)),
        "avg_predictions_per_entity": (
            prediction_count / len(gt)
        ),
        "pair_precision": precision,
        "pair_recall": recall,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def print_result(name, result):

    print()
    print(name)
    print("-" * 70)

    print(
        f"Macro F0.5:              "
        f"{result['macro_f05']:.6f}"
    )

    print(
        f"Pair precision:          "
        f"{result['pair_precision']:.6f}"
    )

    print(
        f"Pair recall:             "
        f"{result['pair_recall']:.6f}"
    )

    print(
        f"Avg predictions/entity:  "
        f"{result['avg_predictions_per_entity']:.3f}"
    )

    print(
        f"TP: {result['tp']}   "
        f"FP: {result['fp']}   "
        f"FN: {result['fn']}"
    )


def main():

    holdout_ids = load_entities(
        "holdout_entities.tsv"
    )

    prediction_path = (
        MODEL_DIR
        / "fresh_address_intelligence_holdout_predictions.tsv"
    )

    predictions = pd.read_csv(
        prediction_path,
        sep="\t",
        dtype=str,
    )

    predictions["probability"] = (
        predictions["probability"]
        .astype(float)
    )

    if set(predictions["entity_id"]) != holdout_ids:
        raise RuntimeError(
            "Holdout prediction entities do not match "
            "holdout_entities.tsv"
        )

    holdout_gt = load_gt(holdout_ids)

    print("=" * 70)
    print("ADDRESS INTELLIGENCE — FINAL HOLDOUT")
    print("=" * 70)

    print(
        f"Holdout entities: {len(holdout_ids)}"
    )

    print(
        f"Prediction rows:  {len(predictions):,}"
    )

    # -----------------------------------------------------
    # RULE SELECTED ON DEV
    # -----------------------------------------------------

    dev_selected = evaluate_rule(
        predictions,
        holdout_gt,
        threshold=0.57,
        max_k=6,
    )

    print_result(
        "DEV-SELECTED RULE: threshold=0.57, max_k=6",
        dev_selected,
    )

    # -----------------------------------------------------
    # Fixed threshold diagnostic
    # -----------------------------------------------------

    fixed_threshold = evaluate_rule(
        predictions,
        holdout_gt,
        threshold=0.69,
        max_k=None,
    )

    print_result(
        "FIXED THRESHOLD: threshold=0.69, no top-k",
        fixed_threshold,
    )

    # -----------------------------------------------------
    # Established baseline
    # -----------------------------------------------------

    baseline = 0.919182

    improvement = (
        dev_selected["macro_f05"]
        - baseline
    )

    relative = (
        improvement / baseline * 100
    )

    print()
    print("=" * 70)
    print("COMPARISON WITH ESTABLISHED BASELINE")
    print("=" * 70)

    print(
        f"39-feature baseline:       {baseline:.6f}"
    )

    print(
        f"49-feature address model:  "
        f"{dev_selected['macro_f05']:.6f}"
    )

    print(
        f"Absolute change:           "
        f"{improvement:+.6f}"
    )

    print(
        f"Relative change:           "
        f"{relative:+.2f}%"
    )

    print()
    print(
        "IMPORTANT: The 0.57 + top-k=6 rule was "
        "selected on DEV before this holdout evaluation."
    )


if __name__ == "__main__":
    main()