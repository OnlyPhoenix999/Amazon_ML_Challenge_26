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

        prediction_count += len(selected)

        scores.append(
            entity_f05(
                gt[entity_id],
                selected,
            )
        )

    return (
        float(np.mean(scores)),
        prediction_count / len(gt),
    )


def search_dev(
    dev_predictions,
    dev_gt,
):

    rows = []

    # -----------------------------------------------------
    # Global threshold
    # -----------------------------------------------------

    for threshold in np.round(
        np.arange(
            0.55,
            0.851,
            0.01,
        ),
        2,
    ):

        score, avg = evaluate_rule(
            dev_predictions,
            dev_gt,
            threshold,
            max_k=None,
        )

        rows.append({
            "rule": "global_threshold",
            "threshold": threshold,
            "max_k": 0,
            "macro_f05": score,
            "avg_predictions_per_entity": avg,
        })

    # -----------------------------------------------------
    # Threshold + top-k
    # -----------------------------------------------------

    for threshold in np.round(
        np.arange(
            0.55,
            0.851,
            0.01,
        ),
        2,
    ):

        for k in range(1, 7):

            score, avg = evaluate_rule(
                dev_predictions,
                dev_gt,
                threshold,
                max_k=k,
            )

            rows.append({
                "rule": "threshold_top_k",
                "threshold": threshold,
                "max_k": k,
                "macro_f05": score,
                "avg_predictions_per_entity": avg,
            })

    return pd.DataFrame(rows)


def main():

    dev_ids = load_entities(
        "dev_entities.tsv"
    )

    predictions_path = (
        MODEL_DIR
        / "fresh_base39_dev_predictions.tsv"
    )

    predictions = pd.read_csv(
        predictions_path,
        sep="\t",
        dtype=str,
    )

    predictions["probability"] = (
        predictions["probability"]
        .astype(float)
    )

    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------

    if set(predictions["entity_id"]) != dev_ids:

        raise RuntimeError(
            "Prediction entities do not match "
            "dev_entities.tsv"
        )

    dev_gt = load_gt(dev_ids)

    print("=" * 70)
    print("ADDRESS INTELLIGENCE — DEV ONLY")
    print("=" * 70)

    print(
        f"Prediction rows: {len(predictions):,}"
    )

    print(
        f"Dev entities: {len(dev_ids):,}"
    )

    # -----------------------------------------------------
    # Search threshold/rule
    # -----------------------------------------------------

    results = search_dev(
        predictions,
        dev_gt,
    )

    best = results.loc[
        results["macro_f05"].idxmax()
    ]

    print()
    print("BEST ADDRESS-INTELLIGENCE DEV RULE")
    print("-" * 70)

    print(
        f"Rule:       {best['rule']}"
    )

    print(
        f"Threshold:  {best['threshold']:.2f}"
    )

    print(
        f"Max K:      {int(best['max_k'])}"
    )

    print(
        f"Dev F0.5:   {best['macro_f05']:.6f}"
    )

    print(
        f"Avg pred/e: "
        f"{best['avg_predictions_per_entity']:.3f}"
    )

    # -----------------------------------------------------
    # Fixed threshold comparison
    # -----------------------------------------------------

    fixed_score, fixed_avg = evaluate_rule(
        predictions,
        dev_gt,
        threshold=0.69,
        max_k=None,
    )

    print()
    print("FIXED THRESHOLD 0.69")
    print("-" * 70)

    print(
        f"Dev F0.5:   {fixed_score:.6f}"
    )

    print(
        f"Avg pred/e: {fixed_avg:.3f}"
    )

    # -----------------------------------------------------
    # Baseline reference
    #
    # Established fresh baseline:
    # HOLDOUT = 0.919182
    #
    # The old dev score should NOT be confused with this.
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("REFERENCE")
    print("=" * 70)

    print(
        "Established fresh 39-feature "
        "holdout baseline: 0.919182"
    )

    print()
    print(
        "NO HOLDOUT WAS READ OR EVALUATED."
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    out_path = (
        MODEL_DIR
        / "fresh_base39_dev_results.tsv"
    )

    results.to_csv(
        out_path,
        sep="\t",
        index=False,
    )

    print()
    print(
        f"Saved: {out_path}"
    )


if __name__ == "__main__":
    main()
