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


def entity_f05(true_targets, predicted_targets):
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


def evaluate_rule(predictions, gt, threshold, max_k=None):
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

        selected = set(group["target_entity_id"])

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


def search_dev(dev_predictions, dev_gt):
    rows = []

    # Baseline global-threshold family.
    for threshold in np.round(
        np.arange(0.55, 0.851, 0.01),
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

    # Threshold + per-entity top-k cap.
    for threshold in np.round(
        np.arange(0.55, 0.851, 0.01),
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

    result = pd.DataFrame(rows)

    return result


def main():
    dev_ids = load_entities("dev_entities.tsv")
    holdout_ids = load_entities("holdout_entities.tsv")

    dev_predictions = pd.read_csv(
        MODEL_DIR / "fresh_base39_dev_predictions.tsv",
        sep="\t",
        dtype=str,
    )

    holdout_predictions = pd.read_csv(
        MODEL_DIR / "fresh_base39_holdout_predictions.tsv",
        sep="\t",
        dtype=str,
    )

    dev_predictions["probability"] = (
        dev_predictions["probability"].astype(float)
    )

    holdout_predictions["probability"] = (
        holdout_predictions["probability"].astype(float)
    )

    if set(dev_predictions["entity_id"]) != dev_ids:
        raise RuntimeError(
            "Development prediction entities do not match "
            "dev_entities.tsv"
        )

    if set(holdout_predictions["entity_id"]) != holdout_ids:
        raise RuntimeError(
            "Holdout prediction entities do not match "
            "holdout_entities.tsv"
        )

    dev_gt = load_gt(dev_ids)
    holdout_gt = load_gt(holdout_ids)

    print("=" * 70)
    print("ENTITY-LEVEL RANKING / SELECTION EXPERIMENT")
    print("=" * 70)

    dev_results = search_dev(
        dev_predictions,
        dev_gt,
    )

    best_dev = dev_results.loc[
        dev_results["macro_f05"].idxmax()
    ]

    print()
    print("Best dev rule:")
    print(
        f"Rule:       {best_dev['rule']}"
    )
    print(
        f"Threshold:  {best_dev['threshold']:.2f}"
    )
    print(
        f"Max K:      {int(best_dev['max_k'])}"
    )
    print(
        f"Dev F0.5:   {best_dev['macro_f05']:.6f}"
    )
    print(
        f"Avg pred/e: "
        f"{best_dev['avg_predictions_per_entity']:.3f}"
    )

    # Fixed baseline from the prior fresh-holdout experiment.
    baseline_score, baseline_avg = evaluate_rule(
        dev_predictions,
        dev_gt,
        threshold=0.69,
        max_k=None,
    )

    print()
    print("Dev baseline at threshold 0.69:")
    print(f"F0.5:       {baseline_score:.6f}")
    print(f"Avg pred/e:  {baseline_avg:.3f}")

    # Apply the chosen dev rule ONCE to untouched holdout.
    holdout_score, holdout_avg = evaluate_rule(
        holdout_predictions,
        holdout_gt,
        threshold=float(best_dev["threshold"]),
        max_k=(
            int(best_dev["max_k"])
            if int(best_dev["max_k"]) > 0
            else None
        ),
    )

    baseline_holdout_score, baseline_holdout_avg = evaluate_rule(
        holdout_predictions,
        holdout_gt,
        threshold=0.69,
        max_k=None,
    )

    print()
    print("=" * 70)
    print("FRESH HOLDOUT RESULTS")
    print("=" * 70)

    print(
        f"Baseline threshold 0.69: "
        f"{baseline_holdout_score:.6f} "
        f"(avg={baseline_holdout_avg:.3f})"
    )

    print(
        f"Dev-selected ranking rule: "
        f"{holdout_score:.6f} "
        f"(avg={holdout_avg:.3f})"
    )

    dev_results.to_csv(
        MODEL_DIR / "fresh_ranking_dev_results.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()

