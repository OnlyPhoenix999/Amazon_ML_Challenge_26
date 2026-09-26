from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PRED_PATH = ROOT / "experiments" / "models" / "xgboost_validation_predictions.tsv"
GT_PATH = ROOT / "data" / "raw" / "dataset" / "train" / "train_ground_truth.tsv"

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
CHUNK_SIZE = 250000


def f05_from_sets(true_targets, predicted_targets):
    true_targets = set(true_targets)
    predicted_targets = set(predicted_targets)

    # Official special case.
    if not true_targets:
        return 1.0 if not predicted_targets else 0.0

    tp = len(true_targets & predicted_targets)
    fp = len(predicted_targets - true_targets)
    fn = len(true_targets - predicted_targets)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return 1.25 * precision * recall / (0.25 * precision + recall)


def load_validation_ground_truth(validation_ids):
    validation_ids = set(validation_ids)
    gt = {entity_id: set() for entity_id in validation_ids}

    print("Scanning ground truth...")

    for chunk in pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.fillna("")

        filtered = chunk[
            chunk["source1_entity_id"].isin(validation_ids)
        ]

        for row in filtered.itertuples(index=False):
            s1_id = row.source1_entity_id
            value = str(row.matched_entity_ids).strip()

            if value:
                gt[s1_id].update(
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                )

    return gt


def main():
    pred = pd.read_csv(
        PRED_PATH,
        sep="\t",
        dtype=str,
    ).fillna("")

    pred["probability"] = pred["probability"].astype(float)

    validation_ids = pred["entity_id"].unique()

    print(f"Validation S1 entities: {len(validation_ids):,}")

    gt = load_validation_ground_truth(validation_ids)

    positive_entity_count = sum(bool(v) for v in gt.values())
    zero_match_entity_count = len(gt) - positive_entity_count

    print(f"GT entities with matches: {positive_entity_count:,}")
    print(f"GT zero-match entities:   {zero_match_entity_count:,}")

    results = []

    for threshold in THRESHOLDS:
        macro_scores = []
        total_predicted = 0

        for entity_id in validation_ids:
            group = pred[pred["entity_id"] == entity_id]

            predicted_targets = set(
                group.loc[
                    group["probability"] >= threshold,
                    "target_entity_id",
                ]
            )

            total_predicted += len(predicted_targets)

            score = f05_from_sets(
                gt.get(entity_id, set()),
                predicted_targets,
            )

            macro_scores.append(score)

        results.append({
            "threshold": threshold,
            "entity_macro_f05": float(np.mean(macro_scores)),
            "avg_predictions_per_entity": (
                total_predicted / len(validation_ids)
            ),
        })

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df["entity_macro_f05"].idxmax()
    ]

    print()
    print("=" * 70)
    print("PROPER END-TO-END XGBOOST VALIDATION")
    print("=" * 70)
    print(f"Best threshold:           {best['threshold']:.2f}")
    print(f"Entity Macro F0.5:       {best['entity_macro_f05']:.6f}")
    print(
        f"Avg predictions/entity:  "
        f"{best['avg_predictions_per_entity']:.3f}"
    )

    print()
    print(results_df.to_string(index=False))

    output = ROOT / "experiments" / "models" / "xgboost_end_to_end_thresholds.tsv"
    results_df.to_csv(output, sep="\t", index=False)

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
