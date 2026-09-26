from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

GT_PATH = ROOT / "data" / "raw" / "dataset" / "train" / "train_ground_truth.tsv"
PRED_DIR = ROOT / "experiments" / "models"

CHUNK_SIZE = 250000
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


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

    if precision == 0 or recall == 0:
        return 0.0

    return 1.25 * precision * recall / (
        0.25 * precision + recall
    )


def evaluate(path, gt):
    pred = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )

    pred["probability"] = pred["probability"].astype(float)

    entities = pred["entity_id"].unique()

    best_score = -1
    best_threshold = None
    best_avg = None

    for threshold in THRESHOLDS:
        scores = []
        prediction_count = 0

        for entity_id in entities:
            group = pred[pred["entity_id"] == entity_id]

            targets = set(
                group.loc[
                    group["probability"] >= threshold,
                    "target_entity_id",
                ]
            )

            prediction_count += len(targets)

            scores.append(
                entity_f05(
                    gt.get(entity_id, set()),
                    targets,
                )
            )

        score = float(np.mean(scores))

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_avg = prediction_count / len(entities)

    return best_threshold, best_score, best_avg


def main():
    files = {
        "base39": PRED_DIR / "xgboost_base39_validation_predictions.tsv",
        "relationship45": PRED_DIR / "xgboost_relationship45_validation_predictions.tsv",
        "suffixconflict43": PRED_DIR / "xgboost_suffixconflict43_validation_predictions.tsv",
        "full49": PRED_DIR / "xgboost_full49_validation_predictions.tsv",
    }

    first = next(iter(files.values()))

    validation_ids = pd.read_csv(
        first,
        sep="\t",
        dtype=str,
        usecols=["entity_id"],
    )["entity_id"].unique()

    gt = load_gt(validation_ids)

    print("=" * 70)
    print("XGBOOST FEATURE ABLATION")
    print("=" * 70)

    results = []

    for name, path in files.items():
        threshold, score, avg = evaluate(path, gt)

        results.append({
            "variant": name,
            "threshold": threshold,
            "entity_macro_f05": score,
            "avg_predictions_per_entity": avg,
        })

        print(
            f"{name:18s} | "
            f"threshold={threshold:.2f} | "
            f"Macro F0.5={score:.6f} | "
            f"avg_pred/entity={avg:.3f}"
        )

    pd.DataFrame(results).to_csv(
        PRED_DIR / "xgboost_feature_ablation_summary.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
