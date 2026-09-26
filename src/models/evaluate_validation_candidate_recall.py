from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PRED_PATH = ROOT / "experiments" / "models" / "xgboost_validation_predictions.tsv"
PAIRS_PATH = ROOT / "experiments" / "training_pairs" / "training_pairs_v4.tsv"
GT_PATH = ROOT / "data" / "raw" / "dataset" / "train" / "train_ground_truth.tsv"

CHUNK_SIZE = 250000


def load_gt(validation_ids):
    validation_ids = set(validation_ids)
    gt = {x: set() for x in validation_ids}

    for chunk in pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):
        chunk = chunk.fillna("")
        chunk = chunk[
            chunk["source1_entity_id"].isin(validation_ids)
        ]

        for row in chunk.itertuples(index=False):
            ids = [
                x.strip()
                for x in str(row.matched_entity_ids).split(",")
                if x.strip()
            ]
            gt[row.source1_entity_id].update(ids)

    return gt


def main():
    pred = pd.read_csv(
        PRED_PATH,
        sep="\t",
        dtype=str,
    )

    validation_ids = set(pred["entity_id"].unique())

    gt = load_gt(validation_ids)

    pairs = pd.read_csv(
        PAIRS_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "target_entity_id",
        ],
    ).fillna("")

    pairs = pairs[
        pairs["entity_id"].isin(validation_ids)
    ]

    candidates = (
        pairs.groupby("entity_id")["target_entity_id"]
        .apply(set)
        .to_dict()
    )

    rows = []

    total_gt = 0
    total_recovered = 0

    for entity_id in sorted(validation_ids):
        true_ids = gt.get(entity_id, set())
        candidate_ids = candidates.get(entity_id, set())

        recovered = true_ids & candidate_ids

        total_gt += len(true_ids)
        total_recovered += len(recovered)

        rows.append({
            "entity_id": entity_id,
            "gt_count": len(true_ids),
            "candidate_count": len(candidate_ids),
            "recovered_gt_count": len(recovered),
            "missed_gt_count": len(true_ids - candidate_ids),
            "candidate_recall": (
                len(recovered) / len(true_ids)
                if true_ids else 1.0
            ),
        })

    result = pd.DataFrame(rows)

    print("=" * 70)
    print("VALIDATION CANDIDATE RECALL")
    print("=" * 70)

    print(f"S1 entities:           {len(result):,}")
    print(f"GT pairs:               {total_gt:,}")
    print(f"Recovered GT pairs:     {total_recovered:,}")
    print(f"Missed GT pairs:        {total_gt - total_recovered:,}")

    print(
        f"Pair candidate recall:  "
        f"{total_recovered / total_gt:.6%}"
        if total_gt
        else "Pair candidate recall: N/A"
    )

    print()
    print(
        f"Entities with any GT miss: "
        f"{(result['missed_gt_count'] > 0).sum():,}"
    )

    print(
        f"Entities with complete candidate recall: "
        f"{(result['missed_gt_count'] == 0).sum():,}"
    )

    output = ROOT / "experiments" / "models" / "validation_candidate_recall.tsv"
    result.to_csv(output, sep="\t", index=False)

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
