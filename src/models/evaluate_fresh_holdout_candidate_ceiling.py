from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PAIR_PATH = ROOT / "experiments" / "training_pairs" / "training_pairs_v4.tsv"
GT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)
SPLIT_DIR = ROOT / "experiments" / "splits"

CHUNK_SIZE = 250000


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

        filtered = chunk[
            chunk["source1_entity_id"].isin(entity_ids)
        ]

        for row in filtered.itertuples(index=False):
            gt[row.source1_entity_id].update(
                x.strip()
                for x in str(row.matched_entity_ids).split(",")
                if x.strip()
            )

    return gt


def perfect_candidate_f05(true_targets, recovered_targets):
    true_targets = set(true_targets)
    recovered_targets = set(recovered_targets)

    # Perfect classifier predicts only recovered true matches.
    # Therefore FP = 0 and FN = missed GT.
    if not true_targets:
        return 1.0

    tp = len(recovered_targets)
    fn = len(true_targets - recovered_targets)

    # With FP=0:
    # precision = 1 whenever tp > 0
    # recall    = tp / (tp + fn)
    if tp == 0:
        return 0.0

    precision = 1.0
    recall = tp / (tp + fn)

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def main():
    holdout_ids = set(
        pd.read_csv(
            SPLIT_DIR / "holdout_entities.tsv",
            sep="\t",
            dtype=str,
        )["entity_id"]
    )

    gt = load_gt(holdout_ids)

    # ALL_V4 candidate pairs present in the existing training-pair file.
    pairs = pd.read_csv(
        PAIR_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "target_entity_id",
        ],
    ).fillna("")

    pairs = pairs[
        pairs["entity_id"].isin(holdout_ids)
    ]

    candidates = (
        pairs.groupby("entity_id")["target_entity_id"]
        .apply(set)
        .to_dict()
    )

    scores = []

    total_gt = 0
    total_recovered = 0
    missed_entities = 0

    for entity_id in sorted(holdout_ids):
        true_targets = gt[entity_id]
        candidate_targets = candidates.get(
            entity_id,
            set(),
        )

        recovered = true_targets & candidate_targets

        total_gt += len(true_targets)
        total_recovered += len(recovered)

        if true_targets - candidate_targets:
            missed_entities += 1

        scores.append(
            perfect_candidate_f05(
                true_targets,
                recovered,
            )
        )

    candidate_recall = (
        total_recovered / total_gt
        if total_gt
        else 1.0
    )

    ceiling = float(np.mean(scores))

    print("=" * 70)
    print("FRESH HOLDOUT ALL_V4 PERFECT-CLASSIFIER CEILING")
    print("=" * 70)
    print(f"Holdout entities:       {len(holdout_ids):,}")
    print(f"GT pairs:               {total_gt:,}")
    print(f"Recovered GT pairs:     {total_recovered:,}")
    print(f"Missed GT pairs:        {total_gt - total_recovered:,}")
    print(f"Pair candidate recall:  {candidate_recall:.6%}")
    print(f"Entities with GT miss:   {missed_entities:,}")
    print(f"Perfect-model ceiling:   {ceiling:.6f}")


if __name__ == "__main__":
    main()
