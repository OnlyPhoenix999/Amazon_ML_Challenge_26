import sys
from pathlib import Path
import random

import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

S1_PATH = ROOT / "data/raw/dataset/train/train_source1.tsv"
S2_PATH = ROOT / "data/raw/dataset/train/train_source2.tsv"
S3_PATH = ROOT / "data/raw/dataset/train/train_source3.tsv"
GT_PATH = ROOT / "data/raw/dataset/train/train_ground_truth.tsv"

OUTPUT_PATH = ROOT / "data/processed/training_features_sample.tsv"

# ============================================================
# IMPORT FEATURE ENGINEERING
# ============================================================

sys.path.insert(0, str(ROOT))

from src.features.similarity_features import build_pair_features


# ============================================================
# SETTINGS
# ============================================================

S1_SAMPLE_SIZE = 1000

# Keep this small for laptop
NEGATIVES_PER_S1 = 5

# Number of target rows kept as negative pool
NEGATIVE_POOL_SIZE = 10000

CHUNK_SIZE = 250000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def read_first_rows(path, n):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=n
    ).fillna("")


def load_ground_truth_for_ids(selected_ids):
    """
    Read GT in chunks and keep only selected S1 entities.
    """

    print("Scanning ground truth...")

    selected_ids = set(selected_ids)
    matches = {}

    for chunk in pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        chunk = chunk.fillna("")

        filtered = chunk[
            chunk["source1_entity_id"].isin(selected_ids)
        ]

        for _, row in filtered.iterrows():

            s1_id = row["source1_entity_id"]
            value = row["matched_entity_ids"]

            if value.strip():
                matches[s1_id] = [
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                ]
            else:
                matches[s1_id] = []

    return matches


def load_target_records(target_path, required_ids):
    """
    Scan target dataset in chunks and retrieve only required IDs.
    """

    required_ids = set(required_ids)
    records = {}

    print(f"Scanning {target_path.name} for positive records...")

    for chunk in pd.read_csv(
        target_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        chunk = chunk.fillna("")

        filtered = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in filtered.iterrows():
            records[row["entity_id"]] = row.to_dict()

        if len(records) >= len(required_ids):
            break

    return records


def make_pair(
    s1_record,
    target_record,
    source,
    label
):
    """
    Build one training example.
    """

    features = build_pair_features(
        s1_record["business_name"],
        s1_record["business_address"],
        s1_record["country"],

        target_record["business_name"],
        target_record["business_address"],
        target_record["country"],

        source
    )

    features["s1_entity_id"] = s1_record["entity_id"]
    features["target_entity_id"] = target_record["entity_id"]
    features["label"] = label

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading small Source 1 sample...")

    # IMPORTANT:
    # Only first 1000 S1 rows.
    # No 2.2M-row dataframe.
    s1 = read_first_rows(
        S1_PATH,
        S1_SAMPLE_SIZE
    )

    print(f"Selected {len(s1):,} Source 1 entities.")

    selected_s1_ids = s1["entity_id"].tolist()

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt = load_ground_truth_for_ids(
        selected_s1_ids
    )

    print(
        f"Found ground truth for "
        f"{len(gt):,} Source 1 entities."
    )

    # --------------------------------------------------------
    # Collect positive target IDs
    # --------------------------------------------------------

    s2_positive_ids = set()
    s3_positive_ids = set()

    for ids in gt.values():

        for entity_id in ids:

            if entity_id.startswith("S2-"):
                s2_positive_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                s3_positive_ids.add(entity_id)

    print(
        f"S2 positive IDs: {len(s2_positive_ids):,}"
    )

    print(
        f"S3 positive IDs: {len(s3_positive_ids):,}"
    )

    # --------------------------------------------------------
    # Load ONLY positive records
    # --------------------------------------------------------

    s2_positive = load_target_records(
        S2_PATH,
        s2_positive_ids
    )

    s3_positive = load_target_records(
        S3_PATH,
        s3_positive_ids
    )

    print(
        f"Loaded S2 positive records: "
        f"{len(s2_positive):,}"
    )

    print(
        f"Loaded S3 positive records: "
        f"{len(s3_positive):,}"
    )

    # --------------------------------------------------------
    # Small negative pools
    # --------------------------------------------------------

    print("\nLoading small negative pools...")

    s2_pool = read_first_rows(
        S2_PATH,
        NEGATIVE_POOL_SIZE
    )

    s3_pool = read_first_rows(
        S3_PATH,
        NEGATIVE_POOL_SIZE
    )

    print(
        f"S2 negative pool: {len(s2_pool):,}"
    )

    print(
        f"S3 negative pool: {len(s3_pool):,}"
    )

    # --------------------------------------------------------
    # Generate training rows
    # --------------------------------------------------------

    print("\nGenerating training pairs...")

    training_rows = []

    positive_count = 0
    negative_count = 0

    for idx, s1_row in s1.iterrows():

        s1_record = s1_row.to_dict()

        s1_id = s1_record["entity_id"]

        positive_ids = set(gt.get(s1_id, []))

        # ====================================================
        # POSITIVES
        # ====================================================

        for target_id in positive_ids:

            if target_id.startswith("S2-"):

                target_record = s2_positive.get(
                    target_id
                )

                if target_record is not None:

                    row = make_pair(
                        s1_record,
                        target_record,
                        "S2",
                        1
                    )

                    training_rows.append(row)
                    positive_count += 1

            elif target_id.startswith("S3-"):

                target_record = s3_positive.get(
                    target_id
                )

                if target_record is not None:

                    row = make_pair(
                        s1_record,
                        target_record,
                        "S3",
                        1
                    )

                    training_rows.append(row)
                    positive_count += 1

        # ====================================================
        # NEGATIVES
        # ====================================================

        # 3 S2 negatives
        s2_candidates = s2_pool[
            ~s2_pool["entity_id"].isin(positive_ids)
        ]

        # 2 S3 negatives
        s3_candidates = s3_pool[
            ~s3_pool["entity_id"].isin(positive_ids)
        ]

        n2 = min(3, len(s2_candidates))
        n3 = min(2, len(s3_candidates))

        if n2 > 0:

            sampled_s2 = s2_candidates.sample(
                n=n2,
                random_state=RANDOM_SEED + idx
            )

            for _, target_row in sampled_s2.iterrows():

                target_record = target_row.to_dict()

                row = make_pair(
                    s1_record,
                    target_record,
                    "S2",
                    0
                )

                training_rows.append(row)
                negative_count += 1

        if n3 > 0:

            sampled_s3 = s3_candidates.sample(
                n=n3,
                random_state=RANDOM_SEED + idx
            )

            for _, target_row in sampled_s3.iterrows():

                target_record = target_row.to_dict()

                row = make_pair(
                    s1_record,
                    target_record,
                    "S3",
                    0
                )

                training_rows.append(row)
                negative_count += 1

        # Progress
        if (idx + 1) % 100 == 0:

            print(
                f"Processed {idx + 1:,}/"
                f"{len(s1):,} S1 entities..."
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df = pd.DataFrame(training_rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n========================================")
    print("TRAINING SAMPLE COMPLETE")
    print("========================================")

    print(f"Rows:       {len(df):,}")
    print(f"Positives:  {positive_count:,}")
    print(f"Negatives:  {negative_count:,}")
    print(f"Features:   {len(df.columns):,}")

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()