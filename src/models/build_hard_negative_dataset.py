from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = ROOT / "data" / "processed" / "training_features_v4.tsv"
VAL_PRED_PATH = ROOT / "experiments" / "models" / "xgboost_validation_predictions.tsv"
OUTPUT_PATH = ROOT / "data" / "processed" / "training_features_v4_hardneg.tsv"

RANDOM_STATE = 42
TOTAL_NEGATIVES = 160000
HARD_NEGATIVES = 40000

rng = np.random.default_rng(RANDOM_STATE)


def main():
    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t",
        dtype={
            "entity_id": str,
            "target_entity_id": str,
            "label": int,
        },
    )

    validation_entities = set(
        pd.read_csv(
            VAL_PRED_PATH,
            sep="\t",
            dtype=str,
            usecols=["entity_id"],
        )["entity_id"].unique()
    )

    train = df[~df["entity_id"].isin(validation_entities)].copy()

    positives = train[train["label"] == 1].copy()
    negatives = train[train["label"] == 0].copy()

    print("=" * 70)
    print("HARD NEGATIVE MINING")
    print("=" * 70)
    print(f"Total feature rows:       {len(df):,}")
    print(f"Validation S1 entities:   {len(validation_entities):,}")
    print(f"Training rows:             {len(train):,}")
    print(f"Training positives:        {len(positives):,}")
    print(f"Training negatives:        {len(negatives):,}")

    # ------------------------------------------------------------
    # Hardness score
    # ------------------------------------------------------------

    negatives["_hard_score"] = (
        0.35 * negatives["norm_name_ratio"]
        + 0.35 * negatives["norm_address_ratio"]
        + 0.15 * negatives["name_no_suffix_wratio"]
        + 0.15 * negatives["norm_address_token_set_ratio"]
    )

    # ------------------------------------------------------------
    # Hard-negative buckets
    # ------------------------------------------------------------

    buckets = {}

    buckets["A_high_name_bad_address"] = (
        (negatives["norm_name_ratio"] >= 85)
        & (negatives["norm_address_ratio"] <= 70)
    )

    buckets["B_high_address_weak_name"] = (
        (negatives["norm_address_token_set_ratio"] >= 85)
        & (negatives["norm_name_ratio"] <= 60)
    )

    buckets["C_no_suffix_high_name_bad_address"] = (
        (negatives["name_no_suffix_wratio"] >= 90)
        & (negatives["norm_address_ratio"] <= 70)
    )

    buckets["D_high_name_number_conflict"] = (
        (negatives["norm_name_ratio"] >= 85)
        & (negatives["address_number_match"] == 0)
        & (negatives["address1_missing"] == 0)
        & (negatives["address2_missing"] == 0)
    )

    bucket_frames = []

    print()
    print("Bucket counts:")

    for name, mask in buckets.items():
        subset = negatives.loc[mask].copy()
        subset["_hard_bucket"] = name
        bucket_frames.append(subset)

        print(f"{name:35s}: {len(subset):>7,}")

    # ------------------------------------------------------------
    # Union of all bucketed negatives
    # ------------------------------------------------------------

    hard_pool = pd.concat(
        bucket_frames,
        ignore_index=False,
    ).drop_duplicates(
        subset=["entity_id", "target_entity_id"]
    )

    hard_pool = hard_pool.sort_values(
        "_hard_score",
        ascending=False,
    )

    print()
    print(f"Unique hard-negative pool: {len(hard_pool):,}")

    # ------------------------------------------------------------
    # Select hard negatives
    # ------------------------------------------------------------

    hard_take = min(
        HARD_NEGATIVES,
        len(hard_pool),
    )

    hard_selected = hard_pool.head(hard_take).copy()

    # Remaining negatives are sampled randomly so total negative
    # count stays equal to the original training set.
    hard_keys = set(
        zip(
            hard_selected["entity_id"],
            hard_selected["target_entity_id"],
        )
    )

    remaining = negatives[
        ~negatives.apply(
            lambda r: (
                r["entity_id"],
                r["target_entity_id"],
            ) in hard_keys,
            axis=1,
        )
    ].copy()

    ordinary_take = TOTAL_NEGATIVES - len(hard_selected)

    if ordinary_take > len(remaining):
        ordinary_take = len(remaining)

    ordinary_selected = remaining.sample(
        n=ordinary_take,
        random_state=RANDOM_STATE,
    )

    selected_negatives = pd.concat(
        [
            hard_selected,
            ordinary_selected,
        ],
        ignore_index=True,
    )

    # Keep exact original negative count.
    selected_negatives = selected_negatives.head(
        TOTAL_NEGATIVES
    )

    final_df = pd.concat(
        [
            positives,
            selected_negatives,
        ],
        ignore_index=True,
    )

    final_df = final_df.drop(
        columns=["_hard_score", "_hard_bucket"],
        errors="ignore",
    )

    final_df = final_df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 70)
    print("HARD-NEGATIVE DATASET")
    print("=" * 70)
    print(f"Positives:              {len(positives):,}")
    print(f"Hard negatives:         {len(hard_selected):,}")
    print(f"Ordinary negatives:     {len(ordinary_selected):,}")
    print(f"Final rows:             {len(final_df):,}")
    print(f"Output:                 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

