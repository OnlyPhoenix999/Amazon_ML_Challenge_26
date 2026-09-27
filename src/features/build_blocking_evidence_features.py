from pathlib import Path
import pandas as pd

from src.preprocessing.benchmark_combo2_address_first_v4 import build_keys


ROOT = Path(__file__).resolve().parents[2]

PAIR_PATH = (
    ROOT
    / "experiments"
    / "training_pairs"
    / "training_pairs_v4.tsv"
)

BASE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_features_v4_augmented.tsv"
)

OUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_features_phase1_blocking.tsv"
)


KEY_NAMES = [
    "name_first",
    "name_prefix2",
    "address_number",
    "address_first",
]


def safe_key_match(k1, k2):
    return bool(k1) and bool(k2) and k1 == k2


def build_features(row):
    s1_keys = build_keys(
        row["business_name"],
        row["business_address"],
        row["country"],
        True,
    )

    target_keys = build_keys(
        row["target_business_name"],
        row["target_business_address"],
        row["target_country"],
        True,
    )

    matches = {
        key_name: safe_key_match(
            s1_keys.get(key_name),
            target_keys.get(key_name),
        )
        for key_name in KEY_NAMES
    }

    name_count = (
        int(matches["name_first"])
        + int(matches["name_prefix2"])
    )

    address_count = (
        int(matches["address_number"])
        + int(matches["address_first"])
    )

    total_count = name_count + address_count

    return {
        "v4_shared_key_count": total_count,
        "v4_shared_name_key_count": name_count,
        "v4_shared_address_key_count": address_count,

        "v4_name_first_match":
            int(matches["name_first"]),

        "v4_name_prefix2_match":
            int(matches["name_prefix2"]),

        "v4_address_number_match":
            int(matches["address_number"]),

        "v4_address_first_match":
            int(matches["address_first"]),
    }


def main():
    print("Loading training pairs...")
    pairs = pd.read_csv(
        PAIR_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Pairs: {len(pairs):,}")

    print("Loading 49-feature dataset...")
    base = pd.read_csv(
        BASE_PATH,
        sep="\t",
        keep_default_na=False,
    )

    print(f"Base rows: {len(base):,}")

    if len(base) != len(pairs):
        raise RuntimeError(
            f"Row mismatch: pairs={len(pairs):,}, "
            f"features={len(base):,}"
        )

    # ------------------------------------------------------
    # Build blocking evidence
    # ------------------------------------------------------

    rows = []

    for i, row in pairs.iterrows():

        if i > 0 and i % 50_000 == 0:
            print(f"Processed {i:,} pairs...")

        rows.append(build_features(row))

    blocking_df = pd.DataFrame(rows)

    # ------------------------------------------------------
    # Merge by row order
    # ------------------------------------------------------

    result = pd.concat(
        [
            base.reset_index(drop=True),
            blocking_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    expected_features = 49 + 7

    feature_cols = [
        c for c in result.columns
        if c not in {
            "entity_id",
            "target_entity_id",
            "label",
        }
    ]

    print(
        f"Total features: {len(feature_cols)}"
    )

    if len(feature_cols) != expected_features:
        raise RuntimeError(
            f"Expected {expected_features} features, "
            f"got {len(feature_cols)}"
        )

    if result[
        [
            "v4_shared_key_count",
            "v4_shared_name_key_count",
            "v4_shared_address_key_count",
            "v4_name_first_match",
            "v4_name_prefix2_match",
            "v4_address_number_match",
            "v4_address_first_match",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Missing blocking evidence values."
        )

    # ALL_V4 candidate pairs should have at least
    # one matching V4 key.
    if (result["v4_shared_key_count"] < 1).any():
        bad = int(
            (result["v4_shared_key_count"] < 1).sum()
        )

        raise RuntimeError(
            f"Found {bad:,} pairs that do not share "
            f"any ALL_V4 blocking key."
        )

    result.to_csv(
        OUT_PATH,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 70)
    print("PHASE 1 BLOCKING EVIDENCE COMPLETE")
    print("=" * 70)
    print(f"Output: {OUT_PATH}")
    print(f"Rows: {len(result):,}")
    print(f"Features: {len(feature_cols)}")

    print()
    print(
        result[
            "v4_shared_key_count"
        ].value_counts().sort_index()
    )


if __name__ == "__main__":
    main()
