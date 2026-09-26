from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PAIR_PATH = ROOT / "experiments" / "training_pairs" / "training_pairs_v4.tsv"
BASE_FEATURE_PATH = ROOT / "data" / "processed" / "training_features_v4.tsv"
OUTPUT_PATH = ROOT / "data" / "processed" / "training_features_v4_augmented.tsv"

KEYS = [
    "entity_id",
    "target_entity_id",
]

sys.path.insert(0, str(ROOT))

from src.preprocessing.normalization import normalize_record


def first_number(text):
    text = "" if text is None else str(text)
    match = re.search(r"\d+", text)
    return match.group(0) if match else ""


def build_extra_features(row):
    name_sim = float(row["norm_name_ratio"])
    addr_sim = float(row["norm_address_ratio"])

    name_addr_gap = name_sim - addr_sim

    extra = {
        "name_address_mean": (name_sim + addr_sim) / 2.0,
        "name_address_min": min(name_sim, addr_sim),
        "name_address_max": max(name_sim, addr_sim),
        "name_address_gap": name_addr_gap,
        "name_dominance": max(name_addr_gap, 0.0),
        "address_dominance": max(-name_addr_gap, 0.0),
        "suffix_both_present": 0,
        "suffix_same": 0,
        "suffix_conflict": 0,
        "address_number_conflict": 0,
    }

    record1 = normalize_record(
        row["business_name"],
        row["business_address"],
        row["country"],
    )

    record2 = normalize_record(
        row["target_business_name"],
        row["target_business_address"],
        row["target_country"],
    )

    suffix1 = str(
        record1.get("legal_suffix", "") or ""
    ).strip()

    suffix2 = str(
        record2.get("legal_suffix", "") or ""
    ).strip()

    both_suffix = bool(suffix1 and suffix2)

    extra["suffix_both_present"] = int(both_suffix)
    extra["suffix_same"] = int(
        both_suffix and suffix1 == suffix2
    )
    extra["suffix_conflict"] = int(
        both_suffix and suffix1 != suffix2
    )

    number1 = first_number(row["business_address"])
    number2 = first_number(row["target_business_address"])

    extra["address_number_conflict"] = int(
        bool(number1)
        and bool(number2)
        and number1 != number2
    )

    return extra


def main():
    print("Loading base features...")

    features = pd.read_csv(
        BASE_FEATURE_PATH,
        sep="\t",
        dtype={
            "entity_id": str,
            "target_entity_id": str,
            "label": int,
        },
    )

    print("Loading training pairs...")

    pairs = pd.read_csv(
        PAIR_PATH,
        sep="\t",
        dtype=str,
    ).fillna("")

    # The base feature table has one row per pair.
    if features[KEYS].duplicated().any():
        duplicate_count = int(
            features[KEYS].duplicated().sum()
        )
        raise RuntimeError(
            f"Base feature table contains "
            f"{duplicate_count:,} duplicate pair keys."
        )

    if pairs[KEYS].duplicated().any():
        duplicate_count = int(
            pairs[KEYS].duplicated().sum()
        )
        raise RuntimeError(
            f"Training-pair table contains "
            f"{duplicate_count:,} duplicate pair keys."
        )

    # Keep only the fields needed for the additional features.
    pair_columns = [
        "entity_id",
        "target_entity_id",
        "target_source",
        "business_name",
        "business_address",
        "country",
        "target_business_name",
        "target_business_address",
        "target_country",
    ]

    pair_lookup = pairs[pair_columns].set_index(KEYS)

    if len(pair_lookup) != len(features):
        raise RuntimeError(
            f"Row/key mismatch: feature rows={len(features):,}, "
            f"pair rows={len(pair_lookup):,}"
        )

    feature_keys = set(
        map(tuple, features[KEYS].to_numpy())
    )

    pair_keys = set(pair_lookup.index)

    if feature_keys != pair_keys:
        missing_from_pairs = feature_keys - pair_keys
        missing_from_features = pair_keys - feature_keys

        raise RuntimeError(
            "Pair keys do not match. "
            f"Missing from pairs={len(missing_from_pairs):,}; "
            f"missing from features={len(missing_from_features):,}"
        )

    extras = []

    total = len(features)

    print("Building additional features...")

    for i, row in enumerate(
        features.itertuples(index=False),
        start=1,
    ):
        pair = pair_lookup.loc[
            (
                row.entity_id,
                row.target_entity_id,
            )
        ]

        extra_input = {
            "business_name": pair["business_name"],
            "business_address": pair["business_address"],
            "country": pair["country"],
            "target_business_name": pair["target_business_name"],
            "target_business_address": pair["target_business_address"],
            "target_country": pair["target_country"],
            "norm_name_ratio": row.norm_name_ratio,
            "norm_address_ratio": row.norm_address_ratio,
        }

        extras.append(
            build_extra_features(extra_input)
        )

        if i % 25000 == 0:
            print(f"Processed {i:,}/{total:,}")

    extra_df = pd.DataFrame(extras)

    if len(extra_df) != len(features):
        raise RuntimeError(
            f"Row mismatch: expected {len(features):,}, "
            f"got {len(extra_df):,}"
        )

    augmented = pd.concat(
        [
            features.reset_index(drop=True),
            extra_df.reset_index(drop=True),
        ],
        axis=1,
    )

    if augmented.columns.duplicated().any():
        duplicated = augmented.columns[
            augmented.columns.duplicated()
        ].tolist()

        raise RuntimeError(
            f"Duplicate output columns: {duplicated}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    augmented.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 70)
    print("AUGMENTED FEATURE TABLE")
    print("=" * 70)
    print(f"Rows:           {len(augmented):,}")
    print("Original:       39")
    print(f"Added:          {len(extra_df.columns):,}")
    print(f"Total features: {len(augmented.columns) - 3:,}")
    print(
        f"Missing values: "
        f"{int(augmented.isna().sum().sum()):,}"
    )
    print(f"Output:         {OUTPUT_PATH}")

    print()
    print("New columns:")
    for column in extra_df.columns:
        print(f"  {column}")


if __name__ == "__main__":
    main()
