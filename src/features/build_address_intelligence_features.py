import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PAIR_PATH = ROOT / "experiments" / "training_pairs" / "training_pairs_v4.tsv"
BASE_FEATURE_PATH = ROOT / "data" / "processed" / "training_features_v4.tsv"

OUT_PATH = ROOT / "data" / "processed" / "training_features_address_intelligence.tsv"


# ---------------------------------------------------------
# Address token helpers
# ---------------------------------------------------------

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).upper().strip()

    # Normalize common separators
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("/", " ")
    text = text.replace(",", " ")

    return text


def numeric_tokens(text):
    """
    Extract numeric sequences and normalize leading zeros.

    Examples:
        '003633 MARKET PL' -> {'3633'}
        '3633 MARKET PLACE' -> {'3633'}
        '61-25 79 STREET' -> {'61', '25', '79'}
    """
    text = clean_text(text)

    nums = re.findall(r"\d+", text)

    normalized = set()

    for n in nums:
        try:
            normalized.add(str(int(n)))
        except ValueError:
            normalized.add(n.lstrip("0") or "0")

    return normalized


def compound_number_tokens(text):
    """
    Extract compound address numbers such as:

        61-25
        61-38D
        12-4A

    This helps distinguish different branches that share
    the same first numeric component.
    """
    text = clean_text(text)

    matches = re.findall(
        r"\b\d+[A-Z]?(?:-\d+[A-Z]?)\b",
        text
    )

    return set(matches)


def alnum_address_tokens(text):
    """
    Tokenize address into meaningful alphanumeric pieces.
    """
    text = clean_text(text)

    tokens = re.findall(
        r"[A-Z0-9]+(?:-[A-Z0-9]+)*",
        text
    )

    return set(tokens)


def postal_tokens(text):
    """
    Extract likely postal / ZIP codes.

    This intentionally remains generic because countries
    have different postal formats.
    """
    text = clean_text(text)

    tokens = re.findall(
        r"\b\d{4,6}\b",
        text
    )

    return {
        str(int(x)) if x.isdigit() else x
        for x in tokens
    }


def jaccard(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def overlap_min(a, b):
    """
    Intersection divided by size of smaller set.
    Useful when one address contains extra components.
    """
    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def exact_set_match(a, b):
    if not a or not b:
        return 0.0

    return float(a == b)


def first_numeric(text):
    nums = numeric_tokens(text)

    if not nums:
        return ""

    # Preserve the original order rather than set order
    raw = re.findall(r"\d+", clean_text(text))

    if not raw:
        return ""

    x = raw[0]

    try:
        return str(int(x))
    except ValueError:
        return x.lstrip("0") or "0"


# ---------------------------------------------------------
# Pair feature generation
# ---------------------------------------------------------

def build_address_intelligence(row):

    a1 = row["business_address"]
    a2 = row["target_business_address"]

    nums1 = numeric_tokens(a1)
    nums2 = numeric_tokens(a2)

    compounds1 = compound_number_tokens(a1)
    compounds2 = compound_number_tokens(a2)

    tokens1 = alnum_address_tokens(a1)
    tokens2 = alnum_address_tokens(a2)

    postal1 = postal_tokens(a1)
    postal2 = postal_tokens(a2)

    first1 = first_numeric(a1)
    first2 = first_numeric(a2)

    features = {

        # -------------------------------------------------
        # Numeric address structure
        # -------------------------------------------------

        "address_num_jaccard":
            jaccard(nums1, nums2),

        "address_num_overlap_min":
            overlap_min(nums1, nums2),

        "address_num_exact_set":
            exact_set_match(nums1, nums2),

        "address_num_count_diff":
            abs(len(nums1) - len(nums2)),

        # -------------------------------------------------
        # Compound house-number structure
        # -------------------------------------------------

        "address_compound_num_exact":
            float(bool(compounds1 & compounds2)),

        "address_compound_num_jaccard":
            jaccard(compounds1, compounds2),

        # -------------------------------------------------
        # General address token structure
        # -------------------------------------------------

        "address_alnum_token_jaccard":
            jaccard(tokens1, tokens2),

        "address_alnum_token_overlap_min":
            overlap_min(tokens1, tokens2),

        # -------------------------------------------------
        # First numeric component
        # -------------------------------------------------

        "address_first_num_exact_normalized":
            float(
                bool(first1)
                and bool(first2)
                and first1 == first2
            ),

        # -------------------------------------------------
        # Postal / ZIP evidence
        # -------------------------------------------------

        "address_postal_overlap":
            float(bool(postal1 & postal2)),
    }

    return features


def main():

    print("Loading training pairs...")
    pairs = pd.read_csv(
        PAIR_PATH,
        sep="\t",
        dtype=str
    )

    print(f"Training pairs: {len(pairs):,}")

    print("Loading existing 39-feature table...")
    base = pd.read_csv(
        BASE_FEATURE_PATH,
        sep="\t"
    )

    print(f"Base feature rows: {len(base):,}")

    # -----------------------------------------------------
    # Build address intelligence features
    # -----------------------------------------------------

    print("Building address intelligence features...")

    rows = []

    for _, row in pairs.iterrows():
        rows.append(build_address_intelligence(row))

    address_features = pd.DataFrame(rows)

    # Ensure same number of rows
    if len(address_features) != len(pairs):
        raise RuntimeError(
            "Address feature row count does not match training pairs."
        )

    # -----------------------------------------------------
    # Merge using pair IDs
    # -----------------------------------------------------

    feature_keys = [
        "entity_id",
        "target_entity_id",
    ]

    address_features = pd.concat(
        [
            pairs[feature_keys].reset_index(drop=True),
            address_features.reset_index(drop=True),
        ],
        axis=1
    )

    # Check duplicate pair keys
    dup_count = address_features.duplicated(
        feature_keys
    ).sum()

    print(f"Duplicate pair keys: {dup_count}")

    if dup_count:
        raise RuntimeError(
            "Duplicate entity_id + target_entity_id pairs detected."
        )

    # Remove accidental duplicate columns before merge
    new_feature_names = [
        c for c in address_features.columns
        if c not in feature_keys
    ]

    address_features = address_features[
        feature_keys + new_feature_names
    ]

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    result = base.merge(
        address_features,
        on=feature_keys,
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if len(result) != len(base):
        raise RuntimeError(
            "Row count changed after merge."
        )

    missing = result[new_feature_names].isna().sum().sum()

    print(f"New address features: {len(new_feature_names)}")
    print(f"Missing new feature values: {missing}")

    if missing:
        raise RuntimeError(
            "Missing address-intelligence feature values detected."
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    result.to_csv(
        OUT_PATH,
        sep="\t",
        index=False
    )

    print()
    print("=" * 60)
    print("ADDRESS INTELLIGENCE FEATURES COMPLETE")
    print("=" * 60)
    print(f"Output: {OUT_PATH}")
    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns):,}")

    print()
    print("New features:")
    for name in new_feature_names:
        print("  ", name)


if __name__ == "__main__":
    main()