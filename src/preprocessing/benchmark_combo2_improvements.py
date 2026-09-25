"""
Benchmark improvements to COMBO_2.

IMPORTANT
---------
We are NOT introducing new blocking families.

We keep exactly the same COMBO_2 structure:

    name_first
    OR name_prefix2
    OR address_number
    OR address_first

We only improve how those existing keys are constructed.

Variants tested
---------------
BASELINE
    Original COMBO_2.

NAME_V2
    Improve name_first + name_prefix2 by removing common
    leading titles / prefixes.

ADDRESS_NUMBER_V2
    Improve address_number by canonicalizing leading zeros,
    slash-style numbers, and letter suffixes.

ADDRESS_FIRST_V2
    Improve address_first by removing common leading
    address labels / floor prefixes.

ALL_V2
    All of the above improvements together.

The goal is to measure:

    recall
    candidate volume
    marginal improvement over baseline
"""


# ============================================================
# IMPORTS
# ============================================================

import sys
import re
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.preprocessing.normalization import (
    normalize_business_name,
    normalize_address,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_DIR = ROOT / "data" / "raw" / "dataset" / "train"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

OUTPUT_DIR = ROOT / "experiments" / "candidate_generation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "combo2_improvement_benchmark.tsv"
)

S1_SAMPLE_SIZE = 2000
RANDOM_STATE = 42
CHUNK_SIZE = 100_000


# ============================================================
# ORIGINAL COMBO_2 KEY FUNCTIONS
# ============================================================

def first_token(text):
    if not text:
        return ""

    parts = str(text).split()

    return parts[0] if parts else ""


def first_two_chars(text):
    if not text:
        return ""

    compact = str(text).replace(" ", "")

    return compact[:2]


def first_address_number(text):
    """
    Original COMBO_2 behavior.
    """

    if not text:
        return ""

    match = re.search(r"\d+", str(text))

    if match:
        return match.group(0)

    return ""


# ============================================================
# IMPROVED NAME KEY
# ============================================================

# Based directly on the observed missed-pair examples.
#
# These are leading titles / presentation prefixes, NOT legal
# business suffixes.
#
# Legal suffixes remain untouched.

NAME_PREFIXES_TO_REMOVE = {
    "the",
    "dr",
    "mr",
    "mrs",
    "ms",
    "miss",
    "prof",
    "smt",
    "shri",
    "sri",
}


def clean_name_leading_prefixes(name):
    """
    Remove common leading personal/presentation prefixes.

    Examples
    --------
    "Dr Green Developers Private Limited"
        -> "green developers private limited"

    "The Professional Wealth Digital LLC"
        -> "professional wealth digital llc"

    "M/s Kaira Education Private Limited"
        -> "kaira education private limited"
    """

    name_norm = normalize_business_name(name)

    if not name_norm:
        return ""

    tokens = name_norm.split()

    while tokens:

        first = tokens[0]

        # Single-token prefixes.
        if first in NAME_PREFIXES_TO_REMOVE:
            tokens.pop(0)
            continue

        # Normalization may turn "M/s" into "m s".
        if len(tokens) >= 2:
            if tokens[0] == "m" and tokens[1] == "s":
                tokens = tokens[2:]
                continue

        break

    return " ".join(tokens)


def improved_name_first(name):
    """
    name_first_v2
    """

    cleaned = clean_name_leading_prefixes(name)

    return first_token(cleaned)


def improved_name_prefix2(name):
    """
    name_prefix2_v2
    """

    cleaned = clean_name_leading_prefixes(name)

    return first_two_chars(cleaned)


# ============================================================
# IMPROVED ADDRESS NUMBER
# ============================================================

ADDRESS_NUMBER_PATTERN = re.compile(
    r"\d+(?:\s*/\s*\d+)?[A-Za-z]?"
)


def canonicalize_address_number(number_text):
    """
    Canonicalize the numeric component of an address.

    Examples
    --------
    04B       -> 4b
    00302     -> 302
    005/8     -> 5/8
    02404     -> 2404
    """

    if not number_text:
        return ""

    value = str(number_text).strip().lower()

    # Remove whitespace around slash.
    value = re.sub(
        r"\s*/\s*",
        "/",
        value,
    )

    # Split slash-style numbers.
    if "/" in value:

        pieces = value.split("/")

        normalized = []

        for piece in pieces:

            match = re.match(
                r"(\d+)([a-z]?)$",
                piece,
            )

            if not match:
                normalized.append(piece)
                continue

            digits = match.group(1)
            suffix = match.group(2)

            digits = digits.lstrip("0") or "0"

            normalized.append(
                digits + suffix
            )

        return "/".join(normalized)

    # Normal numeric + optional letter suffix.
    match = re.match(
        r"(\d+)([a-z]?)$",
        value,
    )

    if match:

        digits = match.group(1)
        suffix = match.group(2)

        digits = digits.lstrip("0") or "0"

        return digits + suffix

    return value


def improved_address_number(address):
    """
    address_number_v2
    """

    address_norm = normalize_address(address)

    if not address_norm:
        return ""

    match = ADDRESS_NUMBER_PATTERN.search(
        address_norm
    )

    if not match:
        return ""

    return canonicalize_address_number(
        match.group(0)
    )


# ============================================================
# IMPROVED ADDRESS FIRST TOKEN
# ============================================================

ADDRESS_PREFIX_WORDS = {
    "door",
    "house",
    "flat",
    "plot",
    "unit",
    "suite",
    "apt",
    "apartment",
    "room",
}

FLOOR_WORDS = {
    "ground",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
}


def looks_like_floor_number(token):
    return bool(
        re.fullmatch(
            r"\d+(?:st|nd|rd|th)",
            token,
        )
    )


def looks_like_address_number(token):
    return bool(
        re.search(
            r"\d",
            token,
        )
    )


def improved_address_first(address):
    """
    address_first_v2

    Remove common structural prefixes before taking the
    first meaningful address token.

    Examples
    --------
    "Door No 04B, First Floor, Chokkalinganagar..."
        -> "chokkalinganagar"

    "H No 209, B.D. Chambers..."
        -> "b.d."

    "First Floor, Chokkalinganagar..."
        -> "chokkalinganagar"
    """

    address_norm = normalize_address(address)

    if not address_norm:
        return ""

    tokens = address_norm.split()

    changed = True

    while changed and tokens:

        changed = False

        # ----------------------------------------------------
        # "door no 123"
        # "house no 123"
        # "flat no 123"
        # "plot no 123"
        # "unit no 123"
        # ----------------------------------------------------

        if tokens[0] in ADDRESS_PREFIX_WORDS:

            tokens.pop(0)
            changed = True

            if tokens and tokens[0] == "no":
                tokens.pop(0)

                if tokens:
                    tokens.pop(0)

            continue

        # ----------------------------------------------------
        # "h no 123"
        # "h.no 123" after normalization
        # ----------------------------------------------------

        if tokens[0] == "h":

            tokens.pop(0)
            changed = True

            if tokens and tokens[0] == "no":
                tokens.pop(0)

            if tokens:
                tokens.pop(0)

            continue

        # ----------------------------------------------------
        # "no 123"
        # ----------------------------------------------------

        if tokens[0] == "no":

            tokens.pop(0)
            changed = True

            if tokens:
                tokens.pop(0)

            continue

        # ----------------------------------------------------
        # "first floor"
        # "second floor"
        # "ground floor"
        # "1st floor"
        # ----------------------------------------------------

        if (
            len(tokens) >= 2
            and tokens[0] in FLOOR_WORDS
            and tokens[1] == "floor"
        ):

            tokens = tokens[2:]
            changed = True
            continue

        if (
            len(tokens) >= 2
            and looks_like_floor_number(tokens[0])
            and tokens[1] == "floor"
        ):

            tokens = tokens[2:]
            changed = True
            continue

    return tokens[0] if tokens else ""


# ============================================================
# KEY BUILDERS
# ============================================================

def build_keys(
    name,
    address,
    country,
    variant,
):
    """
    Build the four COMBO_2 keys for a specific variant.
    """

    name_norm = normalize_business_name(name)
    address_norm = normalize_address(address)

    country_norm = str(country).strip().casefold()

    # -------------------------
    # NAME KEYS
    # -------------------------

    if variant in {
        "NAME_V2",
        "ALL_V2",
    }:

        name_first_value = (
            improved_name_first(name)
        )

        name_prefix2_value = (
            improved_name_prefix2(name)
        )

    else:

        name_first_value = (
            first_token(name_norm)
        )

        name_prefix2_value = (
            first_two_chars(name_norm)
        )

    # -------------------------
    # ADDRESS NUMBER
    # -------------------------

    if variant in {
        "ADDRESS_NUMBER_V2",
        "ALL_V2",
    }:

        address_number_value = (
            improved_address_number(address)
        )

    else:

        address_number_value = (
            first_address_number(address_norm)
        )

    # -------------------------
    # ADDRESS FIRST
    # -------------------------

    if variant in {
        "ADDRESS_FIRST_V2",
        "ALL_V2",
    }:

        address_first_value = (
            improved_address_first(address)
        )

    else:

        address_first_value = (
            first_token(address_norm)
        )

    return {
        "name_first": (
            country_norm,
            name_first_value,
        ),

        "name_prefix2": (
            country_norm,
            name_prefix2_value,
        ),

        "address_number": (
            country_norm,
            address_number_value,
        ),

        "address_first": (
            country_norm,
            address_first_value,
        ),
    }


# ============================================================
# GROUND TRUTH
# ============================================================

def parse_ground_truth(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    benchmark_start = time.time()

    print("\n" + "=" * 70)
    print("COMBO_2 IMPROVEMENT BENCHMARK")
    print("=" * 70)

    variants = [
        "BASELINE",
        "NAME_V2",
        "ADDRESS_NUMBER_V2",
        "ADDRESS_FIRST_V2",
        "ALL_V2",
    ]

    for variant in variants:
        print(f"  {variant}")

    print(f"\nSample size: {S1_SAMPLE_SIZE:,}")
    print(f"Chunk size:  {CHUNK_SIZE:,}")

    # ========================================================
    # INPUT CHECK
    # ========================================================

    print("\nChecking input files...")

    for path in [
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
        GROUND_TRUTH_PATH,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

    print("Input files verified.")

    # ========================================================
    # SOURCE1
    # ========================================================

    print("\nLoading Source1...")

    source1 = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    source1_sample = source1.sample(
        n=min(
            S1_SAMPLE_SIZE,
            len(source1),
        ),
        random_state=RANDOM_STATE,
    ).copy()

    source1_sample.reset_index(
        drop=True,
        inplace=True,
    )

    sampled_s1_ids = set(
        source1_sample["entity_id"]
        .astype(str)
    )

    print(
        f"Total Source1 rows: "
        f"{len(source1):,}"
    )

    print(
        f"Sampled Source1 entities: "
        f"{len(source1_sample):,}"
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    print("\nLoading ground truth...")

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    target_to_s1 = defaultdict(set)

    total_true_matches = 0

    for _, row in ground_truth.iterrows():

        s1_id = str(
            row["source1_entity_id"]
        )

        if s1_id not in sampled_s1_ids:
            continue

        true_targets = parse_ground_truth(
            row["matched_entity_ids"]
        )

        for target_id in true_targets:

            target_to_s1[target_id].add(
                s1_id
            )

            total_true_matches += 1

    print(
        f"Total true matches in sample: "
        f"{total_true_matches:,}"
    )

    # ========================================================
    # BUILD INDEXES
    # ========================================================

    print("\nBuilding Source1 indexes...")

    # indexes[variant][key_name][key_value]
    # -> set(S1 IDs)

    indexes = {
        variant: {
            "name_first": defaultdict(set),
            "name_prefix2": defaultdict(set),
            "address_number": defaultdict(set),
            "address_first": defaultdict(set),
        }
        for variant in variants
    }

    for _, row in source1_sample.iterrows():

        s1_id = str(
            row["entity_id"]
        )

        for variant in variants:

            keys = build_keys(
                row["business_name"],
                row["business_address"],
                row["country"],
                variant,
            )

            for key_name, key_value in keys.items():

                if not key_value[1]:
                    continue

                indexes[
                    variant
                ][
                    key_name
                ][
                    key_value
                ].add(s1_id)

    print("Indexes created.")

    # ========================================================
    # METRICS
    # ========================================================

    candidate_counts = {
        variant: 0
        for variant in variants
    }

    recovered_true = {
        variant: defaultdict(set)
        for variant in variants
    }

    # ========================================================
    # TARGET SCAN
    # ========================================================

    target_files = [
        ("source2", SOURCE2_PATH),
        ("source3", SOURCE3_PATH),
    ]

    scan_start = time.time()

    for source_name, source_path in target_files:

        print("\n" + "-" * 70)
        print(
            f"Scanning {source_name}: "
            f"{source_path.name}"
        )
        print("-" * 70)

        processed = 0
        source_start = time.time()

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ):

            processed += len(chunk)

            for _, row in chunk.iterrows():

                target_id = str(
                    row["entity_id"]
                )

                true_s1_ids = (
                    target_to_s1.get(
                        target_id
                    )
                )

                # Build keys only once per variant.
                variant_keys = {}

                for variant in variants:

                    variant_keys[
                        variant
                    ] = build_keys(
                        row["business_name"],
                        row["business_address"],
                        row["country"],
                        variant,
                    )

                # --------------------------------------------
                # Evaluate every variant
                # --------------------------------------------

                for variant in variants:

                    keys = variant_keys[variant]

                    candidate_s1_ids = set()

                    for key_name in (
                        "name_first",
                        "name_prefix2",
                        "address_number",
                        "address_first",
                    ):

                        key_value = keys[
                            key_name
                        ]

                        if not key_value[1]:
                            continue

                        candidate_s1_ids.update(
                            indexes[
                                variant
                            ][
                                key_name
                            ].get(
                                key_value,
                                set(),
                            )
                        )

                    if not candidate_s1_ids:
                        continue

                    candidate_counts[
                        variant
                    ] += len(
                        candidate_s1_ids
                    )

                    # ----------------------------------------
                    # Recover true pairs
                    # ----------------------------------------

                    if true_s1_ids:

                        for s1_id in (
                            candidate_s1_ids
                            & true_s1_ids
                        ):

                            recovered_true[
                                variant
                            ][
                                s1_id
                            ].add(
                                target_id
                            )

            print(
                f"\r{source_name}: "
                f"{processed:,} rows",
                end="",
                flush=True,
            )

        print(
            f"\nFinished {source_name} "
            f"in {time.time() - source_start:.2f}s"
        )

    print(
        f"\nTotal target scan time: "
        f"{time.time() - scan_start:.2f}s"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    baseline_candidates = candidate_counts[
        "BASELINE"
    ]

    baseline_recovered = sum(
        len(v)
        for v in recovered_true[
            "BASELINE"
        ].values()
    )

    baseline_recall = (
        baseline_recovered
        / total_true_matches
        * 100
    )

    for variant in variants:

        recovered = sum(
            len(v)
            for v in recovered_true[
                variant
            ].values()
        )

        recall = (
            recovered
            / total_true_matches
            * 100
            if total_true_matches
            else 0
        )

        candidates = candidate_counts[
            variant
        ]

        avg_candidates = (
            candidates
            / len(source1_sample)
        )

        reduction_vs_baseline = (
            1
            - candidates
            / baseline_candidates
        ) * 100 if baseline_candidates else 0

        recall_gain_vs_baseline = (
            recall
            - baseline_recall
        )

        results.append({
            "variant": variant,

            "candidate_recall": recall,

            "recall_gain_vs_baseline_pp":
                recall_gain_vs_baseline,

            "total_candidates":
                candidates,

            "avg_candidates_per_source1":
                avg_candidates,

            "candidate_change_vs_baseline_pct":
                -reduction_vs_baseline,
        })

    results_df = pd.DataFrame(results)

    # Put baseline first, then highest recall.
    results_df["_sort"] = (
        results_df["variant"]
        .eq("BASELINE")
        .map({True: 0, False: 1})
    )

    results_df = (
        results_df
        .sort_values(
            [
                "_sort",
                "candidate_recall",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop(
            columns="_sort"
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 70)
    print("COMBO_2 IMPROVEMENT RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    print(
        f"\nBaseline recall: "
        f"{baseline_recall:.2f}%"
    )

    for _, row in results_df.iterrows():

        if row["variant"] == "BASELINE":
            continue

        print(
            f"\n{row['variant']}:"
            f"\n  recall: "
            f"{row['candidate_recall']:.2f}%"
            f"\n  recall gain: "
            f"{row['recall_gain_vs_baseline_pp']:+.2f} pp"
            f"\n  candidates: "
            f"{int(row['total_candidates']):,}"
            f"\n  avg/S1: "
            f"{row['avg_candidates_per_source1']:,.2f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nTotal runtime: "
        f"{time.time() - benchmark_start:.2f}s"
    )


if __name__ == "__main__":
    main()