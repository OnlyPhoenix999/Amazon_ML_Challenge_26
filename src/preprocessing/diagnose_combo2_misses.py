"""
Diagnose missed true matches from COMBO_2.

COMBO_2
-------
country + name_first
OR
country + name_prefix2
OR
country + address_number
OR
country + address_first

This script DOES NOT introduce new blocking strategies.

It only analyzes true matches that COMBO_2 currently misses,
so we can improve the existing keys empirically.
"""

import sys
import re
import time
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from rapidfuzz import fuzz

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

OUTPUT_PAIRS = OUTPUT_DIR / "combo2_missed_pairs.tsv"
OUTPUT_SUMMARY = OUTPUT_DIR / "combo2_missed_summary.tsv"

S1_SAMPLE_SIZE = 2000
RANDOM_STATE = 42
CHUNK_SIZE = 100_000


# ============================================================
# BASIC KEY FUNCTIONS
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
    if not text:
        return ""

    match = re.search(r"\d+", str(text))

    if match:
        return match.group(0)

    return ""


def make_combo2_keys(name, address, country):
    """
    EXACTLY reproduce the current COMBO_2 keys.
    """

    name_norm = normalize_business_name(name)
    address_norm = normalize_address(address)

    country_norm = str(country).strip().casefold()

    return {
        "name_first": (
            country_norm,
            first_token(name_norm),
        ),

        "name_prefix2": (
            country_norm,
            first_two_chars(name_norm),
        ),

        "address_number": (
            country_norm,
            first_address_number(address_norm),
        ),

        "address_first": (
            country_norm,
            first_token(address_norm),
        ),
    }


def parse_ground_truth(value):
    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


def script_type(text):
    """
    Lightweight script classification for diagnostics.
    """

    if not text:
        return "empty"

    has_latin = False
    has_devanagari = False
    has_other = False

    for ch in str(text):

        code = ord(ch)

        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            has_latin = True

        elif 0x0900 <= code <= 0x097F:
            has_devanagari = True

        elif ch.isalpha():
            has_other = True

    if has_latin and has_devanagari:
        return "mixed_latin_devanagari"

    if has_latin:
        return "latin"

    if has_devanagari:
        return "devanagari"

    if has_other:
        return "other"

    return "non_alpha"


def get_key_status(s1_keys, target_keys):
    """
    Determine which COMBO_2 key(s) recover a pair.
    """

    status = {}

    for key_name in (
        "name_first",
        "name_prefix2",
        "address_number",
        "address_first",
    ):
        s1_value = s1_keys[key_name]
        target_value = target_keys[key_name]

        # Empty keys should NOT count as matching blocks.
        status[key_name] = (
            bool(s1_value[1])
            and bool(target_value[1])
            and s1_value == target_value
        )

    return status


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)
    print("COMBO_2 MISSED-POSITIVE DIAGNOSTIC")
    print("=" * 70)

    print(
        "\nCOMBO_2 = "
        "name_first OR name_prefix2 OR "
        "address_number OR address_first"
    )

    # ========================================================
    # LOAD SOURCE1
    # ========================================================

    print("\nLoading Source1...")

    source1 = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Source1 rows: {len(source1):,}"
    )

    source1_sample = source1.sample(
        n=min(S1_SAMPLE_SIZE, len(source1)),
        random_state=RANDOM_STATE,
    ).copy()

    source1_sample = source1_sample.reset_index(drop=True)

    sampled_s1_ids = set(
        source1_sample["entity_id"].astype(str)
    )

    print(
        f"Sampled Source1 entities: "
        f"{len(source1_sample):,}"
    )

    # ========================================================
    # SOURCE1 LOOKUP
    # ========================================================

    s1_lookup = {}

    for _, row in source1_sample.iterrows():

        s1_id = str(row["entity_id"])

        s1_lookup[s1_id] = {
            "entity_id": s1_id,
            "business_name": row["business_name"],
            "business_address": row["business_address"],
            "country": row["country"],
            "keys": make_combo2_keys(
                row["business_name"],
                row["business_address"],
                row["country"],
            ),
        }

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    print("\nLoading ground truth...")

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    # ========================================================
    # BUILD:
    #
    # target_entity_id -> true Source1 IDs
    #
    # This lets us inspect only TRUE pairs while scanning
    # the target sources.
    # ========================================================

    print("\nBuilding true target-pair lookup...")

    target_to_s1 = defaultdict(set)

    total_true_pairs = 0

    for _, row in ground_truth.iterrows():

        s1_id = str(row["source1_entity_id"])

        if s1_id not in sampled_s1_ids:
            continue

        true_target_ids = parse_ground_truth(
            row["matched_entity_ids"]
        )

        for target_id in true_target_ids:

            target_to_s1[target_id].add(
                s1_id
            )

            total_true_pairs += 1

    print(
        f"Total sampled true pairs: "
        f"{total_true_pairs:,}"
    )

    print(
        f"Unique target IDs involved: "
        f"{len(target_to_s1):,}"
    )

    # ========================================================
    # DIAGNOSTIC STORAGE
    # ========================================================

    missed_rows = []

    recovered_pairs = 0
    missed_pairs = 0

    # ========================================================
    # SCAN SOURCE2 + SOURCE3
    # ========================================================

    target_files = [
        ("source2", SOURCE2_PATH),
        ("source3", SOURCE3_PATH),
    ]

    for source_name, source_path in target_files:

        print("\n" + "-" * 70)
        print(
            f"Scanning {source_name}: "
            f"{source_path.name}"
        )
        print("-" * 70)

        processed = 0
        true_rows_seen = 0

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ):

            processed += len(chunk)

            for _, row in chunk.iterrows():

                target_id = str(row["entity_id"])

                # Most rows are not part of the sampled
                # ground-truth positives. Skip immediately.
                true_s1_ids = target_to_s1.get(
                    target_id
                )

                if not true_s1_ids:
                    continue

                true_rows_seen += 1

                target_keys = make_combo2_keys(
                    row["business_name"],
                    row["business_address"],
                    row["country"],
                )

                for s1_id in true_s1_ids:

                    s1 = s1_lookup[s1_id]

                    key_status = get_key_status(
                        s1["keys"],
                        target_keys,
                    )

                    recovered = any(
                        key_status.values()
                    )

                    if recovered:

                        recovered_pairs += 1

                        continue

                    # ====================================================
                    # MISSED PAIR
                    # ====================================================

                    missed_pairs += 1

                    s1_name = s1["business_name"]
                    s1_address = s1["business_address"]

                    target_name = row["business_name"]
                    target_address = row["business_address"]

                    name_norm_s1 = normalize_business_name(
                        s1_name
                    )

                    name_norm_target = normalize_business_name(
                        target_name
                    )

                    address_norm_s1 = normalize_address(
                        s1_address
                    )

                    address_norm_target = normalize_address(
                        target_address
                    )

                    name_ratio = fuzz.ratio(
                        name_norm_s1,
                        name_norm_target,
                    )

                    name_token_ratio = fuzz.token_set_ratio(
                        name_norm_s1,
                        name_norm_target,
                    )

                    address_ratio = fuzz.ratio(
                        address_norm_s1,
                        address_norm_target,
                    )

                    address_token_ratio = fuzz.token_set_ratio(
                        address_norm_s1,
                        address_norm_target,
                    )

                    missed_rows.append({
                        "source": source_name,
                        "source1_entity_id": s1_id,
                        "target_entity_id": target_id,

                        "country": s1["country"],

                        "source1_name": s1_name,
                        "target_name": target_name,

                        "source1_address": s1_address,
                        "target_address": target_address,

                        "source1_name_script": script_type(
                            s1_name
                        ),
                        "target_name_script": script_type(
                            target_name
                        ),

                        "source1_address_script": script_type(
                            s1_address
                        ),
                        "target_address_script": script_type(
                            target_address
                        ),

                        "name_ratio": round(
                            name_ratio,
                            2,
                        ),

                        "name_token_ratio": round(
                            name_token_ratio,
                            2,
                        ),

                        "address_ratio": round(
                            address_ratio,
                            2,
                        ),

                        "address_token_ratio": round(
                            address_token_ratio,
                            2,
                        ),

                        "s1_name_first": s1["keys"][
                            "name_first"
                        ][1],

                        "target_name_first": target_keys[
                            "name_first"
                        ][1],

                        "s1_name_prefix2": s1["keys"][
                            "name_prefix2"
                        ][1],

                        "target_name_prefix2": target_keys[
                            "name_prefix2"
                        ][1],

                        "s1_address_number": s1["keys"][
                            "address_number"
                        ][1],

                        "target_address_number": target_keys[
                            "address_number"
                        ][1],

                        "s1_address_first": s1["keys"][
                            "address_first"
                        ][1],

                        "target_address_first": target_keys[
                            "address_first"
                        ][1],

                        "name_first_match": key_status[
                            "name_first"
                        ],

                        "name_prefix2_match": key_status[
                            "name_prefix2"
                        ],

                        "address_number_match": key_status[
                            "address_number"
                        ],

                        "address_first_match": key_status[
                            "address_first"
                        ],
                    })

            print(
                f"\rProcessed {processed:,} rows",
                end="",
                flush=True,
            )

        print(
            f"\nTrue target rows encountered: "
            f"{true_rows_seen:,}"
        )

    # ========================================================
    # SAVE MISSED PAIRS
    # ========================================================

    missed_df = pd.DataFrame(missed_rows)

    missed_df.to_csv(
        OUTPUT_PAIRS,
        sep="\t",
        index=False,
    )

    # ========================================================
    # SUMMARY ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("COMBO_2 DIAGNOSTIC RESULTS")
    print("=" * 70)

    print(
        f"\nTotal true pairs: "
        f"{total_true_pairs:,}"
    )

    print(
        f"Recovered by COMBO_2: "
        f"{recovered_pairs:,}"
    )

    print(
        f"Missed by COMBO_2: "
        f"{missed_pairs:,}"
    )

    recall = (
        recovered_pairs / total_true_pairs * 100
        if total_true_pairs
        else 0
    )

    print(
        f"COMBO_2 recall: "
        f"{recall:.2f}%"
    )

    if missed_df.empty:

        print("\nNo missed pairs found.")

        return

    # ========================================================
    # MISS PATTERN COUNTS
    # ========================================================

    summary_rows = []

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    for source, count in (
        missed_df["source"]
        .value_counts()
        .items()
    ):

        summary_rows.append({
            "category": "source",
            "value": source,
            "count": int(count),
            "pct_of_misses": (
                count / missed_pairs * 100
            ),
        })

    # --------------------------------------------------------
    # Name script
    # --------------------------------------------------------

    for script, count in (
        missed_df[
            "target_name_script"
        ].value_counts()
        .items()
    ):

        summary_rows.append({
            "category": "target_name_script",
            "value": script,
            "count": int(count),
            "pct_of_misses": (
                count / missed_pairs * 100
            ),
        })

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    for country, count in (
        missed_df["country"]
        .value_counts()
        .items()
    ):

        summary_rows.append({
            "category": "country",
            "value": country,
            "count": int(count),
            "pct_of_misses": (
                count / missed_pairs * 100
            ),
        })

    # --------------------------------------------------------
    # Name similarity bucket
    # --------------------------------------------------------

    name_bins = [
        (-1, 40, "<40"),
        (40, 60, "40-59"),
        (60, 70, "60-69"),
        (70, 80, "70-79"),
        (80, 90, "80-89"),
        (90, 101, "90-100"),
    ]

    for low, high, label in name_bins:

        mask = (
            (missed_df["name_ratio"] >= low)
            & (missed_df["name_ratio"] < high)
        )

        count = int(mask.sum())

        summary_rows.append({
            "category": "name_ratio_bucket",
            "value": label,
            "count": count,
            "pct_of_misses": (
                count / missed_pairs * 100
            ),
        })

    # --------------------------------------------------------
    # Address similarity bucket
    # --------------------------------------------------------

    address_bins = [
        (-1, 40, "<40"),
        (40, 60, "40-59"),
        (60, 70, "60-69"),
        (70, 80, "70-79"),
        (80, 90, "80-89"),
        (90, 101, "90-100"),
    ]

    for low, high, label in address_bins:

        mask = (
            (missed_df["address_ratio"] >= low)
            & (missed_df["address_ratio"] < high)
        )

        count = int(mask.sum())

        summary_rows.append({
            "category": "address_ratio_bucket",
            "value": label,
            "count": count,
            "pct_of_misses": (
                count / missed_pairs * 100
            ),
        })

    # --------------------------------------------------------
    # Missing address
    # --------------------------------------------------------

    s1_missing = (
        missed_df["source1_address"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    target_missing = (
        missed_df["target_address"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    both_missing = (
        s1_missing
        & target_missing
    )

    either_missing = (
        s1_missing
        | target_missing
    )

    for label, mask in [
        ("source1_address_missing", s1_missing),
        ("target_address_missing", target_missing),
        ("either_address_missing", either_missing),
        ("both_address_missing", both_missing),
    ]:

        count = int(mask.sum())

        summary_rows.append({
            "category": "address_missingness",
            "value": label,
            "count": count,
            "pct_of_misses": (
                count / missed_pairs * 100
            ),
        })

    # --------------------------------------------------------
    # Which individual keys differ
    # --------------------------------------------------------

    key_columns = [
        "name_first_match",
        "name_prefix2_match",
        "address_number_match",
        "address_first_match",
    ]

    for column in key_columns:

        count = int(
            (~missed_df[column]).sum()
        )

        summary_rows.append({
            "category": "key_failure",
            "value": column,
            "count": count,
            "pct_of_misses": (
                count / missed_pairs * 100
            ),
        })

    # --------------------------------------------------------
    # Cross-script
    # --------------------------------------------------------

    cross_script = (
        missed_df["source1_name_script"]
        != missed_df["target_name_script"]
    )

    cross_script_count = int(
        cross_script.sum()
    )

    summary_rows.append({
        "category": "name_script_change",
        "value": "source1_vs_target_different",
        "count": cross_script_count,
        "pct_of_misses": (
            cross_script_count / missed_pairs * 100
        ),
    })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        sep="\t",
        index=False,
    )

    # ========================================================
    # PRINT USEFUL SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print("MISSED PAIR SUMMARY")
    print("-" * 70)

    print("\nBy source:")

    print(
        missed_df["source"]
        .value_counts()
        .to_string()
    )

    print("\nBy country:")

    print(
        missed_df["country"]
        .value_counts()
        .to_string()
    )

    print("\nBy target name script:")

    print(
        missed_df[
            "target_name_script"
        ]
        .value_counts()
        .to_string()
    )

    print("\nMean similarities among missed pairs:")

    print(
        f"  name ratio: "
        f"{missed_df['name_ratio'].mean():.2f}"
    )

    print(
        f"  name token ratio: "
        f"{missed_df['name_token_ratio'].mean():.2f}"
    )

    print(
        f"  address ratio: "
        f"{missed_df['address_ratio'].mean():.2f}"
    )

    print(
        f"  address token ratio: "
        f"{missed_df['address_token_ratio'].mean():.2f}"
    )

    print("\nKey failure counts:")

    failure_rows = []

    for column in key_columns:

        failure_count = int(
            (~missed_df[column]).sum()
        )

        failure_rows.append({
            "key": column,
            "failures": failure_count,
            "pct": failure_count / missed_pairs * 100,
        })

    failure_df = pd.DataFrame(
        failure_rows
    ).sort_values(
        "failures",
        ascending=False,
    )

    print(
        failure_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\nTop missed examples:")

    display_columns = [
        "source",
        "country",
        "name_ratio",
        "address_ratio",
        "source1_name",
        "target_name",
        "source1_address",
        "target_address",
    ]

    print(
        missed_df[
            display_columns
        ]
        .sort_values(
            [
                "name_ratio",
                "address_ratio",
            ],
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"\nMissed-pair details saved to:\n"
        f"{OUTPUT_PAIRS}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{OUTPUT_SUMMARY}"
    )

    print(
        f"\nRuntime: "
        f"{time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()