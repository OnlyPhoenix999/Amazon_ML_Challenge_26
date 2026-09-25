"""
Diagnose the remaining missed true matches from ALL_V2.

ALL_V2 architecture
-------------------
country + name_first_v2
OR
country + name_prefix2_v2
OR
country + address_number_v2
OR
country + address_first_v2

This script DOES NOT introduce any new blocking logic.

It answers:
    Why are the remaining ALL_V2 true pairs missed?

For each missed true pair we record:
    - source
    - country
    - source1 / target name
    - source1 / target address
    - name similarity
    - address similarity
    - exact status of each ALL_V2 key
    - script information
    - address missingness
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
from rapidfuzz import fuzz

from src.preprocessing.normalization import (
    normalize_business_name,
    normalize_address,
)

# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = ROOT / "data" / "raw" / "dataset" / "train"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "candidate_generation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_MISSES = (
    OUTPUT_DIR
    / "all_v2_missed_pairs.tsv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "all_v2_missed_summary.tsv"
)

S1_SAMPLE_SIZE = 2000
RANDOM_STATE = 42
CHUNK_SIZE = 100_000


# ============================================================
# NAME V2
# ============================================================

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
    Same NAME_V2 logic used previously.
    """

    name_norm = normalize_business_name(name)

    if not name_norm:
        return ""

    tokens = name_norm.split()

    while tokens:

        first = tokens[0]

        if first in NAME_PREFIXES_TO_REMOVE:
            tokens.pop(0)
            continue

        # Handles M/s -> m s after normalization.
        if (
            len(tokens) >= 2
            and tokens[0] == "m"
            and tokens[1] == "s"
        ):
            tokens = tokens[2:]
            continue

        break

    return " ".join(tokens)


def name_first_v2(name):
    cleaned = clean_name_leading_prefixes(name)

    if not cleaned:
        return ""

    return cleaned.split()[0]


def name_prefix2_v2(name):
    cleaned = clean_name_leading_prefixes(name)

    if not cleaned:
        return ""

    compact = cleaned.replace(" ", "")

    return compact[:2]


# ============================================================
# ADDRESS NUMBER V2
# ============================================================

ADDRESS_NUMBER_PATTERN = re.compile(
    r"\d+(?:\s*/\s*\d+)?[A-Za-z]?"
)


def canonicalize_address_number(number_text):
    if not number_text:
        return ""

    value = str(
        number_text
    ).strip().lower()

    value = re.sub(
        r"\s*/\s*",
        "/",
        value,
    )

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

            digits = (
                digits.lstrip("0")
                or "0"
            )

            normalized.append(
                digits + suffix
            )

        return "/".join(normalized)

    match = re.match(
        r"(\d+)([a-z]?)$",
        value,
    )

    if match:

        digits = match.group(1)
        suffix = match.group(2)

        digits = (
            digits.lstrip("0")
            or "0"
        )

        return digits + suffix

    return value


def address_number_v2(address):
    address_norm = normalize_address(
        address
    )

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
# ADDRESS FIRST V2
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
            r"\d+(st|nd|rd|th)",
            token,
        )
    )


def address_first_v2(address):
    address_norm = normalize_address(
        address
    )

    if not address_norm:
        return ""

    tokens = address_norm.split()

    changed = True

    while changed and tokens:

        changed = False

        # door no 123
        # flat no 123
        # plot no 123
        # etc.
        if tokens[0] in ADDRESS_PREFIX_WORDS:

            tokens.pop(0)
            changed = True

            if tokens and tokens[0] == "no":
                tokens.pop(0)

                if tokens:
                    tokens.pop(0)

            continue

        # h no 123
        if tokens[0] == "h":

            tokens.pop(0)
            changed = True

            if tokens and tokens[0] == "no":
                tokens.pop(0)

            if tokens:
                tokens.pop(0)

            continue

        # no 123
        if tokens[0] == "no":

            tokens.pop(0)
            changed = True

            if tokens:
                tokens.pop(0)

            continue

        # first floor / ground floor / etc.
        if (
            len(tokens) >= 2
            and tokens[0] in FLOOR_WORDS
            and tokens[1] == "floor"
        ):

            tokens = tokens[2:]
            changed = True
            continue

        # 1st floor / 2nd floor / etc.
        if (
            len(tokens) >= 2
            and looks_like_floor_number(
                tokens[0]
            )
            and tokens[1] == "floor"
        ):

            tokens = tokens[2:]
            changed = True
            continue

    return tokens[0] if tokens else ""


# ============================================================
# ALL V2 KEYS
# ============================================================

def build_all_v2_keys(
    name,
    address,
    country,
):
    country_norm = (
        str(country)
        .strip()
        .casefold()
    )

    return {
        "name_first": (
            country_norm,
            name_first_v2(name),
        ),

        "name_prefix2": (
            country_norm,
            name_prefix2_v2(name),
        ),

        "address_number": (
            country_norm,
            address_number_v2(
                address
            ),
        ),

        "address_first": (
            country_norm,
            address_first_v2(
                address
            ),
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
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


# ============================================================
# SCRIPT
# ============================================================

def detect_script(text):

    if not text:
        return "empty"

    latin = False
    devanagari = False
    other_alpha = False

    for ch in str(text):

        code = ord(ch)

        if (
            ("A" <= ch <= "Z")
            or ("a" <= ch <= "z")
        ):
            latin = True

        elif 0x0900 <= code <= 0x097F:
            devanagari = True

        elif ch.isalpha():
            other_alpha = True

    if latin and devanagari:
        return "mixed_latin_devanagari"

    if latin:
        return "latin"

    if devanagari:
        return "devanagari"

    if other_alpha:
        return "other"

    return "non_alpha"


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)
    print("ALL_V2 MISSED-POSITIVE DIAGNOSTIC")
    print("=" * 70)

    print(
        "\nALL_V2:"
        "\n  name_first_v2"
        "\n  OR name_prefix2_v2"
        "\n  OR address_number_v2"
        "\n  OR address_first_v2"
    )

    # ========================================================
    # CHECK FILES
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
    # SOURCE1 LOOKUP
    # ========================================================

    s1_lookup = {}

    for _, row in source1_sample.iterrows():

        s1_id = str(
            row["entity_id"]
        )

        s1_lookup[s1_id] = {
            "entity_id": s1_id,
            "business_name": row[
                "business_name"
            ],
            "business_address": row[
                "business_address"
            ],
            "country": row["country"],
            "keys": build_all_v2_keys(
                row["business_name"],
                row["business_address"],
                row["country"],
            ),
        }

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

        target_ids = parse_ground_truth(
            row["matched_entity_ids"]
        )

        for target_id in target_ids:

            target_to_s1[
                target_id
            ].add(
                s1_id
            )

            total_true_matches += 1

    print(
        f"Total sampled true matches: "
        f"{total_true_matches:,}"
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
    # TARGET SCAN
    # ========================================================

    for source_name, source_path in [
        ("source2", SOURCE2_PATH),
        ("source3", SOURCE3_PATH),
    ]:

        print("\n" + "-" * 70)
        print(
            f"Scanning {source_name}: "
            f"{source_path.name}"
        )
        print("-" * 70)

        source_start = time.time()
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

                target_id = str(
                    row["entity_id"]
                )

                true_s1_ids = (
                    target_to_s1.get(
                        target_id
                    )
                )

                if not true_s1_ids:
                    continue

                true_rows_seen += 1

                target_keys = build_all_v2_keys(
                    row["business_name"],
                    row["business_address"],
                    row["country"],
                )

                for s1_id in true_s1_ids:

                    s1 = s1_lookup[s1_id]

                    s1_keys = s1["keys"]

                    # ----------------------------------------
                    # Exact ALL_V2 block matches
                    # ----------------------------------------

                    key_matches = {}

                    for key_name in (
                        "name_first",
                        "name_prefix2",
                        "address_number",
                        "address_first",
                    ):

                        s1_value = s1_keys[
                            key_name
                        ]

                        target_value = target_keys[
                            key_name
                        ]

                        key_matches[
                            key_name
                        ] = (
                            bool(s1_value[1])
                            and bool(target_value[1])
                            and s1_value
                            == target_value
                        )

                    recovered = any(
                        key_matches.values()
                    )

                    if recovered:

                        recovered_pairs += 1
                        continue

                    # ========================================
                    # MISSED ALL_V2 PAIR
                    # ========================================

                    missed_pairs += 1

                    s1_name = s1[
                        "business_name"
                    ]

                    target_name = row[
                        "business_name"
                    ]

                    s1_address = s1[
                        "business_address"
                    ]

                    target_address = row[
                        "business_address"
                    ]

                    name_norm_s1 = (
                        normalize_business_name(
                            s1_name
                        )
                    )

                    name_norm_target = (
                        normalize_business_name(
                            target_name
                        )
                    )

                    address_norm_s1 = (
                        normalize_address(
                            s1_address
                        )
                    )

                    address_norm_target = (
                        normalize_address(
                            target_address
                        )
                    )

                    name_ratio = fuzz.ratio(
                        name_norm_s1,
                        name_norm_target,
                    )

                    name_token_ratio = (
                        fuzz.token_set_ratio(
                            name_norm_s1,
                            name_norm_target,
                        )
                    )

                    address_ratio = fuzz.ratio(
                        address_norm_s1,
                        address_norm_target,
                    )

                    address_token_ratio = (
                        fuzz.token_set_ratio(
                            address_norm_s1,
                            address_norm_target,
                        )
                    )

                    row_out = {
                        "source": source_name,

                        "source1_entity_id": s1_id,
                        "target_entity_id": target_id,

                        "country": s1[
                            "country"
                        ],

                        "source1_name": s1_name,
                        "target_name": target_name,

                        "source1_address": s1_address,
                        "target_address": target_address,

                        "source1_name_script":
                            detect_script(
                                s1_name
                            ),

                        "target_name_script":
                            detect_script(
                                target_name
                            ),

                        "source1_address_script":
                            detect_script(
                                s1_address
                            ),

                        "target_address_script":
                            detect_script(
                                target_address
                            ),

                        "name_ratio":
                            round(
                                name_ratio,
                                2,
                            ),

                        "name_token_ratio":
                            round(
                                name_token_ratio,
                                2,
                            ),

                        "address_ratio":
                            round(
                                address_ratio,
                                2,
                            ),

                        "address_token_ratio":
                            round(
                                address_token_ratio,
                                2,
                            ),

                        # ------------------------------------
                        # Source1 V2 keys
                        # ------------------------------------

                        "s1_name_first":
                            s1_keys[
                                "name_first"
                            ][1],

                        "target_name_first":
                            target_keys[
                                "name_first"
                            ][1],

                        "s1_name_prefix2":
                            s1_keys[
                                "name_prefix2"
                            ][1],

                        "target_name_prefix2":
                            target_keys[
                                "name_prefix2"
                            ][1],

                        "s1_address_number":
                            s1_keys[
                                "address_number"
                            ][1],

                        "target_address_number":
                            target_keys[
                                "address_number"
                            ][1],

                        "s1_address_first":
                            s1_keys[
                                "address_first"
                            ][1],

                        "target_address_first":
                            target_keys[
                                "address_first"
                            ][1],

                        # ------------------------------------
                        # Exact key failures
                        # ------------------------------------

                        "name_first_match":
                            key_matches[
                                "name_first"
                            ],

                        "name_prefix2_match":
                            key_matches[
                                "name_prefix2"
                            ],

                        "address_number_match":
                            key_matches[
                                "address_number"
                            ],

                        "address_first_match":
                            key_matches[
                                "address_first"
                            ],
                    }

                    # ----------------------------------------
                    # Missingness
                    # ----------------------------------------

                    s1_address_missing = (
                        str(
                            s1_address
                        ).strip()
                        == ""
                    )

                    target_address_missing = (
                        str(
                            target_address
                        ).strip()
                        == ""
                    )

                    row_out[
                        "source1_address_missing"
                    ] = s1_address_missing

                    row_out[
                        "target_address_missing"
                    ] = target_address_missing

                    row_out[
                        "either_address_missing"
                    ] = (
                        s1_address_missing
                        or target_address_missing
                    )

                    # ----------------------------------------
                    # Character/name-token diagnostics
                    # ----------------------------------------

                    s1_name_tokens = (
                        name_norm_s1.split()
                    )

                    target_name_tokens = (
                        name_norm_target.split()
                    )

                    row_out[
                        "source1_name_token_count"
                    ] = len(
                        s1_name_tokens
                    )

                    row_out[
                        "target_name_token_count"
                    ] = len(
                        target_name_tokens
                    )

                    row_out[
                        "name_first_token_different"
                    ] = (
                        name_first_v2(
                            s1_name
                        )
                        != name_first_v2(
                            target_name
                        )
                    )

                    # ----------------------------------------
                    # Useful structural comparisons
                    # ----------------------------------------

                    row_out[
                        "normalized_name_equal"
                    ] = (
                        name_norm_s1
                        == name_norm_target
                    )

                    row_out[
                        "normalized_address_equal"
                    ] = (
                        address_norm_s1
                        == address_norm_target
                    )

                    # ----------------------------------------
                    # Simple diagnostic categories
                    # ----------------------------------------

                    if (
                        name_ratio >= 85
                        and address_token_ratio >= 80
                    ):
                        category = (
                            "high_name_high_address"
                        )

                    elif (
                        name_ratio >= 85
                        and address_token_ratio < 60
                    ):
                        category = (
                            "high_name_low_address"
                        )

                    elif (
                        name_ratio < 60
                        and address_token_ratio >= 80
                    ):
                        category = (
                            "low_name_high_address"
                        )

                    elif (
                        name_ratio < 60
                        and address_token_ratio < 60
                    ):
                        category = (
                            "low_name_low_address"
                        )

                    else:
                        category = (
                            "mixed_similarity"
                        )

                    row_out[
                        "similarity_category"
                    ] = category

                    missed_rows.append(
                        row_out
                    )

            print(
                f"\rProcessed {processed:,} rows",
                end="",
                flush=True,
            )

        print(
            f"\nTrue target rows encountered: "
            f"{true_rows_seen:,}"
        )

        print(
            f"{source_name} time: "
            f"{time.time() - source_start:.2f}s"
        )

    # ========================================================
    # SAVE MISSES
    # ========================================================

    missed_df = pd.DataFrame(
        missed_rows
    )

    missed_df.to_csv(
        OUTPUT_MISSES,
        sep="\t",
        index=False,
    )

    # ========================================================
    # BASIC RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("ALL_V2 DIAGNOSTIC RESULTS")
    print("=" * 70)

    print(
        f"\nTotal true pairs: "
        f"{total_true_matches:,}"
    )

    print(
        f"Recovered by ALL_V2: "
        f"{recovered_pairs:,}"
    )

    print(
        f"Missed by ALL_V2: "
        f"{missed_pairs:,}"
    )

    recall = (
        recovered_pairs
        / total_true_matches
        * 100
        if total_true_matches
        else 0
    )

    print(
        f"ALL_V2 recall: "
        f"{recall:.2f}%"
    )

    if missed_df.empty:

        print(
            "\nALL_V2 recovered every sampled "
            "true pair."
        )

        return

    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary_rows = []

    def add_summary(
        category,
        value,
        count,
    ):
        summary_rows.append({
            "category": category,
            "value": value,
            "count": int(count),
            "pct_of_misses": (
                count
                / missed_pairs
                * 100
            ),
        })

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    for value, count in (
        missed_df[
            "source"
        ].value_counts().items()
    ):
        add_summary(
            "source",
            value,
            count,
        )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    for value, count in (
        missed_df[
            "country"
        ].value_counts().items()
    ):
        add_summary(
            "country",
            value,
            count,
        )

    # --------------------------------------------------------
    # Target name script
    # --------------------------------------------------------

    for value, count in (
        missed_df[
            "target_name_script"
        ].value_counts().items()
    ):
        add_summary(
            "target_name_script",
            value,
            count,
        )

    # --------------------------------------------------------
    # Similarity category
    # --------------------------------------------------------

    for value, count in (
        missed_df[
            "similarity_category"
        ].value_counts().items()
    ):
        add_summary(
            "similarity_category",
            value,
            count,
        )

    # --------------------------------------------------------
    # Exact key failure
    # --------------------------------------------------------

    for column in (
        "name_first_match",
        "name_prefix2_match",
        "address_number_match",
        "address_first_match",
    ):

        failures = (
            ~missed_df[column]
        ).sum()

        add_summary(
            "key_failure",
            column,
            failures,
        )

    # --------------------------------------------------------
    # Address missingness
    # --------------------------------------------------------

    for column in (
        "source1_address_missing",
        "target_address_missing",
        "either_address_missing",
    ):

        count = (
            missed_df[column]
        ).sum()

        add_summary(
            "address_missingness",
            column,
            count,
        )

    # --------------------------------------------------------
    # Name first mismatch
    # --------------------------------------------------------

    count = (
        missed_df[
            "name_first_token_different"
        ].sum()
    )

    add_summary(
        "name_structure",
        "name_first_token_different",
        count,
    )

    # --------------------------------------------------------
    # Similarity buckets
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

        add_summary(
            "name_ratio_bucket",
            label,
            mask.sum(),
        )

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

        add_summary(
            "address_ratio_bucket",
            label,
            mask.sum(),
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        sep="\t",
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print("BY SOURCE")
    print("-" * 70)

    print(
        missed_df[
            "source"
        ].value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("BY COUNTRY")
    print("-" * 70)

    print(
        missed_df[
            "country"
        ].value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("SIMILARITY CATEGORIES")
    print("-" * 70)

    print(
        missed_df[
            "similarity_category"
        ].value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("TARGET NAME SCRIPT")
    print("-" * 70)

    print(
        missed_df[
            "target_name_script"
        ].value_counts()
        .to_string()
    )

    print("\n" + "-" * 70)
    print("KEY FAILURE COUNTS")
    print("-" * 70)

    key_rows = []

    for column in (
        "name_first_match",
        "name_prefix2_match",
        "address_number_match",
        "address_first_match",
    ):

        count = int(
            (~missed_df[column]).sum()
        )

        key_rows.append({
            "key": column,
            "failures": count,
            "pct": (
                count
                / missed_pairs
                * 100
            ),
        })

    key_df = pd.DataFrame(
        key_rows
    ).sort_values(
        "failures",
        ascending=False,
    )

    print(
        key_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "-" * 70)
    print("ADDRESS MISSINGNESS")
    print("-" * 70)

    for column in (
        "source1_address_missing",
        "target_address_missing",
        "either_address_missing",
    ):

        count = int(
            missed_df[column].sum()
        )

        print(
            f"{column}: "
            f"{count:,} "
            f"({count / missed_pairs * 100:.2f}%)"
        )

    # ========================================================
    # SIMILARITY MEANS
    # ========================================================

    print("\n" + "-" * 70)
    print("MEAN SIMILARITY OF MISSED PAIRS")
    print("-" * 70)

    print(
        f"name ratio: "
        f"{missed_df['name_ratio'].mean():.2f}"
    )

    print(
        f"name token ratio: "
        f"{missed_df['name_token_ratio'].mean():.2f}"
    )

    print(
        f"address ratio: "
        f"{missed_df['address_ratio'].mean():.2f}"
    )

    print(
        f"address token ratio: "
        f"{missed_df['address_token_ratio'].mean():.2f}"
    )

    # ========================================================
    # TOP MISSES
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP MISSED TRUE PAIRS")
    print("=" * 70)

    display_columns = [
        "source",
        "country",
        "name_ratio",
        "name_token_ratio",
        "address_ratio",
        "address_token_ratio",
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
                "address_token_ratio",
            ],
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )

    # ========================================================
    # HIGH-VALUE MISSES
    # ========================================================

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE-LOOKING MISSES")
    print("=" * 70)

    high_conf = missed_df[
        (
            missed_df["name_ratio"] >= 80
        )
        |
        (
            missed_df[
                "address_token_ratio"
            ] >= 90
        )
    ].copy()

    print(
        f"Count: "
        f"{len(high_conf):,}"
    )

    if not high_conf.empty:

        print(
            high_conf[
                display_columns
            ]
            .sort_values(
                [
                    "name_ratio",
                    "address_token_ratio",
                ],
                ascending=False,
            )
            .head(30)
            .to_string(index=False)
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"\nMissed-pair details saved to:"
        f"\n{OUTPUT_MISSES}"
    )

    print(
        f"\nSummary saved to:"
        f"\n{OUTPUT_SUMMARY}"
    )

    print(
        f"\nRuntime: "
        f"{time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()