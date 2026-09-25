"""
Diagnose remaining missed true matches from ALL_V3.

ALL_V3
------
country + name_first_v2
OR country + name_prefix2_v2
OR country + address_number_v2
OR country + address_first_v3

Purpose
-------
Find exactly why the remaining true pairs are still missed.

This script does NOT add a new blocking strategy.

It reports:
    - total true pairs
    - ALL_V3 recovered / missed
    - source / country distribution
    - name/address similarity
    - script patterns
    - exact failure of each blocking key
    - missingness
    - structural patterns
    - examples of difficult misses
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

TRAIN_DIR = (
    ROOT
    / "data"
    / "raw"
    / "dataset"
    / "train"
)

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
    / "all_v3_missed_pairs.tsv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "all_v3_missed_summary.tsv"
)

OUTPUT_EXAMPLES = (
    OUTPUT_DIR
    / "all_v3_missed_high_confidence.tsv"
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
    name_norm = normalize_business_name(name)

    if not name_norm:
        return ""

    tokens = name_norm.split()

    while tokens:

        first = tokens[0]

        if first in NAME_PREFIXES_TO_REMOVE:
            tokens.pop(0)
            continue

        # M/s -> m s after normalization
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
# ADDRESS-FIRST V3
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


def is_numeric_like_token(token):

    if not token:
        return False

    compact = re.sub(
        r"[^0-9A-Za-z]",
        "",
        token,
    )

    if not compact:
        return False

    return bool(
        re.fullmatch(
            r"\d+[A-Za-z]?",
            compact,
        )
    )


def is_meaningful_alpha_token(token):

    if not token:
        return False

    if is_numeric_like_token(token):
        return False

    return any(
        ch.isalpha()
        for ch in token
    )


def address_first_v3(address):

    address_norm = normalize_address(
        address
    )

    if not address_norm:
        return ""

    tokens = address_norm.split()

    i = 0

    while i < len(tokens):

        token = tokens[i]

        # ----------------------------------------------------
        # Structural prefixes
        # ----------------------------------------------------

        if token in ADDRESS_PREFIX_WORDS:

            i += 1

            if (
                i < len(tokens)
                and tokens[i] == "no"
            ):
                i += 1

            if (
                i < len(tokens)
                and is_numeric_like_token(
                    tokens[i]
                )
            ):
                i += 1

            continue

        # ----------------------------------------------------
        # h no 123
        # ----------------------------------------------------

        if token == "h":

            i += 1

            if (
                i < len(tokens)
                and tokens[i] == "no"
            ):
                i += 1

            if (
                i < len(tokens)
                and is_numeric_like_token(
                    tokens[i]
                )
            ):
                i += 1

            continue

        # ----------------------------------------------------
        # no 123
        # ----------------------------------------------------

        if token == "no":

            i += 1

            if (
                i < len(tokens)
                and is_numeric_like_token(
                    tokens[i]
                )
            ):
                i += 1

            continue

        # ----------------------------------------------------
        # Floor descriptors
        # ----------------------------------------------------

        if (
            token in FLOOR_WORDS
            and i + 1 < len(tokens)
            and tokens[i + 1] == "floor"
        ):

            i += 2
            continue

        if (
            looks_like_floor_number(token)
            and i + 1 < len(tokens)
            and tokens[i + 1] == "floor"
        ):

            i += 2
            continue

        # ----------------------------------------------------
        # Leading number
        # ----------------------------------------------------

        if is_numeric_like_token(token):
            i += 1
            continue

        # ----------------------------------------------------
        # Punctuation artifact
        # ----------------------------------------------------

        if not any(
            ch.isalpha()
            for ch in token
        ):
            i += 1
            continue

        # ----------------------------------------------------
        # First meaningful alphabetic token
        # ----------------------------------------------------

        if is_meaningful_alpha_token(token):
            return token

        i += 1

    return ""


# ============================================================
# ALL_V3 KEY BUILDER
# ============================================================

def build_all_v3_keys(
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
            address_number_v2(address),
        ),

        "address_first": (
            country_norm,
            address_first_v3(address),
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
# SCRIPT DETECTION
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
    print("ALL_V3 MISSED-POSITIVE DIAGNOSTIC")
    print("=" * 70)

    print(
        "\nALL_V3:"
        "\n  name_first_v2"
        "\n  OR name_prefix2_v2"
        "\n  OR address_number_v2"
        "\n  OR address_first_v3"
    )

    print(
        f"\nSample size: "
        f"{S1_SAMPLE_SIZE:,}"
    )

    print(
        f"Chunk size: "
        f"{CHUNK_SIZE:,}"
    )

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
        source1_sample[
            "entity_id"
        ].astype(str)
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

            "country": row[
                "country"
            ],

            "keys": build_all_v3_keys(
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
        f"Total true matches: "
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

    recovered_pairs = set()

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

                target_keys = build_all_v3_keys(
                    row["business_name"],
                    row["business_address"],
                    row["country"],
                )

                for s1_id in true_s1_ids:

                    s1 = s1_lookup[s1_id]

                    s1_keys = s1["keys"]

                    # ------------------------------------------------
                    # Exact ALL_V3 key comparisons
                    # ------------------------------------------------

                    key_matches = {}

                    for key_name in (
                        "name_first",
                        "name_prefix2",
                        "address_number",
                        "address_first",
                    ):

                        s1_key = s1_keys[
                            key_name
                        ]

                        target_key = target_keys[
                            key_name
                        ]

                        key_matches[
                            key_name
                        ] = (
                            bool(s1_key[1])
                            and bool(target_key[1])
                            and s1_key
                            == target_key
                        )

                    recovered = any(
                        key_matches.values()
                    )

                    pair = (
                        s1_id,
                        target_id,
                    )

                    if recovered:

                        recovered_pairs.add(
                            pair
                        )

                        continue

                    # ================================================
                    # MISSED PAIR
                    # ================================================

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

                    # ------------------------------------------------
                    # Similarities
                    # ------------------------------------------------

                    s1_name_norm = (
                        normalize_business_name(
                            s1_name
                        )
                    )

                    target_name_norm = (
                        normalize_business_name(
                            target_name
                        )
                    )

                    s1_address_norm = (
                        normalize_address(
                            s1_address
                        )
                    )

                    target_address_norm = (
                        normalize_address(
                            target_address
                        )
                    )

                    name_ratio = fuzz.ratio(
                        s1_name_norm,
                        target_name_norm,
                    )

                    name_token_ratio = (
                        fuzz.token_set_ratio(
                            s1_name_norm,
                            target_name_norm,
                        )
                    )

                    name_sort_ratio = (
                        fuzz.token_sort_ratio(
                            s1_name_norm,
                            target_name_norm,
                        )
                    )

                    address_ratio = fuzz.ratio(
                        s1_address_norm,
                        target_address_norm,
                    )

                    address_token_ratio = (
                        fuzz.token_set_ratio(
                            s1_address_norm,
                            target_address_norm,
                        )
                    )

                    address_sort_ratio = (
                        fuzz.token_sort_ratio(
                            s1_address_norm,
                            target_address_norm,
                        )
                    )

                    # ------------------------------------------------
                    # Name/address missingness
                    # ------------------------------------------------

                    target_address_missing = (
                        str(
                            target_address
                        ).strip()
                        == ""
                    )

                    s1_address_missing = (
                        str(
                            s1_address
                        ).strip()
                        == ""
                    )

                    # ------------------------------------------------
                    # Token structure
                    # ------------------------------------------------

                    s1_name_tokens = (
                        s1_name_norm.split()
                    )

                    target_name_tokens = (
                        target_name_norm.split()
                    )

                    s1_address_tokens = (
                        s1_address_norm.split()
                    )

                    target_address_tokens = (
                        target_address_norm.split()
                    )

                    # ------------------------------------------------
                    # Diagnostic category
                    # ------------------------------------------------

                    if (
                        name_ratio >= 85
                        and address_token_ratio >= 80
                    ):
                        category = (
                            "high_name_high_address"
                        )

                    elif (
                        name_ratio >= 85
                        and address_token_ratio < 80
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

                    # ------------------------------------------------
                    # Possible character corruption indicator
                    # ------------------------------------------------

                    normalized_name_equal = (
                        s1_name_norm
                        == target_name_norm
                    )

                    # ------------------------------------------------
                    # Row
                    # ------------------------------------------------

                    missed_rows.append({

                        "source":
                            source_name,

                        "source1_entity_id":
                            s1_id,

                        "target_entity_id":
                            target_id,

                        "country":
                            s1["country"],

                        "source1_name":
                            s1_name,

                        "target_name":
                            target_name,

                        "source1_address":
                            s1_address,

                        "target_address":
                            target_address,

                        # ----------------------------
                        # Script
                        # ----------------------------

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

                        # ----------------------------
                        # Similarity
                        # ----------------------------

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

                        "name_sort_ratio":
                            round(
                                name_sort_ratio,
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

                        "address_sort_ratio":
                            round(
                                address_sort_ratio,
                                2,
                            ),

                        # ----------------------------
                        # ALL_V3 keys
                        # ----------------------------

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

                        # ----------------------------
                        # Key matches
                        # ----------------------------

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

                        # ----------------------------
                        # Missingness
                        # ----------------------------

                        "source1_address_missing":
                            s1_address_missing,

                        "target_address_missing":
                            target_address_missing,

                        "either_address_missing":
                            (
                                s1_address_missing
                                or
                                target_address_missing
                            ),

                        # ----------------------------
                        # Structure
                        # ----------------------------

                        "source1_name_token_count":
                            len(
                                s1_name_tokens
                            ),

                        "target_name_token_count":
                            len(
                                target_name_tokens
                            ),

                        "source1_address_token_count":
                            len(
                                s1_address_tokens
                            ),

                        "target_address_token_count":
                            len(
                                target_address_tokens
                            ),

                        "name_token_count_difference":
                            abs(
                                len(
                                    s1_name_tokens
                                )
                                -
                                len(
                                    target_name_tokens
                                )
                            ),

                        "address_token_count_difference":
                            abs(
                                len(
                                    s1_address_tokens
                                )
                                -
                                len(
                                    target_address_tokens
                                )
                            ),

                        # ----------------------------
                        # Exact normalized equality
                        # ----------------------------

                        "normalized_name_equal":
                            normalized_name_equal,

                        "normalized_address_equal":
                            (
                                s1_address_norm
                                ==
                                target_address_norm
                            ),

                        # ----------------------------
                        # Category
                        # ----------------------------

                        "similarity_category":
                            category,
                    })

            print(
                f"\rProcessed "
                f"{processed:,} rows",
                end="",
                flush=True,
            )

        print(
            f"\nTrue target rows encountered: "
            f"{true_rows_seen:,}"
        )

        print(
            f"{source_name} scan time: "
            f"{time.time() - source_start:.2f}s"
        )

    # ========================================================
    # RESULTS
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
    # BASIC METRICS
    # ========================================================

    recovered_count = len(
        recovered_pairs
    )

    missed_count = len(
        missed_df
    )

    recall = (
        recovered_count
        / total_true_matches
        * 100
        if total_true_matches
        else 0
    )

    print("\n" + "=" * 70)
    print("ALL_V3 DIAGNOSTIC RESULTS")
    print("=" * 70)

    print(
        f"\nTotal true pairs: "
        f"{total_true_matches:,}"
    )

    print(
        f"Recovered by ALL_V3: "
        f"{recovered_count:,}"
    )

    print(
        f"Missed by ALL_V3: "
        f"{missed_count:,}"
    )

    print(
        f"ALL_V3 recall: "
        f"{recall:.2f}%"
    )

    if missed_df.empty:

        print(
            "\nALL_V3 recovered every sampled "
            "true pair."
        )

        return

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    def add_summary(
        category,
        value,
        count,
    ):

        summary_rows.append({
            "category":
                category,

            "value":
                value,

            "count":
                int(count),

            "pct_of_misses":
                (
                    count
                    / missed_count
                    * 100
                ),
        })

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    for value, count in (
        missed_df[
            "source"
        ]
        .value_counts()
        .items()
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
        ]
        .value_counts()
        .items()
    ):

        add_summary(
            "country",
            value,
            count,
        )

    # --------------------------------------------------------
    # Similarity category
    # --------------------------------------------------------

    for value, count in (
        missed_df[
            "similarity_category"
        ]
        .value_counts()
        .items()
    ):

        add_summary(
            "similarity_category",
            value,
            count,
        )

    # --------------------------------------------------------
    # Target name script
    # --------------------------------------------------------

    for value, count in (
        missed_df[
            "target_name_script"
        ]
        .value_counts()
        .items()
    ):

        add_summary(
            "target_name_script",
            value,
            count,
        )

    # --------------------------------------------------------
    # KEY FAILURES
    # --------------------------------------------------------

    for column in (
        "name_first_match",
        "name_prefix2_match",
        "address_number_match",
        "address_first_match",
    ):

        count = int(
            (
                ~missed_df[column]
            ).sum()
        )

        add_summary(
            "key_failure",
            column,
            count,
        )

    # --------------------------------------------------------
    # Missing address
    # --------------------------------------------------------

    for column in (
        "source1_address_missing",
        "target_address_missing",
        "either_address_missing",
    ):

        count = int(
            missed_df[column].sum()
        )

        add_summary(
            "address_missingness",
            column,
            count,
        )

    # --------------------------------------------------------
    # Similarity buckets
    # --------------------------------------------------------

    similarity_bins = [
        (-1, 40, "<40"),
        (40, 60, "40-59"),
        (60, 70, "60-69"),
        (70, 80, "70-79"),
        (80, 90, "80-89"),
        (90, 101, "90-100"),
    ]

    for low, high, label in similarity_bins:

        mask = (
            (
                missed_df[
                    "name_ratio"
                ] >= low
            )
            &
            (
                missed_df[
                    "name_ratio"
                ] < high
            )
        )

        add_summary(
            "name_ratio_bucket",
            label,
            mask.sum(),
        )

    for low, high, label in similarity_bins:

        mask = (
            (
                missed_df[
                    "address_token_ratio"
                ] >= low
            )
            &
            (
                missed_df[
                    "address_token_ratio"
                ] < high
            )
        )

        add_summary(
            "address_token_ratio_bucket",
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
    print("SIMILARITY CATEGORIES")
    print("-" * 70)

    print(
        missed_df[
            "similarity_category"
        ]
        .value_counts()
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
            (
                ~missed_df[column]
            ).sum()
        )

        key_rows.append({
            "key":
                column,

            "failures":
                count,

            "pct":
                count
                / missed_count
                * 100,
        })

    key_df = (
        pd.DataFrame(key_rows)
        .sort_values(
            "failures",
            ascending=False,
        )
    )

    print(
        key_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # ========================================================
    # ADDRESS MISSINGNESS
    # ========================================================

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
            f"("
            f"{count / missed_count * 100:.2f}%"
            f")"
        )

    # ========================================================
    # MEAN SIMILARITIES
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
        f"name sort ratio: "
        f"{missed_df['name_sort_ratio'].mean():.2f}"
    )

    print(
        f"address ratio: "
        f"{missed_df['address_ratio'].mean():.2f}"
    )

    print(
        f"address token ratio: "
        f"{missed_df['address_token_ratio'].mean():.2f}"
    )

    print(
        f"address sort ratio: "
        f"{missed_df['address_sort_ratio'].mean():.2f}"
    )

    # ========================================================
    # TOP MISSES
    # ========================================================

    display_columns = [
        "source",
        "country",
        "name_ratio",
        "name_token_ratio",
        "name_sort_ratio",
        "address_ratio",
        "address_token_ratio",
        "source1_name",
        "target_name",
        "source1_address",
        "target_address",
    ]

    print("\n" + "=" * 70)
    print("TOP MISSED TRUE PAIRS")
    print("=" * 70)

    print(
        missed_df[
            display_columns
        ]
        .sort_values(
            [
                "address_token_ratio",
                "name_token_ratio",
            ],
            ascending=False,
        )
        .head(40)
        .to_string(index=False)
    )

    # ========================================================
    # HIGH-CONFIDENCE MISSES
    # ========================================================

    high_conf_mask = (
        (
            missed_df[
                "name_token_ratio"
            ] >= 85
        )
        |
        (
            missed_df[
                "address_token_ratio"
            ] >= 90
        )
    )

    high_conf = missed_df[
        high_conf_mask
    ].copy()

    high_conf.to_csv(
        OUTPUT_EXAMPLES,
        sep="\t",
        index=False,
    )

    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE-LOOKING MISSES")
    print("=" * 70)

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
                    "address_token_ratio",
                    "name_token_ratio",
                ],
                ascending=False,
            )
            .head(40)
            .to_string(index=False)
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"\nMissed pairs saved to:"
        f"\n{OUTPUT_MISSES}"
    )

    print(
        f"\nSummary saved to:"
        f"\n{OUTPUT_SUMMARY}"
    )

    print(
        f"\nHigh-confidence misses saved to:"
        f"\n{OUTPUT_EXAMPLES}"
    )

    print(
        f"\nRuntime: "
        f"{time.time() - start_time:.2f}s"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()