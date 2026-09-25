"""
COMBO_2 Address-First V4 Benchmark

Architecture stays exactly the same:

    name_first_v2
    OR name_prefix2_v2
    OR address_number_v2
    OR address_first_v4

Only address_first is changed from V3 -> V4.

V4 idea
-------
Addresses are often reordered across sources.

Examples:

    "Accokeek, MD, 16309 Tortola Drive"
    "TORTOLA DRIVE, ACCOKEEK, MD"

    "RS No 226, Flat No C-808 Anantpuram ..."
    "FLAT NO C-808 ANANTPURAM ..., RS No D/226"

V3 used the first meaningful alphabetic token.

V4 instead selects one stable informative alphabetic
token from the address after removing common structural
and generic address words.

The experiment measures:
    - candidate recall
    - candidate volume
    - recall gain vs ALL_V3
    - candidate volume change
    - recovery of previous ALL_V3 missed pairs
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

PREVIOUS_RESULTS_PATH = (
    ROOT
    / "experiments"
    / "candidate_generation"
    / "combo2_address_first_v3_benchmark.tsv"
)

PREVIOUS_MISSES_PATH = (
    ROOT
    / "experiments"
    / "candidate_generation"
    / "all_v3_missed_pairs.tsv"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "candidate_generation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_RESULTS = (
    OUTPUT_DIR
    / "combo2_address_first_v4_benchmark.tsv"
)

OUTPUT_RECOVERED = (
    OUTPUT_DIR
    / "combo2_address_first_v4_recovered.tsv"
)


# ============================================================
# CONFIG
# ============================================================

S1_SAMPLE_SIZE = 2000
RANDOM_STATE = 42
CHUNK_SIZE = 100_000

# Minimum token length for V4.
MIN_INFORMATIVE_TOKEN_LENGTH = 4


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

    name_norm = normalize_business_name(
        name
    )

    if not name_norm:
        return ""

    tokens = name_norm.split()

    while tokens:

        first = tokens[0]

        if first in NAME_PREFIXES_TO_REMOVE:
            tokens.pop(0)
            continue

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

    cleaned = clean_name_leading_prefixes(
        name
    )

    if not cleaned:
        return ""

    return cleaned.split()[0]


def name_prefix2_v2(name):

    cleaned = clean_name_leading_prefixes(
        name
    )

    if not cleaned:
        return ""

    compact = cleaned.replace(
        " ",
        "",
    )

    return compact[:2]


# ============================================================
# ADDRESS NUMBER V2
# ============================================================

ADDRESS_NUMBER_PATTERN = re.compile(
    r"\d+(?:\s*/\s*\d+)?[A-Za-z]?"
)


def canonicalize_address_number(
    number_text
):

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
                normalized.append(
                    piece
                )
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

        return "/".join(
            normalized
        )

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
# ADDRESS V3
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


# ============================================================
# ADDRESS V4
# ============================================================

# Generic address words that are common across many records
# and therefore are poor anchors.
#
# This list is deliberately conservative.
GENERIC_ADDRESS_WORDS = {
    "road",
    "rd",
    "street",
    "st",
    "lane",
    "ln",
    "drive",
    "dr",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "highway",
    "hwy",
    "way",
    "place",
    "pl",
    "parkway",
    "pkwy",
    "circle",
    "cir",
    "court",
    "ct",
    "terrace",
    "ter",
    "square",
    "sq",
    "building",
    "bldg",
    "complex",
    "floor",
    "level",
    "block",
    "sector",
    "phase",
    "wing",
    "unit",
    "suite",
    "flat",
    "house",
    "door",
    "room",
    "apt",
    "apartment",
    "no",
    "near",
    "opp",
    "opposite",
    "city",
    "county",
    "district",
    "state",
    "village",
    "town",
    "township",
    "area",
}


def clean_address_tokens_v4(address):

    address_norm = normalize_address(
        address
    )

    if not address_norm:
        return []

    tokens = address_norm.split()

    cleaned = []

    i = 0

    while i < len(tokens):

        token = tokens[i]

        # ----------------------------------------------------
        # Structural labels
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
        # floor
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
        # Numeric tokens
        # ----------------------------------------------------

        if is_numeric_like_token(token):

            i += 1
            continue

        # ----------------------------------------------------
        # Remove punctuation artifacts
        # ----------------------------------------------------

        alpha_chars = [
            ch
            for ch in token
            if ch.isalpha()
        ]

        if not alpha_chars:

            i += 1
            continue

        # ----------------------------------------------------
        # Ignore generic address descriptors
        # ----------------------------------------------------

        token_clean = re.sub(
            r"[^A-Za-z]+",
            "",
            token,
        ).casefold()

        if not token_clean:

            i += 1
            continue

        if token_clean in GENERIC_ADDRESS_WORDS:

            i += 1
            continue

        # ----------------------------------------------------
        # Minimum length
        # ----------------------------------------------------

        if len(token_clean) < MIN_INFORMATIVE_TOKEN_LENGTH:

            i += 1
            continue

        cleaned.append(
            token_clean
        )

        i += 1

    return cleaned


def address_first_v4(address):

    tokens = clean_address_tokens_v4(
        address
    )

    if not tokens:
        return ""

    # Select the longest informative token.
    #
    # Tie-breaker:
    # earliest occurrence in the address.
    best = max(
        tokens,
        key=len,
    )

    return best


# ============================================================
# ALL V3 / V4 KEYS
# ============================================================

def build_keys(
    name,
    address,
    country,
    use_v4,
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
            (
                address_first_v4(address)
                if use_v4
                else ""
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
# PREVIOUS RESULT
# ============================================================

def load_previous_result():

    if not PREVIOUS_RESULTS_PATH.exists():

        raise FileNotFoundError(
            "Previous ALL_V3 result not found:\n"
            f"{PREVIOUS_RESULTS_PATH}"
        )

    df = pd.read_csv(
        PREVIOUS_RESULTS_PATH,
        sep="\t",
    )

    row = df[
        df["variant"]
        == "ALL_V3_ADDRESS_FIRST"
    ]

    if row.empty:

        raise ValueError(
            "ALL_V3_ADDRESS_FIRST was not found in:\n"
            f"{PREVIOUS_RESULTS_PATH}"
        )

    row = row.iloc[0]

    return {
        "variant": "ALL_V3_ADDRESS_FIRST",

        "candidate_recall":
            float(
                row["candidate_recall"]
            ),

        "total_candidates":
            int(
                row["total_candidates"]
            ),

        "avg_candidates_per_source1":
            float(
                row[
                    "avg_candidates_per_source1"
                ]
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n" + "=" * 70)
    print("COMBO_2 ADDRESS-FIRST V4 BENCHMARK")
    print("=" * 70)

    print(
        "\nArchitecture remains:"
        "\n  name_first_v2"
        "\n  OR name_prefix2_v2"
        "\n  OR address_number_v2"
        "\n  OR address_first_v4"
    )

    print(
        "\nONLY address_first changes."
    )

    print(
        "\nV4:"
        "\n  remove structural address words"
        "\n  remove numeric/address-number tokens"
        "\n  remove generic address descriptors"
        "\n  keep informative alphabetic tokens"
        "\n  select longest informative token"
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
    # FILE CHECK
    # ========================================================

    print("\nChecking input files...")

    for path in [
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
        GROUND_TRUTH_PATH,
        PREVIOUS_RESULTS_PATH,
        PREVIOUS_MISSES_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

    print("Input files verified.")

    # ========================================================
    # LOAD ALL_V3 RESULT
    # ========================================================

    previous = load_previous_result()

    print(
        "\nPrevious ALL_V3:"
        f"\n  Recall: "
        f"{previous['candidate_recall']:.2f}%"
        f"\n  Candidates: "
        f"{previous['total_candidates']:,}"
    )

    # ========================================================
    # LOAD PREVIOUS MISSES
    # ========================================================

    previous_misses = pd.read_csv(
        PREVIOUS_MISSES_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    previous_miss_pairs = {
        (
            str(
                row[
                    "source1_entity_id"
                ]
            ),
            str(
                row[
                    "target_entity_id"
                ]
            ),
        )
        for _, row in previous_misses.iterrows()
    }

    print(
        f"\nPrevious ALL_V3 missed pairs: "
        f"{len(previous_miss_pairs):,}"
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

    # ========================================================
    # BUILD SOURCE1 INDEX
    # ========================================================

    print(
        "\nBuilding ALL_V4 Source1 indexes..."
    )

    indexes = {
        "name_first":
            defaultdict(set),

        "name_prefix2":
            defaultdict(set),

        "address_number":
            defaultdict(set),

        "address_first":
            defaultdict(set),
    }

    for _, row in source1_sample.iterrows():

        s1_id = str(
            row["entity_id"]
        )

        keys = build_keys(
            row["business_name"],
            row["business_address"],
            row["country"],
            use_v4=True,
        )

        for key_name, key_value in (
            keys.items()
        ):

            if not key_value[1]:
                continue

            indexes[
                key_name
            ][
                key_value
            ].add(
                s1_id
            )

    print("Indexes created.")

    # ========================================================
    # METRICS
    # ========================================================

    candidate_count = 0

    recovered_pairs = set()

    recovered_previous_misses = set()

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

                keys = build_keys(
                    row["business_name"],
                    row["business_address"],
                    row["country"],
                    use_v4=True,
                )

                # ------------------------------------------------
                # Same four-key UNION structure
                # ------------------------------------------------

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
                            key_name
                        ].get(
                            key_value,
                            set(),
                        )
                    )

                if candidate_s1_ids:

                    candidate_count += len(
                        candidate_s1_ids
                    )

                # ------------------------------------------------
                # TRUE PAIR RECOVERY
                # ------------------------------------------------

                true_s1_ids = (
                    target_to_s1.get(
                        target_id
                    )
                )

                if not true_s1_ids:
                    continue

                recovered_here = (
                    candidate_s1_ids
                    & true_s1_ids
                )

                for s1_id in recovered_here:

                    pair = (
                        s1_id,
                        target_id,
                    )

                    recovered_pairs.add(
                        pair
                    )

                    if pair in previous_miss_pairs:

                        recovered_previous_misses.add(
                            pair
                        )

            print(
                f"\r{source_name}: "
                f"{processed:,} rows",
                end="",
                flush=True,
            )

        print(
            f"\nFinished {source_name} "
            f"in "
            f"{time.time() - source_start:.2f}s"
        )

    # ========================================================
    # CALCULATE
    # ========================================================

    recall = (
        len(recovered_pairs)
        / total_true_matches
        * 100
        if total_true_matches
        else 0
    )

    avg_candidates = (
        candidate_count
        / len(source1_sample)
    )

    recall_gain = (
        recall
        - previous[
            "candidate_recall"
        ]
    )

    candidate_change = (
        candidate_count
        / previous[
            "total_candidates"
        ]
        - 1
    ) * 100

    previous_miss_recovery = (
        len(
            recovered_previous_misses
        )
        / len(
            previous_miss_pairs
        )
        * 100
        if previous_miss_pairs
        else 0
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame([
        {
            "variant":
                previous["variant"],

            "candidate_recall":
                previous[
                    "candidate_recall"
                ],

            "recall_gain_vs_all_v3_pp":
                0.0,

            "total_candidates":
                previous[
                    "total_candidates"
                ],

            "avg_candidates_per_source1":
                previous[
                    "avg_candidates_per_source1"
                ],

            "candidate_change_vs_all_v3_pct":
                0.0,
        },

        {
            "variant":
                "ALL_V4_ADDRESS_FIRST",

            "candidate_recall":
                recall,

            "recall_gain_vs_all_v3_pp":
                recall_gain,

            "total_candidates":
                candidate_count,

            "avg_candidates_per_source1":
                avg_candidates,

            "candidate_change_vs_all_v3_pct":
                candidate_change,
        },
    ])

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("ADDRESS-FIRST V4 RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "-" * 70)
    print("PREVIOUS ALL_V3 MISSES")
    print("-" * 70)

    print(
        f"Previous ALL_V3 misses: "
        f"{len(previous_miss_pairs):,}"
    )

    print(
        f"Recovered by V4: "
        f"{len(recovered_previous_misses):,}"
    )

    print(
        f"Recovery of previous misses: "
        f"{previous_miss_recovery:.2f}%"
    )

    print("\n" + "-" * 70)
    print("RECALL EFFECT")
    print("-" * 70)

    print(
        f"ALL_V3 recall: "
        f"{previous['candidate_recall']:.2f}%"
    )

    print(
        f"ALL_V4 recall: "
        f"{recall:.2f}%"
    )

    print(
        f"Recall gain: "
        f"{recall_gain:+.2f} pp"
    )

    print("\n" + "-" * 70)
    print("CANDIDATE VOLUME EFFECT")
    print("-" * 70)

    print(
        f"ALL_V3 candidates: "
        f"{previous['total_candidates']:,}"
    )

    print(
        f"ALL_V4 candidates: "
        f"{candidate_count:,}"
    )

    print(
        f"Candidate volume change: "
        f"{candidate_change:+.2f}%"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        OUTPUT_RESULTS,
        sep="\t",
        index=False,
    )

    # Recover previous misses into a standalone file.
    previous_misses["pair_key"] = list(
        zip(
            previous_misses[
                "source1_entity_id"
            ].astype(str),

            previous_misses[
                "target_entity_id"
            ].astype(str),
        )
    )

    recovered_df = previous_misses[
        previous_misses[
            "pair_key"
        ].isin(
            recovered_previous_misses
        )
    ].drop(
        columns=["pair_key"]
    )

    recovered_df.to_csv(
        OUTPUT_RECOVERED,
        sep="\t",
        index=False,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_RESULTS}"
    )

    print(
        f"\nRecovered-pair details saved to:"
        f"\n{OUTPUT_RECOVERED}"
    )

    print(
        f"\nRuntime: "
        f"{time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()