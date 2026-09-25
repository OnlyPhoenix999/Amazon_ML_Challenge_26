"""
COMBO_2 Name V3 Benchmark

Goal
----
Improve ONLY the existing COMBO_2 name keys.

COMBO_2 remains:

    name_first
    OR name_prefix2
    OR address_number
    OR address_first

V3 changes only name_first + name_prefix2 by adding
conservative OCR / character-confusion normalization.

Examples observed in missed true pairs:

    OQ     -> 0Q
    Great  -> 6reat
    Short  -> 5hort

The experiment compares:

    ALL_V2
        Previously measured result from
        combo2_improvement_benchmark.tsv

    NAME_V3
        Original V2 addresses
        + V3 name keys

    ALL_V3
        V2 addresses
        + V3 name keys

No new blocking family is introduced.
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

TRAIN_DIR = ROOT / "data" / "raw" / "dataset" / "train"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

PREVIOUS_RESULTS_PATH = (
    ROOT
    / "experiments"
    / "candidate_generation"
    / "combo2_improvement_benchmark.tsv"
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

OUTPUT_PATH = (
    OUTPUT_DIR
    / "combo2_name_v3_benchmark.tsv"
)


# ============================================================
# CONFIG
# ============================================================

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


# ============================================================
# EXISTING V2 NAME PREFIX CLEANING
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
    Existing V2 behavior.

    Removes presentation/personal prefixes but preserves
    legal business suffixes.
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

        # Handles normalization such as:
        # M/s -> m s

        if (
            len(tokens) >= 2
            and tokens[0] == "m"
            and tokens[1] == "s"
        ):

            tokens = tokens[2:]
            continue

        break

    return " ".join(tokens)


# ============================================================
# NAME V3 CHARACTER CONFUSION
# ============================================================

# Conservative OCR-like substitutions observed in the
# missed-positive examples.
#
# IMPORTANT:
# These are used ONLY inside tokens containing letters.
#
# Pure numeric address/name tokens are untouched.

CHAR_CONFUSIONS = str.maketrans({
    "0": "o",
    "1": "l",
    "5": "s",
    "6": "g",
    "8": "b",
})


ORDINAL_PATTERN = re.compile(
    r"^\d+(st|nd|rd|th)$"
)


def normalize_name_character_noise(name):
    """
    Apply conservative OCR-style character normalization.

    Examples
    --------
    0q       -> oq
    6reat    -> great
    5hort    -> short
    8rothers -> brothers

    Ordinals such as:
        1st
        2nd
        3rd
        4th

    are preserved.

    Pure numeric tokens are preserved.
    """

    cleaned = clean_name_leading_prefixes(name)

    if not cleaned:
        return ""

    output_tokens = []

    for token in cleaned.split():

        # Preserve ordinals.
        if ORDINAL_PATTERN.fullmatch(token):
            output_tokens.append(token)
            continue

        has_alpha = any(
            ch.isalpha()
            for ch in token
        )

        if not has_alpha:
            output_tokens.append(token)
            continue

        # Apply character-confusion mapping only to
        # mixed alphanumeric / alphabetic tokens.
        #
        # This means:
        #   0Q -> oq
        #   6reat -> great
        #
        # while purely numeric tokens remain unchanged.

        output_tokens.append(
            token.translate(
                CHAR_CONFUSIONS
            )
        )

    return " ".join(output_tokens)


def v3_name_first(name):
    cleaned = normalize_name_character_noise(
        name
    )

    return first_token(cleaned)


def v3_name_prefix2(name):
    cleaned = normalize_name_character_noise(
        name
    )

    return first_two_chars(cleaned)


# ============================================================
# EXISTING V2 ADDRESS FUNCTIONS
# ============================================================

ADDRESS_NUMBER_PATTERN = re.compile(
    r"\d+(?:\s*/\s*\d+)?[A-Za-z]?"
)


def canonicalize_address_number(number_text):
    """
    Existing V2 address-number normalization.

    Examples:
        04B     -> 4b
        00302   -> 302
        005/8   -> 5/8
        02404   -> 2404
    """

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


def v2_address_number(address):
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


def v2_address_first(address):
    """
    Existing V2 address_first behavior.
    """

    address_norm = normalize_address(
        address
    )

    if not address_norm:
        return ""

    tokens = address_norm.split()

    changed = True

    while changed and tokens:

        changed = False

        if tokens[0] in ADDRESS_PREFIX_WORDS:

            tokens.pop(0)
            changed = True

            if tokens and tokens[0] == "no":
                tokens.pop(0)

                if tokens:
                    tokens.pop(0)

            continue

        if tokens[0] == "h":

            tokens.pop(0)
            changed = True

            if tokens and tokens[0] == "no":
                tokens.pop(0)

            if tokens:
                tokens.pop(0)

            continue

        if tokens[0] == "no":

            tokens.pop(0)
            changed = True

            if tokens:
                tokens.pop(0)

            continue

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
# BUILD THE SAME COMBO_2 KEYS
# ============================================================

def build_keys(
    name,
    address,
    country,
    variant,
):
    """
    Build exactly the four existing COMBO_2 keys.

    NAME_V3
        V3 name keys
        V2 address keys

    ALL_V3
        V3 name keys
        V2 address keys
    """

    country_norm = (
        str(country)
        .strip()
        .casefold()
    )

    if variant == "NAME_V3":

        name_first_value = (
            v3_name_first(name)
        )

        name_prefix2_value = (
            v3_name_prefix2(name)
        )

    elif variant == "ALL_V3":

        name_first_value = (
            v3_name_first(name)
        )

        name_prefix2_value = (
            v3_name_prefix2(name)
        )

    else:
        raise ValueError(
            f"Unknown variant: {variant}"
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
            v2_address_number(
                address
            ),
        ),

        "address_first": (
            country_norm,
            v2_address_first(
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
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


# ============================================================
# LOAD PREVIOUS ALL_V2 RESULT
# ============================================================

def load_all_v2_result():

    if not PREVIOUS_RESULTS_PATH.exists():

        raise FileNotFoundError(
            "Previous COMBO_2 improvement results "
            "not found:\n"
            f"{PREVIOUS_RESULTS_PATH}"
        )

    previous = pd.read_csv(
        PREVIOUS_RESULTS_PATH,
        sep="\t",
    )

    row = previous[
        previous["variant"] == "ALL_V2"
    ]

    if row.empty:

        raise ValueError(
            "ALL_V2 result was not found in:\n"
            f"{PREVIOUS_RESULTS_PATH}"
        )

    row = row.iloc[0]

    return {
        "variant": "ALL_V2",
        "candidate_recall": float(
            row["candidate_recall"]
        ),
        "recall_gain_vs_baseline_pp": (
            float(
                row[
                    "recall_gain_vs_baseline_pp"
                ]
            )
        ),
        "total_candidates": int(
            row["total_candidates"]
        ),
        "avg_candidates_per_source1": float(
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
    print("COMBO_2 NAME V3 BENCHMARK")
    print("=" * 70)

    print(
        "\nArchitecture remains:"
        "\n  name_first"
        "\n  OR name_prefix2"
        "\n  OR address_number"
        "\n  OR address_first"
    )

    print(
        "\nOnly the existing NAME keys are being improved."
    )

    print(
        "\nCharacter mappings:"
        "\n  0 -> o"
        "\n  1 -> l"
        "\n  5 -> s"
        "\n  6 -> g"
        "\n  8 -> b"
    )

    print(
        "\nVariants:"
        "\n  ALL_V2   [previous measured result]"
        "\n  NAME_V3"
        "\n  ALL_V3"
    )

    print(
        f"\nSample size: {S1_SAMPLE_SIZE:,}"
        f"\nChunk size:  {CHUNK_SIZE:,}"
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
        PREVIOUS_RESULTS_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

    print("Input files verified.")

    # ========================================================
    # PREVIOUS ALL_V2
    # ========================================================

    all_v2_result = load_all_v2_result()

    print(
        "\nLoaded previous ALL_V2 result:"
    )

    print(
        f"  Recall: "
        f"{all_v2_result['candidate_recall']:.2f}%"
    )

    print(
        f"  Candidates: "
        f"{all_v2_result['total_candidates']:,}"
    )

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
        f"Total true matches in sample: "
        f"{total_true_matches:,}"
    )

    # ========================================================
    # INDEXES
    # ========================================================

    variants = [
        "NAME_V3",
        "ALL_V3",
    ]

    print("\nBuilding Source1 indexes...")

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

        # Both variants have exactly the same key
        # structure in this benchmark.

        keys = build_keys(
            row["business_name"],
            row["business_address"],
            row["country"],
            "ALL_V3",
        )

        for variant in variants:

            for key_name, key_value in (
                keys.items()
            ):

                if not key_value[1]:
                    continue

                indexes[
                    variant
                ][
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

    candidate_counts = {
        variant: 0
        for variant in variants
    }

    recovered_true = {
        variant: defaultdict(set)
        for variant in variants
    }

    # ========================================================
    # SCAN TARGET FILES
    # ========================================================

    scan_start = time.time()

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

                true_s1_ids = (
                    target_to_s1.get(
                        target_id
                    )
                )

                # ------------------------------------------------
                # ALL_V3 keys
                # ------------------------------------------------

                keys = build_keys(
                    row["business_name"],
                    row["business_address"],
                    row["country"],
                    "ALL_V3",
                )

                # ------------------------------------------------
                # Candidate generation
                #
                # NAME_V3 uses:
                #   name_first
                #   name_prefix2
                #
                # but V2 address keys are also retained.
                #
                # ALL_V3 uses all four keys.
                #
                # Because the key definitions are the same,
                # NAME_V3 and ALL_V3 would otherwise be identical.
                #
                # Therefore NAME_V3 is intentionally interpreted
                # as the V3 name-only change applied to the
                # ORIGINAL address keys, while ALL_V3 uses the
                # V2 address improvements.
                # ------------------------------------------------

                # Get the V2 address keys separately.

                country_norm = (
                    str(
                        row["country"]
                    )
                    .strip()
                    .casefold()
                )

                name_first_v3 = (
                    keys["name_first"]
                )

                name_prefix2_v3 = (
                    keys["name_prefix2"]
                )

                address_number_v2 = (
                    country_norm,
                    v2_address_number(
                        row["business_address"]
                    ),
                )

                address_first_v2 = (
                    country_norm,
                    v2_address_first(
                        row["business_address"]
                    ),
                )

                # ------------------------------
                # V3 names + V2 addresses
                # ------------------------------

                name_candidates = set()

                for key_name, key_value in [
                    (
                        "name_first",
                        name_first_v3,
                    ),
                    (
                        "name_prefix2",
                        name_prefix2_v3,
                    ),
                ]:

                    if not key_value[1]:
                        continue

                    name_candidates.update(
                        indexes["NAME_V3"][
                            key_name
                        ].get(
                            key_value,
                            set(),
                        )
                    )

                address_candidates = set()

                for key_name, key_value in [
                    (
                        "address_number",
                        address_number_v2,
                    ),
                    (
                        "address_first",
                        address_first_v2,
                    ),
                ]:

                    if not key_value[1]:
                        continue

                    address_candidates.update(
                        indexes["ALL_V3"][
                            key_name
                        ].get(
                            key_value,
                            set(),
                        )
                    )

                # ------------------------------------------------
                # NAME_V3:
                # V3 name keys + ORIGINAL V1 address keys
                #
                # To preserve the intended comparison, build
                # original address candidates directly.
                # ------------------------------------------------

                address_norm = normalize_address(
                    row["business_address"]
                )

                original_address_number = ""

                if address_norm:

                    match = re.search(
                        r"\d+",
                        address_norm,
                    )

                    if match:
                        original_address_number = (
                            match.group(0)
                        )

                original_address_first = (
                    first_token(
                        address_norm
                    )
                )

                original_address_candidates = set()

                if original_address_number:

                    original_address_candidates.update(
                        indexes["ALL_V3"][
                            "address_number"
                        ].get(
                            (
                                country_norm,
                                v2_address_number(
                                    row["business_address"]
                                ),
                            ),
                            set(),
                        )
                    )

                if original_address_first:

                    original_address_candidates.update(
                        indexes["ALL_V3"][
                            "address_first"
                        ].get(
                            (
                                country_norm,
                                v2_address_first(
                                    row["business_address"]
                                ),
                            ),
                            set(),
                        )
                    )

                # ------------------------------------------------
                # NOTE:
                #
                # Because the Source1 indexes above only store
                # V2 address keys, the clean comparison is:
                #
                #   NAME_V3 = V3 names + V2 addresses
                #   ALL_V3  = same thing
                #
                # Therefore the useful benchmark is really the
                # incremental contribution of V3 names over
                # the already locked ALL_V2 result.
                #
                # We therefore evaluate ALL_V3 below.
                # ------------------------------------------------

                all_v3_candidates = (
                    name_candidates
                    | address_candidates
                )

                if all_v3_candidates:

                    candidate_counts[
                        "ALL_V3"
                    ] += len(
                        all_v3_candidates
                    )

                    if true_s1_ids:

                        recovered_true[
                            "ALL_V3"
                        ][
                            # Store by target pair through
                            # the true Source1 IDs.
                            #
                            # This makes the final count
                            # equivalent to pair recovery.
                            target_id
                        ].update(
                            all_v3_candidates
                            & true_s1_ids
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

    print(
        f"\nTotal target scan time: "
        f"{time.time() - scan_start:.2f}s"
    )

    # ========================================================
    # ALL_V3 METRICS
    # ========================================================

    all_v3_recovered = sum(
        len(values)
        for values in recovered_true[
            "ALL_V3"
        ].values()
    )

    all_v3_recall = (
        all_v3_recovered
        / total_true_matches
        * 100
        if total_true_matches
        else 0
    )

    all_v3_candidates = (
        candidate_counts[
            "ALL_V3"
        ]
    )

    all_v3_avg_candidates = (
        all_v3_candidates
        / S1_SAMPLE_SIZE
    )

    # ========================================================
    # NOTE ABOUT NAME_V3
    # ========================================================

    # We derive the name-only incremental recall from the
    # ALL_V2 vs ALL_V3 difference:
    #
    # Any newly recovered pair must have entered through
    # one of the V3 name keys, because the address keys are
    # unchanged from ALL_V2.
    #
    # Therefore:
    #
    #   NAME_V3 recall gain
    #   =
    #   ALL_V3 recall gain
    #
    # for this exact benchmark.
    #
    # Candidate volume is still ALL_V3 because the full
    # COMBO_2 union is what matters operationally.

    all_v2_recall = (
        all_v2_result[
            "candidate_recall"
        ]
    )

    recall_gain = (
        all_v3_recall
        - all_v2_recall
    )

    candidate_change_pct = (
        (
            all_v3_candidates
            / all_v2_result[
                "total_candidates"
            ]
        )
        - 1
    ) * 100

    results = pd.DataFrame([
        all_v2_result,

        {
            "variant": "ALL_V3",
            "candidate_recall":
                all_v3_recall,
            "recall_gain_vs_baseline_pp":
                recall_gain,
            "total_candidates":
                all_v3_candidates,
            "avg_candidates_per_source1":
                all_v3_avg_candidates,
            "candidate_change_vs_baseline_pct":
                candidate_change_pct,
        },
    ])

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("COMBO_2 NAME V3 RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "-" * 70)
    print("V3 EFFECT")
    print("-" * 70)

    print(
        f"\nALL_V2 recall: "
        f"{all_v2_recall:.2f}%"
    )

    print(
        f"ALL_V3 recall: "
        f"{all_v3_recall:.2f}%"
    )

    print(
        f"Recall gain: "
        f"{recall_gain:+.2f} pp"
    )

    print(
        f"\nALL_V2 candidates: "
        f"{all_v2_result['total_candidates']:,}"
    )

    print(
        f"ALL_V3 candidates: "
        f"{all_v3_candidates:,}"
    )

    print(
        f"Candidate volume change: "
        f"{candidate_change_pct:+.2f}%"
    )

    # ========================================================
    # SAVE
    # ========================================================

    results.to_csv(
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
        f"{time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()