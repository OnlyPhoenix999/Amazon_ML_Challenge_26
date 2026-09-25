"""
Candidate Generation / Blocking - Intersection Benchmark

Purpose
-------
Test selective blocking strategies that use INTERSECTIONS of blocking keys.

This benchmark answers:

    Can we retain high candidate recall while reducing
    candidate volume substantially below the broad UNION blocks?

Pipeline
--------
Source1
   ↓
Blocking indexes
   ↓
Intersection-based candidate generation
   ↓
Candidate recall + candidate volume

IMPORTANT
---------
This script does NOT:
    - classify matches
    - use RapidFuzz
    - use Splink
    - build final candidate-pair files

It only benchmarks blocking strategies.
"""

# ============================================================
# IMPORTS + PROJECT ROOT
# ============================================================

import sys
import re
import time
from pathlib import Path
from collections import defaultdict

# candidate_generation.py is:
# repo/src/preprocessing/candidate_generation.py
#
# parents[0] = preprocessing
# parents[1] = src
# parents[2] = repo root

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

OUTPUT_PATH = OUTPUT_DIR / "blocking_intersections.tsv"

S1_SAMPLE_SIZE = 2000
RANDOM_STATE = 42
CHUNK_SIZE = 100_000


# ============================================================
# BLOCKING STRATEGIES
# ============================================================

# A strategy is an INTERSECTION of blocking keys.
#
# Example:
#   ("name_first", "address_number")
#
# means:
#   same country
#   AND same first name token
#   AND same first address number
#
# The union strategy below combines several selective
# intersections.

INTERSECTION_STRATEGIES = {
    "I1_name_first_AND_address_number": (
        "name_first",
        "address_number",
    ),

    "I2_name_prefix2_AND_address_number": (
        "name_prefix2",
        "address_number",
    ),

    "I3_name_prefix2_AND_address_first": (
        "name_prefix2",
        "address_first",
    ),

    "I4_name_first_AND_address_first": (
        "name_first",
        "address_first",
    ),

    "I5_name_first_AND_name_prefix2_AND_address_number": (
        "name_first",
        "name_prefix2",
        "address_number",
    ),
}


# Union of the most useful selective intersections.
#
# This is intentionally separate from the individual
# intersection strategies so we can see whether combining
# selective blocks gives us a good recall/volume tradeoff.

UNION_STRATEGY_NAME = "UNION_SELECTIVE_INTERSECTIONS"

UNION_COMPONENTS = (
    "I1_name_first_AND_address_number",
    "I2_name_prefix2_AND_address_number",
    "I3_name_prefix2_AND_address_first",
    "I4_name_first_AND_address_first",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def first_token(text):
    """Return first normalized token."""
    if not text:
        return ""

    parts = str(text).split()

    return parts[0] if parts else ""


def last_token(text):
    """Return last normalized token."""
    if not text:
        return ""

    parts = str(text).split()

    return parts[-1] if parts else ""


def first_two_chars(text):
    """
    Return first two non-space characters of normalized text.
    """
    if not text:
        return ""

    compact = str(text).replace(" ", "")

    return compact[:2]


def first_address_token(text):
    """Return first normalized address token."""
    return first_token(text)


def first_address_number(text):
    """
    Extract first numeric component from an address.

    Examples
    --------
    '123 Main Street' -> '123'
    '12A MG Road'     -> '12'
    """
    if not text:
        return ""

    match = re.search(r"\d+", str(text))

    if match:
        return match.group(0)

    return ""


def make_block_keys(name, address, country):
    """
    Generate all blocking keys for one record.

    Every key includes country as the first element.
    """

    name_norm = normalize_business_name(name)
    address_norm = normalize_address(address)

    country_norm = str(country).strip().casefold()

    return {
        "name_first": (
            country_norm,
            first_token(name_norm),
        ),

        "name_last": (
            country_norm,
            last_token(name_norm),
        ),

        "name_prefix2": (
            country_norm,
            first_two_chars(name_norm),
        ),

        "address_first": (
            country_norm,
            first_address_token(address_norm),
        ),

        "address_number": (
            country_norm,
            first_address_number(address_norm),
        ),
    }


def parse_ground_truth(value):
    """
    Parse comma-separated matched_entity_ids.

    Empty / missing value means zero matches.
    """

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


def intersection_sets(index_sets):
    """
    Intersect a sequence of sets.

    Returns an empty set when there is no usable set.
    """

    if not index_sets:
        return set()

    # Start with smallest set for efficiency.
    ordered = sorted(index_sets, key=len)

    result = ordered[0].copy()

    for current in ordered[1:]:
        result.intersection_update(current)

        if not result:
            break

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    benchmark_start = time.time()

    print("\n" + "=" * 70)
    print("CANDIDATE GENERATION / INTERSECTION BLOCKING BENCHMARK")
    print("=" * 70)

    print("\nIntersection strategies:")

    for name, components in INTERSECTION_STRATEGIES.items():
        print(f"  {name}: " + " + ".join(components))

    print(
        f"\n  {UNION_STRATEGY_NAME}:"
        f"\n    UNION of:"
    )

    for component in UNION_COMPONENTS:
        print(f"      - {component}")

    print(f"\nSample size: {S1_SAMPLE_SIZE:,}")
    print(f"Chunk size:  {CHUNK_SIZE:,}")

    # ========================================================
    # CHECK INPUTS
    # ========================================================

    print("\nChecking input files...")

    required_files = [
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
        GROUND_TRUTH_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found:\n{path}"
            )

    print("Input files exist.")

    # ========================================================
    # LOAD SOURCE1
    # ========================================================

    print("\nLoading Source1...")

    source1_start = time.time()

    source1 = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Total Source1 rows: {len(source1):,}")
    print(
        f"Source1 loaded in "
        f"{time.time() - source1_start:.2f}s"
    )

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_s1_columns = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    missing_s1_columns = (
        required_s1_columns - set(source1.columns)
    )

    if missing_s1_columns:
        raise ValueError(
            "Source1 is missing required columns: "
            f"{sorted(missing_s1_columns)}"
        )

    # ========================================================
    # SAMPLE SOURCE1
    # ========================================================

    print("\nSampling Source1...")

    if len(source1) > S1_SAMPLE_SIZE:

        source1_sample = source1.sample(
            n=S1_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        ).copy()

    else:

        source1_sample = source1.copy()

    source1_sample = source1_sample.reset_index(drop=True)

    sampled_s1_ids = (
        source1_sample["entity_id"]
        .astype(str)
        .tolist()
    )

    sampled_s1_id_set = set(sampled_s1_ids)

    print(
        f"Source1 benchmark sample: "
        f"{len(source1_sample):,}"
    )

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    print("\nLoading ground truth...")

    gt_start = time.time()

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Total ground-truth rows: "
        f"{len(ground_truth):,}"
    )

    print(
        f"Ground truth loaded in "
        f"{time.time() - gt_start:.2f}s"
    )

    required_gt_columns = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing_gt_columns = (
        required_gt_columns - set(ground_truth.columns)
    )

    if missing_gt_columns:
        raise ValueError(
            "Ground truth is missing required columns: "
            f"{sorted(missing_gt_columns)}"
        )

    # ========================================================
    # BUILD GROUND TRUTH LOOKUP
    # ========================================================

    print("\nBuilding sampled ground-truth lookup...")

    ground_truth_lookup = {}

    sampled_gt_rows = 0

    for _, row in ground_truth.iterrows():

        s1_id = str(row["source1_entity_id"])

        if s1_id not in sampled_s1_id_set:
            continue

        ground_truth_lookup[s1_id] = parse_ground_truth(
            row["matched_entity_ids"]
        )

        sampled_gt_rows += 1

        if sampled_gt_rows == len(sampled_s1_ids):
            break

    # Make sure every sampled Source1 entity exists
    # in the lookup.
    for s1_id in sampled_s1_ids:
        ground_truth_lookup.setdefault(
            s1_id,
            set(),
        )

    total_true_matches = sum(
        len(ground_truth_lookup[s1_id])
        for s1_id in sampled_s1_ids
    )

    zero_match_entities = sum(
        len(ground_truth_lookup[s1_id]) == 0
        for s1_id in sampled_s1_ids
    )

    print(
        f"Ground-truth rows used for benchmark: "
        f"{sampled_gt_rows:,}"
    )

    print(
        f"Sampled Source1 entities with zero matches: "
        f"{zero_match_entities:,}"
    )

    print(
        f"Total true matches in sample: "
        f"{total_true_matches:,}"
    )

    # ========================================================
    # BUILD SOURCE1 BLOCKING INDEXES
    # ========================================================

    print("\nBuilding Source1 blocking indexes...")

    index_start = time.time()

    # block_type -> block_key -> set(Source1 IDs)
    s1_indexes = {
        "name_first": defaultdict(set),
        "name_last": defaultdict(set),
        "name_prefix2": defaultdict(set),
        "address_first": defaultdict(set),
        "address_number": defaultdict(set),
    }

    for _, row in source1_sample.iterrows():

        s1_id = str(row["entity_id"])

        keys = make_block_keys(
            row["business_name"],
            row["business_address"],
            row["country"],
        )

        for block_type, block_key in keys.items():

            # block_key[0] = country
            # block_key[1] = actual blocking value
            if not block_key[1]:
                continue

            s1_indexes[block_type][block_key].add(
                s1_id
            )

    print("Source1 indexes created.")

    print(
        f"Index construction time: "
        f"{time.time() - index_start:.2f}s"
    )

    # ========================================================
    # METRICS STORAGE
    # ========================================================

    # For each intersection strategy:
    #
    # candidate_count
    #     Total unique (S1, target) pairs generated.
    #
    # recovered_true_ids
    #     True target IDs recovered at least once.
    #
    # We only retain RECOVERED TRUE IDS, which is tiny
    # compared with the entire candidate graph.
    #
    # Structure:
    #
    # recovered_true_ids[strategy][s1_id] = set(target_ids)

    strategy_names = list(INTERSECTION_STRATEGIES.keys())

    candidate_counts = {
        name: 0
        for name in strategy_names
    }

    recovered_true_ids = {
        name: defaultdict(set)
        for name in strategy_names
    }

    union_candidate_count = 0

    union_recovered_true_ids = defaultdict(set)

    # ========================================================
    # SCAN TARGET SOURCES
    # ========================================================

    target_files = [
        ("source2", SOURCE2_PATH),
        ("source3", SOURCE3_PATH),
    ]

    total_scan_start = time.time()

    for source_name, source_path in target_files:

        print("\n" + "-" * 70)
        print(f"Scanning {source_name}: {source_path.name}")
        print("-" * 70)

        source_start = time.time()

        total_rows = 0

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ):

            total_rows += len(chunk)

            print(
                f"\r{source_name}: processed "
                f"{total_rows:,} rows",
                end="",
                flush=True,
            )

            for _, row in chunk.iterrows():

                target_id = str(row["entity_id"])

                keys = make_block_keys(
                    row["business_name"],
                    row["business_address"],
                    row["country"],
                )

                # ------------------------------------------------
                # Find Source1 IDs for each individual blocking key
                # ------------------------------------------------

                matching_ids = {}

                for block_type, block_key in keys.items():

                    actual_value = block_key[1]

                    if not actual_value:
                        matching_ids[block_type] = set()
                        continue

                    matching_ids[block_type] = s1_indexes[
                        block_type
                    ].get(
                        block_key,
                        set(),
                    )

                # ------------------------------------------------
                # Evaluate every intersection strategy
                # ------------------------------------------------

                strategy_candidate_s1_ids = {}

                for strategy_name, components in (
                    INTERSECTION_STRATEGIES.items()
                ):

                    component_sets = [
                        matching_ids[component]
                        for component in components
                    ]

                    # If one component has no matches,
                    # the intersection is automatically empty.
                    if any(
                        not current
                        for current in component_sets
                    ):
                        intersection = set()

                    else:
                        intersection = intersection_sets(
                            component_sets
                        )

                    strategy_candidate_s1_ids[
                        strategy_name
                    ] = intersection

                    if not intersection:
                        continue

                    # Every S1 ID in this intersection creates
                    # one unique candidate pair for this strategy.
                    candidate_counts[strategy_name] += len(
                        intersection
                    )

                    # ------------------------------------------------
                    # Only keep recovered TRUE positives.
                    # Do NOT store all candidate pairs.
                    # ------------------------------------------------

                    for s1_id in intersection:

                        true_ids = ground_truth_lookup.get(
                            s1_id,
                            set(),
                        )

                        if target_id in true_ids:

                            recovered_true_ids[
                                strategy_name
                            ][s1_id].add(
                                target_id
                            )

                # ------------------------------------------------
                # UNION of selective intersections
                # ------------------------------------------------

                #
                # Important:
                # We need UNIQUE candidate pairs across the
                # component intersections.
                #
                # For one target row we can simply union all
                # matching S1 IDs from the four components.
                #

                union_s1_ids = set()

                for component_name in UNION_COMPONENTS:

                    union_s1_ids.update(
                        strategy_candidate_s1_ids[
                            component_name
                        ]
                    )

                if union_s1_ids:

                    union_candidate_count += len(
                        union_s1_ids
                    )

                    for s1_id in union_s1_ids:

                        true_ids = ground_truth_lookup.get(
                            s1_id,
                            set(),
                        )

                        if target_id in true_ids:

                            union_recovered_true_ids[
                                s1_id
                            ].add(
                                target_id
                            )

        elapsed = time.time() - source_start

        print(
            f"\nFinished {source_name}: "
            f"{total_rows:,} rows"
        )

        print(
            f"{source_name} scan time: "
            f"{elapsed:.2f}s"
        )

    total_scan_time = time.time() - total_scan_start

    print(
        f"\nTotal target scan time: "
        f"{total_scan_time:.2f}s"
    )

    # ========================================================
    # EVALUATE RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("INTERSECTION BLOCKING RESULTS")
    print("=" * 70)

    rows = []

    # --------------------------------------------------------
    # Individual intersection strategies
    # --------------------------------------------------------

    for strategy_name in strategy_names:

        recovered_count = sum(
            len(ids)
            for ids in recovered_true_ids[
                strategy_name
            ].values()
        )

        candidate_count = candidate_counts[
            strategy_name
        ]

        recall = (
            recovered_count / total_true_matches
            if total_true_matches
            else 0.0
        )

        avg_candidates = (
            candidate_count / len(source1_sample)
            if len(source1_sample)
            else 0.0
        )

        rows.append({
            "strategy": strategy_name,
            "candidate_recall": recall * 100.0,
            "avg_candidates_per_source1": avg_candidates,
            "total_candidates": candidate_count,
            "candidate_reduction_vs_union_intersections_pct": None,
        })

    # --------------------------------------------------------
    # Union of selective intersections
    # --------------------------------------------------------

    union_recovered_count = sum(
        len(ids)
        for ids in union_recovered_true_ids.values()
    )

    union_recall = (
        union_recovered_count / total_true_matches
        if total_true_matches
        else 0.0
    )

    union_avg_candidates = (
        union_candidate_count / len(source1_sample)
        if len(source1_sample)
        else 0.0
    )

    rows.append({
        "strategy": UNION_STRATEGY_NAME,
        "candidate_recall": union_recall * 100.0,
        "avg_candidates_per_source1": union_avg_candidates,
        "total_candidates": union_candidate_count,
        "candidate_reduction_vs_union_intersections_pct": 0.0,
    })

    # --------------------------------------------------------
    # Build DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Compute reduction relative to selective-intersection
    # union
    # --------------------------------------------------------

    if union_candidate_count > 0:

        results_df[
            "candidate_reduction_vs_union_intersections_pct"
        ] = (
            1.0
            - (
                results_df["total_candidates"]
                / union_candidate_count
            )
        ) * 100.0

    # Sort:
    #   highest recall first
    #   then lowest candidate volume
    results_df = results_df.sort_values(
        by=[
            "candidate_recall",
            "total_candidates",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # ========================================================
    # EXTRA SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(
        f"Total true matches: "
        f"{total_true_matches:,}"
    )

    for _, result in results_df.iterrows():

        strategy = result["strategy"]

        recall = result["candidate_recall"]

        candidates = int(
            result["total_candidates"]
        )

        avg_candidates = result[
            "avg_candidates_per_source1"
        ]

        print(
            f"\n{strategy}"
            f"\n  recall: {recall:.2f}%"
            f"\n  total candidates: {candidates:,}"
            f"\n  avg candidates / S1: {avg_candidates:,.2f}"
        )

    # ========================================================
    # SAVE RESULTS
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
        f"Total runtime: "
        f"{time.time() - benchmark_start:.2f}s"
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()