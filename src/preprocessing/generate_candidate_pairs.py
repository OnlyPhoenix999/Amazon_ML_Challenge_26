"""Production ALL_V4 candidate-pair generator.

Uses the exact ALL_V4 blocking logic from
benchmark_combo2_address_first_v4.py.

Run from the repository root:

    python src/preprocessing/generate_candidate_pairs.py --max-target-rows 10000

After the test is validated, the full run is:

    python src/preprocessing/generate_candidate_pairs.py
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# IMPORTANT:
# This allows "from src..." to work even when this file is
# launched directly with:
# python src/preprocessing/generate_candidate_pairs.py
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# REUSE THE VALIDATED ALL_V4 LOGIC
# ============================================================

from src.preprocessing.benchmark_combo2_address_first_v4 import (
    SOURCE1_PATH,
    SOURCE2_PATH,
    SOURCE3_PATH,
    build_keys,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "candidate_pairs.tsv"


# ============================================================
# SOURCE1 INDEX
# ============================================================

def build_source1_indexes(source1):
    """Build the four country-aware ALL_V4 blocking indexes."""

    indexes = {
        "name_first": defaultdict(set),
        "name_prefix2": defaultdict(set),
        "address_number": defaultdict(set),
        "address_first": defaultdict(set),
    }

    for row in source1.itertuples(index=False):

        s1_id = str(row.entity_id)

        keys = build_keys(
            row.business_name,
            row.business_address,
            row.country,
            use_v4=True,
        )

        for key_name, key_value in keys.items():

            # Never index an empty blocking key.
            if key_value[1]:
                indexes[key_name][key_value].add(s1_id)

    return indexes


# ============================================================
# GENERATE CANDIDATES FOR ONE CHUNK
# ============================================================

def generate_chunk(chunk, indexes):
    """
    Generate candidate pairs for one Source2/Source3 chunk.

    Returns:
        list of (source1_entity_id, target_entity_id)
    """

    pairs = []

    for row in chunk.itertuples(index=False):

        target_id = str(row.entity_id)

        keys = build_keys(
            row.business_name,
            row.business_address,
            row.country,
            use_v4=True,
        )

        candidate_s1_ids = set()

        # EXACT ALL_V4 UNION
        #
        # country + name_first_v2
        # OR
        # country + name_prefix2_v2
        # OR
        # country + address_number_v2
        # OR
        # country + address_first_v4

        for key_name in (
            "name_first",
            "name_prefix2",
            "address_number",
            "address_first",
        ):

            key_value = keys[key_name]

            if not key_value[1]:
                continue

            candidate_s1_ids.update(
                indexes[key_name].get(
                    key_value,
                    (),
                )
            )

        for s1_id in candidate_s1_ids:
            pairs.append(
                (s1_id, target_id)
            )

    return pairs


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate ALL_V4 candidate_pairs.tsv safely."
    )

    parser.add_argument(
        "--max-target-rows",
        type=int,
        default=None,
        help=(
            "Maximum TOTAL Source2+Source3 rows to process. "
            "Use this for testing."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Number of target rows processed at once. "
            "10,000 is conservative for a laptop."
        ),
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output candidate-pair TSV path.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be greater than 0."
        )

    if (
        args.max_target_rows is not None
        and args.max_target_rows <= 0
    ):
        raise ValueError(
            "--max-target-rows must be greater than 0."
        )

    if output_path.exists():
        raise FileExistsError(
            f"\nOutput already exists:\n"
            f"{output_path}\n\n"
            "Delete it first or use --output "
            "to specify another file."
        )

    for path in (
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found:\n{path}"
            )

    # ========================================================
    # START
    # ========================================================

    print("=" * 70)
    print("ALL_V4 CANDIDATE PAIR GENERATION")
    print("=" * 70)

    print(
        f"Chunk size: {args.chunk_size:,}"
    )

    if args.max_target_rows is None:
        print("Target limit: FULL DATASET")
    else:
        print(
            f"Target limit: {args.max_target_rows:,} "
            "(TEST MODE)"
        )

    print(
        f"Output: {output_path}"
    )

    # ========================================================
    # LOAD SOURCE1 + BUILD INDEXES
    # ========================================================

    print(
        "\nLoading Source1 and building indexes..."
    )

    index_start = time.time()

    source1 = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Source1 rows: {len(source1):,}"
    )

    indexes = build_source1_indexes(
        source1
    )

    # Source1 dataframe is no longer needed.
    del source1

    print(
        f"Indexes ready in "
        f"{time.time() - index_start:.2f}s"
    )

    # ========================================================
    # OUTPUT STATE
    # ========================================================

    first_write = True

    total_targets = 0
    total_pairs = 0

    run_start = time.time()

    # ========================================================
    # PROCESS SOURCE2 + SOURCE3
    # ========================================================

    for source_name, source_path in (
        ("source2", SOURCE2_PATH),
        ("source3", SOURCE3_PATH),
    ):

        if (
            args.max_target_rows is not None
            and total_targets >= args.max_target_rows
        ):
            break

        source_start = time.time()

        source_targets = 0
        source_pairs = 0

        print(
            "\n" + "-" * 70
        )

        print(
            f"Processing {source_name}: "
            f"{source_path.name}"
        )

        print(
            "-" * 70
        )

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=args.chunk_size,
            keep_default_na=False,
        ):

            # ------------------------------------------------
            # Respect TOTAL test limit.
            # ------------------------------------------------

            if args.max_target_rows is not None:

                remaining = (
                    args.max_target_rows
                    - total_targets
                )

                if remaining <= 0:
                    break

                if len(chunk) > remaining:
                    chunk = chunk.iloc[
                        :remaining
                    ].copy()

            if chunk.empty:
                break

            # ------------------------------------------------
            # Generate only this chunk's candidates.
            # ------------------------------------------------

            pairs = generate_chunk(
                chunk,
                indexes,
            )

            # ------------------------------------------------
            # STREAM CURRENT CHUNK TO DISK.
            # ------------------------------------------------

            if pairs:

                pairs_df = pd.DataFrame(
                    pairs,
                    columns=[
                        "source1_entity_id",
                        "target_entity_id",
                    ],
                )

                pairs_df.to_csv(
                    output_path,
                    sep="\t",
                    index=False,
                    mode=(
                        "w"
                        if first_write
                        else "a"
                    ),
                    header=first_write,
                )

                first_write = False

                pair_count = len(pairs)

                source_pairs += pair_count
                total_pairs += pair_count

                del pairs_df

            # ------------------------------------------------
            # UPDATE COUNTERS.
            # ------------------------------------------------

            processed = len(chunk)

            source_targets += processed
            total_targets += processed

            print(
                f"\r{source_name}: "
                f"{source_targets:,} rows | "
                f"{source_pairs:,} candidates | "
                f"total rows: {total_targets:,}",
                end="",
                flush=True,
            )

            # Release memory from this chunk.
            del pairs
            del chunk

            if (
                args.max_target_rows is not None
                and total_targets >= args.max_target_rows
            ):
                break

        print(
            f"\nFinished {source_name} in "
            f"{time.time() - source_start:.2f}s"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    elapsed = time.time() - run_start

    print(
        "\n" + "=" * 70
    )

    print(
        "CANDIDATE GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Target rows processed:  {total_targets:,}"
    )

    print(
        f"Candidate pairs:        {total_pairs:,}"
    )

    print(
        f"Runtime:                {elapsed:.2f}s"
    )

    print(
        f"Output:                 {output_path}"
    )

    if output_path.exists():

        size_gb = (
            output_path.stat().st_size
            / (1024 ** 3)
        )

        print(
            f"Output size:            "
            f"{size_gb:.2f} GB"
        )

    else:

        print(
            "WARNING: No candidate pairs "
            "were generated."
        )


if __name__ == "__main__":
    main()
