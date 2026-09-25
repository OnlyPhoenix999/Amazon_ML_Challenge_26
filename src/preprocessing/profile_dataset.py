from pathlib import Path
import pandas as pd
import os
import re


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_DIR = BASE_DIR / "data" / "raw" / "dataset" / "train"

FILES = {
    "source1": TRAIN_DIR / "train_source1.tsv",
    "source2": TRAIN_DIR / "train_source2.tsv",
    "source3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}

CHUNK_SIZE = 100_000


# ============================================================
# HELPERS
# ============================================================

def format_bytes(size):
    """Convert bytes to a readable size."""
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def inspect_file(name, path):
    """Profile one TSV file without loading it entirely into RAM."""

    print("\n" + "=" * 80)
    print(f"FILE: {name}")
    print(f"PATH: {path}")
    print("=" * 80)

    if not path.exists():
        print("ERROR: File not found!")
        return

    print(f"File size: {format_bytes(path.stat().st_size)}")

    total_rows = 0
    missing_counts = None
    dtypes = None
    sample = None

    # Read in chunks
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            on_bad_lines="warn",
        )
    ):

        if chunk_number == 0:
            sample = chunk.head(5).copy()
            dtypes = chunk.dtypes

            missing_counts = {
                col: 0
                for col in chunk.columns
            }

            print("\nCOLUMNS:")
            for col in chunk.columns:
                print(f"  - {col}")

        total_rows += len(chunk)

        # Empty-string counts
        for col in chunk.columns:
            missing_counts[col] += (
                chunk[col].astype(str).str.strip().eq("").sum()
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nROWS:")
    print(f"  {total_rows:,}")

    print("\nCOLUMN COUNT:")
    print(f"  {len(missing_counts)}")

    print("\nMISSING / EMPTY VALUES:")

    for col, count in missing_counts.items():
        percentage = (
            (count / total_rows) * 100
            if total_rows
            else 0
        )

        print(
            f"  {col:<30} "
            f"{count:>12,} "
            f"({percentage:>7.2f}%)"
        )

    print("\nSAMPLE ROWS:")

    if sample is not None:
        print(sample.to_string(index=False))

    return {
        "name": name,
        "rows": total_rows,
        "columns": list(missing_counts.keys()),
        "missing": missing_counts,
    }


def profile_ground_truth(path):
    """Analyze match-count distribution in ground truth."""

    print("\n" + "=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    if not path.exists():
        print("ERROR: Ground truth file not found!")
        return

    total_entities = 0

    zero_matches = 0
    one_match = 0
    multi_matches = 0

    total_matches = 0

    source2_matches = 0
    source3_matches = 0

    max_matches = 0

    match_distribution = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        on_bad_lines="warn",
    ):

        for value in chunk["matched_entity_ids"]:

            total_entities += 1

            value = str(value).strip()

            # Empty = singleton / no match
            if not value:
                match_count = 0

            else:
                ids = [
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                ]

                match_count = len(ids)

                total_matches += match_count

                for entity_id in ids:

                    if entity_id.startswith("S2-"):
                        source2_matches += 1

                    elif entity_id.startswith("S3-"):
                        source3_matches += 1

            # Distribution
            match_distribution[match_count] = (
                match_distribution.get(match_count, 0) + 1
            )

            max_matches = max(
                max_matches,
                match_count
            )

            if match_count == 0:
                zero_matches += 1

            elif match_count == 1:
                one_match += 1

            else:
                multi_matches += 1

    print(f"\nTotal Source 1 entities: {total_entities:,}")

    print(f"Total true matches:       {total_matches:,}")

    print(f"Maximum matches/entity:   {max_matches:,}")

    print("\nMATCH TYPE DISTRIBUTION:")

    if total_entities:

        print(
            f"  Zero matches:  "
            f"{zero_matches:,} "
            f"({zero_matches / total_entities * 100:.2f}%)"
        )

        print(
            f"  One match:     "
            f"{one_match:,} "
            f"({one_match / total_entities * 100:.2f}%)"
        )

        print(
            f"  Multiple:      "
            f"{multi_matches:,} "
            f"({multi_matches / total_entities * 100:.2f}%)"
        )

    print("\nMATCH COUNT DISTRIBUTION:")

    for count in sorted(match_distribution):
        number = match_distribution[count]

        percentage = (
            number / total_entities * 100
            if total_entities
            else 0
        )

        print(
            f"  {count:>3} matches → "
            f"{number:>10,} entities "
            f"({percentage:>7.2f}%)"
        )

    print("\nMATCH SOURCE DISTRIBUTION:")

    print(f"  Source 2 matches: {source2_matches:,}")
    print(f"  Source 3 matches: {source3_matches:,}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("DATASET PROFILER")
    print("=" * 80)

    print(f"\nProject root:")
    print(f"  {BASE_DIR}")

    print(f"\nTraining directory:")
    print(f"  {TRAIN_DIR}")

    print(f"\nChunk size:")
    print(f"  {CHUNK_SIZE:,} rows")

    results = []

    # --------------------------------------------------------
    # Profile Source 1/2/3
    # --------------------------------------------------------

    for name in ["source1", "source2", "source3"]:

        result = inspect_file(
            name,
            FILES[name]
        )

        if result:
            results.append(result)

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    profile_ground_truth(
        FILES["ground_truth"]
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    print(
        f"\n{'Source':<15}"
        f"{'Rows':>15}"
        f"{'Columns':>12}"
    )

    print("-" * 45)

    for result in results:

        print(
            f"{result['name']:<15}"
            f"{result['rows']:>15,}"
            f"{len(result['columns']):>12}"
        )

    print("\nProfiling complete.")
    print("No source files were modified.")


if __name__ == "__main__":
    main()