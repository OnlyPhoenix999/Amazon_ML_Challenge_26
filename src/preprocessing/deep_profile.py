from pathlib import Path
from collections import Counter
import pandas as pd
import unicodedata
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
}

CHUNK_SIZE = 100_000


# ============================================================
# SCRIPT DETECTION
# ============================================================

def detect_script(text):
    """
    Rough Unicode-script classification.

    This is intentionally lightweight.
    We are profiling the data, not doing language detection.
    """

    if not text:
        return "empty"

    has_latin = False
    has_devanagari = False
    has_arabic = False
    has_cyrillic = False
    has_other_alpha = False

    for char in text:

        if not char.isalpha():
            continue

        name = unicodedata.name(char, "")

        if "LATIN" in name:
            has_latin = True

        elif "DEVANAGARI" in name:
            has_devanagari = True

        elif "ARABIC" in name:
            has_arabic = True

        elif "CYRILLIC" in name:
            has_cyrillic = True

        else:
            has_other_alpha = True

    scripts = sum([
        has_latin,
        has_devanagari,
        has_arabic,
        has_cyrillic,
        has_other_alpha
    ])

    if scripts > 1:
        return "mixed"

    if has_latin:
        return "latin"

    if has_devanagari:
        return "devanagari"

    if has_arabic:
        return "arabic"

    if has_cyrillic:
        return "cyrillic"

    if has_other_alpha:
        return "other"

    return "non_alpha"


# ============================================================
# TEXT STATISTICS
# ============================================================

def text_stats(text):

    text = str(text)

    stripped = text.strip()

    char_count = len(stripped)

    words = stripped.split()

    word_count = len(words)

    return char_count, word_count


# ============================================================
# PROFILE ONE SOURCE
# ============================================================

def profile_source(name, path):

    print("\n" + "=" * 80)
    print(f"DEEP PROFILE: {name}")
    print("=" * 80)

    country_counter = Counter()

    name_script_counter = Counter()
    address_script_counter = Counter()

    name_length_bins = Counter()
    address_length_bins = Counter()

    name_word_bins = Counter()
    address_word_bins = Counter()

    name_empty = 0
    address_empty = 0

    total_rows = 0

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

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Country
        # ----------------------------------------------------

        country_counter.update(
            chunk["country"].str.strip()
        )

        # ----------------------------------------------------
        # Process names and addresses
        # ----------------------------------------------------

        for name_value, address_value in zip(
            chunk["business_name"],
            chunk["business_address"]
        ):

            name_value = str(name_value).strip()
            address_value = str(address_value).strip()

            # -----------------------------
            # Empty
            # -----------------------------

            if not name_value:
                name_empty += 1

            if not address_value:
                address_empty += 1

            # -----------------------------
            # Script
            # -----------------------------

            name_script_counter[
                detect_script(name_value)
            ] += 1

            address_script_counter[
                detect_script(address_value)
            ] += 1

            # -----------------------------
            # Length
            # -----------------------------

            name_chars, name_words = text_stats(
                name_value
            )

            address_chars, address_words = text_stats(
                address_value
            )

            # Character bins
            if name_chars == 0:
                name_length_bins["0"] += 1
            elif name_chars <= 10:
                name_length_bins["1-10"] += 1
            elif name_chars <= 25:
                name_length_bins["11-25"] += 1
            elif name_chars <= 50:
                name_length_bins["26-50"] += 1
            elif name_chars <= 100:
                name_length_bins["51-100"] += 1
            else:
                name_length_bins["100+"] += 1

            if address_chars == 0:
                address_length_bins["0"] += 1
            elif address_chars <= 25:
                address_length_bins["1-25"] += 1
            elif address_chars <= 50:
                address_length_bins["26-50"] += 1
            elif address_chars <= 100:
                address_length_bins["51-100"] += 1
            elif address_chars <= 200:
                address_length_bins["101-200"] += 1
            else:
                address_length_bins["200+"] += 1

            # Word bins
            if name_words <= 1:
                name_word_bins["1"] += 1
            elif name_words == 2:
                name_word_bins["2"] += 1
            elif name_words == 3:
                name_word_bins["3"] += 1
            elif name_words <= 5:
                name_word_bins["4-5"] += 1
            else:
                name_word_bins["6+"] += 1

            if address_words <= 2:
                address_word_bins["0-2"] += 1
            elif address_words <= 5:
                address_word_bins["3-5"] += 1
            elif address_words <= 10:
                address_word_bins["6-10"] += 1
            elif address_words <= 20:
                address_word_bins["11-20"] += 1
            else:
                address_word_bins["21+"] += 1

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(f"\nRows: {total_rows:,}")

    print("\nCOUNTRY DISTRIBUTION")
    print("-" * 50)

    for country, count in country_counter.most_common():

        percentage = count / total_rows * 100

        print(
            f"{country:<20}"
            f"{count:>12,}"
            f"  ({percentage:>6.2f}%)"
        )

    print("\nNAME SCRIPT DISTRIBUTION")
    print("-" * 50)

    for script, count in name_script_counter.most_common():

        percentage = count / total_rows * 100

        print(
            f"{script:<20}"
            f"{count:>12,}"
            f"  ({percentage:>6.2f}%)"
        )

    print("\nADDRESS SCRIPT DISTRIBUTION")
    print("-" * 50)

    for script, count in address_script_counter.most_common():

        percentage = count / total_rows * 100

        print(
            f"{script:<20}"
            f"{count:>12,}"
            f"  ({percentage:>6.2f}%)"
        )

    print("\nNAME CHARACTER LENGTH")
    print("-" * 50)

    for bucket, count in name_length_bins.items():

        percentage = count / total_rows * 100

        print(
            f"{bucket:<20}"
            f"{count:>12,}"
            f"  ({percentage:>6.2f}%)"
        )

    print("\nADDRESS CHARACTER LENGTH")
    print("-" * 50)

    for bucket, count in address_length_bins.items():

        percentage = count / total_rows * 100

        print(
            f"{bucket:<20}"
            f"{count:>12,}"
            f"  ({percentage:>6.2f}%)"
        )

    print("\nNAME WORD COUNT")
    print("-" * 50)

    for bucket, count in name_word_bins.items():

        percentage = count / total_rows * 100

        print(
            f"{bucket:<20}"
            f"{count:>12,}"
            f"  ({percentage:>6.2f}%)"
        )

    print("\nADDRESS WORD COUNT")
    print("-" * 50)

    for bucket, count in address_word_bins.items():

        percentage = count / total_rows * 100

        print(
            f"{bucket:<20}"
            f"{count:>12,}"
            f"  ({percentage:>6.2f}%)"
        )

    print("\nEMPTY VALUES")
    print("-" * 50)

    print(
        f"Business name: "
        f"{name_empty:,} "
        f"({name_empty / total_rows * 100:.2f}%)"
    )

    print(
        f"Business address: "
        f"{address_empty:,} "
        f"({address_empty / total_rows * 100:.2f}%)"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("DEEP DATA CHARACTERIZATION")
    print("=" * 80)

    for name, path in FILES.items():

        profile_source(
            name,
            path
        )

    print("\n" + "=" * 80)
    print("DEEP PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()