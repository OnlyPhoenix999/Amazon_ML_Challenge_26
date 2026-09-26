import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rapidfuzz.fuzz import (
    ratio,
    WRatio,
    token_sort_ratio,
    token_set_ratio,
)

from src.preprocessing.normalization import normalize_record


def safe_text(value) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() == "nan":
        return ""

    return text


def text_features(text1: str, text2: str, prefix: str) -> dict:
    text1 = safe_text(text1)
    text2 = safe_text(text2)

    len1 = len(text1)
    len2 = len(text2)

    return {
        f"{prefix}_ratio": ratio(text1, text2),

        f"{prefix}_wratio": WRatio(text1, text2),

        f"{prefix}_token_sort_ratio": token_sort_ratio(
            text1,
            text2,
        ),

        f"{prefix}_token_set_ratio": token_set_ratio(
            text1,
            text2,
        ),

        f"{prefix}_exact": int(
            text1 != "" and text1 == text2
        ),

        f"{prefix}_length_diff": abs(
            len1 - len2
        ),

        f"{prefix}_length_ratio": (
            min(len1, len2) / max(len1, len2)
            if len1 and len2
            else 0.0
        ),

        f"{prefix}_token_count_diff": abs(
            len(text1.split()) -
            len(text2.split())
        ),
    }


def address_number(text: str) -> str:
    text = safe_text(text)

    match = re.search(r"\d+", text)

    return match.group(0) if match else ""


def address_number_match(
    address1: str,
    address2: str,
) -> int:

    num1 = address_number(address1)
    num2 = address_number(address2)

    if not num1 or not num2:
        return 0

    return int(num1 == num2)


def build_pair_features(
    name1: str,
    name2: str,
    address1: str,
    address2: str,
    country1: str = "",
    country2: str = "",
    source: str = "",
) -> dict:

    name1 = safe_text(name1)
    name2 = safe_text(name2)

    address1 = safe_text(address1)
    address2 = safe_text(address2)

    country1 = safe_text(country1).lower()
    country2 = safe_text(country2).lower()

    features = {}

    # =====================================================
    # RAW NAME FEATURES
    # =====================================================

    features.update(
        text_features(
            name1,
            name2,
            "name",
        )
    )

    # =====================================================
    # RAW ADDRESS FEATURES
    # =====================================================

    features.update(
        text_features(
            address1,
            address2,
            "address",
        )
    )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    record1 = normalize_record(
        name1,
        address1,
        country1,
    )

    record2 = normalize_record(
        name2,
        address2,
        country2,
    )

    norm_name1 = record1[
        "business_name_normalized"
    ]

    norm_name2 = record2[
        "business_name_normalized"
    ]

    norm_address1 = record1[
        "business_address_normalized"
    ]

    norm_address2 = record2[
        "business_address_normalized"
    ]

    # =====================================================
    # NORMALIZED NAME FEATURES
    # =====================================================

    features["norm_name_ratio"] = ratio(
        norm_name1,
        norm_name2,
    )

    features["norm_name_wratio"] = WRatio(
        norm_name1,
        norm_name2,
    )

    features["norm_name_token_sort_ratio"] = token_sort_ratio(
        norm_name1,
        norm_name2,
    )

    features["norm_name_token_set_ratio"] = token_set_ratio(
        norm_name1,
        norm_name2,
    )

    features["norm_name_exact"] = int(
        norm_name1 != ""
        and norm_name1 == norm_name2
    )

    # =====================================================
    # LEGAL SUFFIX FEATURES
    # =====================================================

    no_suffix1 = record1[
        "business_name_without_suffix"
    ]

    no_suffix2 = record2[
        "business_name_without_suffix"
    ]

    features["name_no_suffix_ratio"] = ratio(
        no_suffix1,
        no_suffix2,
    )

    features["name_no_suffix_wratio"] = WRatio(
        no_suffix1,
        no_suffix2,
    )

    features["name_no_suffix_exact"] = int(
        no_suffix1 != ""
        and no_suffix1 == no_suffix2
    )

    # =====================================================
    # NAME TOKEN FEATURES
    # =====================================================

    features["name_first_token_same"] = int(
        record1["name_first_token"] != ""
        and record1["name_first_token"]
        == record2["name_first_token"]
    )

    features["name_last_token_same"] = int(
        record1["name_last_token"] != ""
        and record1["name_last_token"]
        == record2["name_last_token"]
    )

    # =====================================================
    # NORMALIZED ADDRESS FEATURES
    # =====================================================

    features["norm_address_ratio"] = ratio(
        norm_address1,
        norm_address2,
    )

    features["norm_address_wratio"] = WRatio(
        norm_address1,
        norm_address2,
    )

    features["norm_address_token_sort_ratio"] = token_sort_ratio(
        norm_address1,
        norm_address2,
    )

    features["norm_address_token_set_ratio"] = token_set_ratio(
        norm_address1,
        norm_address2,
    )

    features["norm_address_exact"] = int(
        norm_address1 != ""
        and norm_address1 == norm_address2
    )

    # =====================================================
    # ADDRESS NUMBER
    # =====================================================

    features["address_number_match"] = address_number_match(
        address1,
        address2,
    )

    # =====================================================
    # COUNTRY
    # =====================================================

    features["same_country"] = int(
        country1 != ""
        and country1 == country2
    )

    # =====================================================
    # SOURCE
    # =====================================================

    features["source_is_s2"] = int(
        source.upper() == "S2"
    )

    features["source_is_s3"] = int(
        source.upper() == "S3"
    )

    # =====================================================
    # MISSING ADDRESS
    # =====================================================

    features["address1_missing"] = int(
        address1 == ""
    )

    features["address2_missing"] = int(
        address2 == ""
    )

    # =====================================================
    # SCRIPT
    # =====================================================

    features["name_script_same"] = int(
        record1["name_script"] != ""
        and record1["name_script"]
        == record2["name_script"]
    )

    features["address_script_same"] = int(
        record1["address_script"] != ""
        and record1["address_script"]
        == record2["address_script"]
    )

    return features


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    import pandas as pd

    s1 = pd.read_csv(
        ROOT / "data/raw/dataset/train/train_source1.tsv",
        sep="\t",
        nrows=1,
    )

    s2 = pd.read_csv(
        ROOT / "data/raw/dataset/train/train_source2.tsv",
        sep="\t",
        nrows=1,
    )

    row1 = s1.iloc[0]
    row2 = s2.iloc[0]

    features = build_pair_features(
        name1=row1["business_name"],
        name2=row2["business_name"],
        address1=row1["business_address"],
        address2=row2["business_address"],
        country1=row1["country"],
        country2=row2["country"],
        source="S2",
    )

    print(
        f"\nGenerated {len(features)} features\n"
    )

    for key, value in features.items():
        print(
            f"{key}: {value}"
        )