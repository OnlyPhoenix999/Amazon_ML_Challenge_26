"""
Build the 39-feature entity-resolution representation described by the team.

The normalization functions remain the project's source of truth:
    normalize_business_name()
    normalize_address()

This module only converts a pair of records into numeric features.
It does not alter normalization.py.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz

from src.preprocessing.normalization import (
    normalize_business_name,
    normalize_address,
)


LEGAL_SUFFIXES = {
    "llc", "ltd", "limited", "inc", "incorporated", "corp",
    "corporation", "co", "company", "plc", "lp", "llp",
    "pvt", "private", "gmbh", "sarl", "sas", "sa", "ag",
    "bv", "oy", "ab",
}

NUMBER_RE = re.compile(r"\d+")


FEATURE_COLUMNS = [
    "raw_name_ratio",
    "raw_name_wratio",
    "raw_name_token_sort_ratio",
    "raw_name_token_set_ratio",
    "raw_name_exact",
    "raw_name_length_diff",
    "raw_name_length_ratio",
    "raw_name_token_count_diff",
    "raw_address_ratio",
    "raw_address_wratio",
    "raw_address_token_sort_ratio",
    "raw_address_token_set_ratio",
    "raw_address_exact",
    "raw_address_length_diff",
    "raw_address_length_ratio",
    "raw_address_token_count_diff",
    "norm_name_ratio",
    "norm_name_wratio",
    "norm_name_token_sort_ratio",
    "norm_name_token_set_ratio",
    "norm_name_exact",
    "norm_address_ratio",
    "norm_address_wratio",
    "norm_address_token_sort_ratio",
    "norm_address_token_set_ratio",
    "norm_address_exact",
    "suffixless_name_ratio",
    "suffixless_name_wratio",
    "suffixless_name_exact",
    "first_token_agree",
    "last_token_agree",
    "address_number_match",
    "same_country",
    "source2",
    "source3",
    "address_missing_any",
    "address_both_present",
    "name_script_same",
    "address_script_same",
]


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def tokens(text: str) -> list[str]:
    return [x for x in text.split() if x]


def ratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return float(fuzz.ratio(a, b))


def wratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return float(fuzz.WRatio(a, b))


def token_sort(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return float(fuzz.token_sort_ratio(a, b))


def token_set(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return float(fuzz.token_set_ratio(a, b))


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def suffixless_name(name: str) -> str:
    parts = tokens(name)
    if parts and parts[-1].casefold() in LEGAL_SUFFIXES:
        return " ".join(parts[:-1])
    return name


def first_token(text: str) -> str:
    t = tokens(text)
    return t[0].casefold() if t else ""


def last_token(text: str) -> str:
    t = tokens(text)
    return t[-1].casefold() if t else ""


def first_number(text: str) -> str:
    m = NUMBER_RE.search(text)
    return m.group(0) if m else ""


def script_class(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            key = "latin"
        elif "DEVANAGARI" in name:
            key = "devanagari"
        elif "CYRILLIC" in name:
            key = "cyrillic"
        elif "GREEK" in name:
            key = "greek"
        elif "TELUGU" in name:
            key = "telugu"
        elif "TAMIL" in name:
            key = "tamil"
        elif "BENGALI" in name:
            key = "bengali"
        elif "GUJARATI" in name:
            key = "gujarati"
        elif "GURMUKHI" in name:
            key = "gurmukhi"
        elif "MALAYALAM" in name:
            key = "malayalam"
        else:
            key = "other"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "none"
    return max(counts, key=counts.get)


def pair_features(
    s1_name,
    s1_address,
    s1_country,
    target_name,
    target_address,
    target_country,
    target_source,
) -> dict[str, float]:
    s1_name = safe_text(s1_name)
    s1_address = safe_text(s1_address)
    t_name = safe_text(target_name)
    t_address = safe_text(target_address)

    n1 = normalize_business_name(s1_name)
    n2 = normalize_business_name(t_name)
    a1 = normalize_address(s1_address)
    a2 = normalize_address(t_address)

    n1_suffixless = suffixless_name(n1)
    n2_suffixless = suffixless_name(n2)

    nt1 = tokens(n1)
    nt2 = tokens(n2)
    at1 = tokens(a1)
    at2 = tokens(a2)

    features = {
        "raw_name_ratio": ratio(s1_name, t_name),
        "raw_name_wratio": wratio(s1_name, t_name),
        "raw_name_token_sort_ratio": token_sort(s1_name, t_name),
        "raw_name_token_set_ratio": token_set(s1_name, t_name),
        "raw_name_exact": float(s1_name.casefold() == t_name.casefold()),
        "raw_name_length_diff": float(abs(len(s1_name) - len(t_name))),
        "raw_name_length_ratio": length_ratio(s1_name, t_name),
        "raw_name_token_count_diff": float(abs(len(nt1) - len(nt2))),
        "raw_address_ratio": ratio(s1_address, t_address),
        "raw_address_wratio": wratio(s1_address, t_address),
        "raw_address_token_sort_ratio": token_sort(s1_address, t_address),
        "raw_address_token_set_ratio": token_set(s1_address, t_address),
        "raw_address_exact": float(
            s1_address.casefold() == t_address.casefold()
        ),
        "raw_address_length_diff": float(
            abs(len(s1_address) - len(t_address))
        ),
        "raw_address_length_ratio": length_ratio(s1_address, t_address),
        "raw_address_token_count_diff": float(
            abs(len(at1) - len(at2))
        ),
        "norm_name_ratio": ratio(n1, n2),
        "norm_name_wratio": wratio(n1, n2),
        "norm_name_token_sort_ratio": token_sort(n1, n2),
        "norm_name_token_set_ratio": token_set(n1, n2),
        "norm_name_exact": float(n1 == n2),
        "norm_address_ratio": ratio(a1, a2),
        "norm_address_wratio": wratio(a1, a2),
        "norm_address_token_sort_ratio": token_sort(a1, a2),
        "norm_address_token_set_ratio": token_set(a1, a2),
        "norm_address_exact": float(a1 == a2),
        "suffixless_name_ratio": ratio(n1_suffixless, n2_suffixless),
        "suffixless_name_wratio": wratio(n1_suffixless, n2_suffixless),
        "suffixless_name_exact": float(n1_suffixless == n2_suffixless),
        "first_token_agree": float(first_token(n1) == first_token(n2)),
        "last_token_agree": float(last_token(n1) == last_token(n2)),
        "address_number_match": float(
            bool(first_number(a1))
            and first_number(a1) == first_number(a2)
        ),
        "same_country": float(
            safe_text(s1_country).casefold()
            == safe_text(target_country).casefold()
        ),
        "source2": float(str(target_source).casefold() in {"source2", "s2"}),
        "source3": float(str(target_source).casefold() in {"source3", "s3"}),
        "address_missing_any": float(not s1_address or not t_address),
        "address_both_present": float(bool(s1_address and t_address)),
        "name_script_same": float(script_class(s1_name) == script_class(t_name)),
        "address_script_same": float(
            script_class(s1_address) == script_class(t_address)
        ),
    }

    return {k: float(features[k]) for k in FEATURE_COLUMNS}


def build_feature_table(input_path, output_path, chunksize=25_000):
    import pandas as pd

    required = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "target_source",
        "target_entity_id",
        "target_business_name",
        "target_business_address",
        "target_country",
        "label",
        "matched_block",
    }

    first = pd.read_csv(
        input_path,
        sep="\t",
        dtype="string",
        nrows=0,
    )
    missing = required - set(first.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    first_write = True

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype="string",
        chunksize=chunksize,
        keep_default_na=False,
    ):
        records = []
        for row in chunk.itertuples(index=False):
            r = row._asdict()
            feat = pair_features(
                r["business_name"],
                r["business_address"],
                r["country"],
                r["target_business_name"],
                r["target_business_address"],
                r["target_country"],
                r["target_source"],
            )
            records.append(
                {
                    "entity_id": r["entity_id"],
                    "target_source": r["target_source"],
                    "target_entity_id": r["target_entity_id"],
                    "label": int(r["label"]),
                    "matched_block": r["matched_block"],
                    **feat,
                }
            )

        out = pd.DataFrame(records)
        out.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False

    print(f"Wrote feature table: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunksize", type=int, default=25_000)
    args = parser.parse_args()

    build_feature_table(
        args.input,
        args.output,
        args.chunksize,
    )
