from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ============================================================
# LEGAL ENTITY SUFFIXES
# ============================================================
#
# IMPORTANT:
# We DO NOT remove these from the normalized name.
# They can be useful discriminative features later.
#
# We only detect them separately.
# ============================================================

LEGAL_SUFFIX_PATTERNS = [
    # Multi-word first
    r"private limited",
    r"pvt limited",
    r"pvt ltd",
    r"private ltd",
    r"public limited",
    r"public ltd",
    r"limited liability company",
    r"limited liability partnership",

    # Common single-word suffixes
    r"limited",
    r"ltd",
    r"llc",
    r"llp",
    r"incorporated",
    r"inc",
    r"corporation",
    r"corp",
    r"company",
    r"co",
    r"plc",
    r"pc",
    r"p c",
    r"pllc",
]


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text: object) -> str:
    """
    Unicode normalization.

    NFKC helps normalize visually equivalent Unicode forms
    without transliterating non-Latin scripts.
    """

    if text is None:
        return ""

    text = str(text)

    if not text:
        return ""

    return unicodedata.normalize(
        "NFKC",
        text
    )


def normalize_case(text: object) -> str:
    """
    Unicode-aware lowercase.
    """

    text = normalize_unicode(text)

    return text.casefold()


def normalize_whitespace(text: object) -> str:
    """
    Collapse repeated whitespace.
    """

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# BUSINESS NAME
# ============================================================

def normalize_business_name(text: object) -> str:
    """
    Normalize a business name while preserving meaningful
    Unicode characters and tokens.

    Example:

        "ABC  Healthcare, Pvt. Ltd."
        ->
        "abc healthcare pvt ltd"
    """

    text = normalize_case(text)

    if not text:
        return ""

    # Convert punctuation/symbol separators to spaces.
    #
    # Keep Unicode letters and numbers.
    text = "".join(
        char if (
            char.isalnum()
            or char.isspace()
            or unicodedata.category(char).startswith("M")
        )
        else " "
        for char in text
    )

    text = normalize_whitespace(text)

    return text


# ============================================================
# ADDRESS
# ============================================================

def normalize_address(text: object) -> str:
    """
    Normalize an address.

    We deliberately preserve:
        - numbers
        - Unicode letters
        - token ordering
        - meaningful address components

    Example:

        "88 Olive Circle, Lebanon, TN"
        ->
        "88 olive circle lebanon tn"
    """

    text = normalize_case(text)

    if not text:
        return ""

    text = "".join(
        char if (
            char.isalnum()
            or char.isspace()
            or unicodedata.category(char).startswith("M")
        )
        else " "
        for char in text
    )

    text = normalize_whitespace(text)

    return text


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: object) -> list[str]:
    """
    Unicode-aware whitespace tokenization.
    """

    text = normalize_whitespace(
        normalize_unicode(text)
    )

    if not text:
        return []

    return text.split()


# ============================================================
# LEGAL SUFFIX EXTRACTION
# ============================================================

def extract_legal_suffix(
    business_name: object
) -> str:
    """
    Extract a legal suffix from the END of a normalized
    business name.

    Examples:

        "abc healthcare pvt ltd"
            -> "pvt ltd"

        "abc healthcare llc"
            -> "llc"

        "abc healthcare"
            -> ""

    The suffix is retained separately because removing it
    completely could destroy useful information.
    """

    name = normalize_business_name(
        business_name
    )

    if not name:
        return ""

    # Longest patterns first.
    patterns = sorted(
        LEGAL_SUFFIX_PATTERNS,
        key=len,
        reverse=True
    )

    for pattern in patterns:

        if re.search(
            rf"(?:^|\s){pattern}$",
            name
        ):
            match = re.search(
                rf"(?:^|\s)({pattern})$",
                name
            )

            if match:
                return match.group(1)

    return ""


# ============================================================
# NAME WITHOUT LEGAL SUFFIX
# ============================================================

def remove_legal_suffix(
    business_name: object
) -> str:
    """
    Remove ONLY a recognized trailing legal suffix.

    This representation is useful as an additional feature,
    NOT as the only representation.
    """

    name = normalize_business_name(
        business_name
    )

    if not name:
        return ""

    patterns = sorted(
        LEGAL_SUFFIX_PATTERNS,
        key=len,
        reverse=True
    )

    for pattern in patterns:

        name = re.sub(
            rf"\s+{pattern}$",
            "",
            name
        )

    return normalize_whitespace(name)


# ============================================================
# CHARACTER / TOKEN FEATURES
# ============================================================

def text_length(text: object) -> int:
    return len(
        normalize_unicode(text)
    )


def token_count(text: object) -> int:
    return len(
        tokenize(text)
    )


def first_token(text: object) -> str:
    tokens = tokenize(text)

    if not tokens:
        return ""

    return tokens[0]


def last_token(text: object) -> str:
    tokens = tokenize(text)

    if not tokens:
        return ""

    return tokens[-1]


# ============================================================
# SCRIPT DETECTION
# ============================================================

def detect_script(text: object) -> str:
    """
    Lightweight script detection.

    This is intentionally not language detection.

    Possible outputs:
        latin
        devanagari
        mixed
        other
        empty
    """

    text = normalize_unicode(text)

    if not text:
        return "empty"

    has_latin = False
    has_devanagari = False
    has_other_alpha = False

    for char in text:

        if not char.isalpha():
            continue

        name = unicodedata.name(
            char,
            ""
        )

        if "LATIN" in name:
            has_latin = True

        elif "DEVANAGARI" in name:
            has_devanagari = True

        else:
            has_other_alpha = True

    scripts = sum([
        has_latin,
        has_devanagari,
        has_other_alpha
    ])

    if scripts > 1:
        return "mixed"

    if has_latin:
        return "latin"

    if has_devanagari:
        return "devanagari"

    if has_other_alpha:
        return "other"

    return "empty"


# ============================================================
# NORMALIZED RECORD
# ============================================================

def normalize_record(
    business_name: object,
    business_address: object,
    country: object
) -> dict:
    """
    Produce all reusable normalized representations for
    one business record.
    """

    name_norm = normalize_business_name(
        business_name
    )

    address_norm = normalize_address(
        business_address
    )

    suffix = extract_legal_suffix(
        name_norm
    )

    name_without_suffix = remove_legal_suffix(
        name_norm
    )

    return {
        # Original semantic fields
        "business_name": (
            "" if business_name is None
            else str(business_name)
        ),

        "business_address": (
            "" if business_address is None
            else str(business_address)
        ),

        "country": (
            "" if country is None
            else str(country)
        ),

        # Normalized fields
        "business_name_normalized":
            name_norm,

        "business_address_normalized":
            address_norm,

        "business_name_without_suffix":
            name_without_suffix,

        "legal_suffix":
            suffix,

        # Structural features
        "name_token_count":
            token_count(name_norm),

        "address_token_count":
            token_count(address_norm),

        "name_length":
            len(name_norm),

        "address_length":
            len(address_norm),

        "name_first_token":
            first_token(name_norm),

        "name_last_token":
            last_token(name_norm),

        # Script features
        "name_script":
            detect_script(name_norm),

        "address_script":
            detect_script(address_norm),
    }


# ============================================================
# QUICK SELF TEST
# ============================================================

if __name__ == "__main__":

    examples = [

        (
            "ABC Healthcare, Pvt. Ltd.",
            "88 Olive Circle, Lebanon, TN",
            "US"
        ),

        (
            "Pediatric  Dental Vanguard Care-Associates LLC",
            "123 Main St., Mumbai, Maharashtra",
            "India"
        ),

        (
            "महाकाल ट्रेडर्स Pvt. Ltd.",
            "F/403, Sahaj Residency, Mumbai",
            "India"
        ),

        (
            "Middletown Pédiatric Dentistry Center Ltd",
            "123 Rue de Paris, Lyon",
            "France"
        ),
    ]

    print("=" * 70)
    print("NORMALIZATION SELF TEST")
    print("=" * 70)

    for name, address, country in examples:

        result = normalize_record(
            name,
            address,
            country
        )

        print("\nRAW")
        print("Name    :", name)
        print("Address :", address)

        print("\nNORMALIZED")
        print(
            "Name    :",
            result["business_name_normalized"]
        )

        print(
            "Address :",
            result["business_address_normalized"]
        )

        print(
            "No suffix:",
            result["business_name_without_suffix"]
        )

        print(
            "Suffix  :",
            result["legal_suffix"]
        )

        print(
            "Name script:",
            result["name_script"]
        )

        print(
            "Address script:",
            result["address_script"]
        )

        print("-" * 70)