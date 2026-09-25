import os
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "data/raw/dataset"

GT_FILE = os.path.join(
    BASE_DIR, "train", "train_ground_truth.tsv"
)

S1_FILE = os.path.join(
    BASE_DIR, "train", "train_source1.tsv"
)

S2_FILE = os.path.join(
    BASE_DIR, "train", "train_source2.tsv"
)

S3_FILE = os.path.join(
    BASE_DIR, "train", "train_source3.tsv"
)

SAMPLE_S1 = 10_000
CHUNK_SIZE = 100_000
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.ratio(str(a), str(b))


def script_type(text):
    if not text:
        return "empty"

    has_latin = False
    has_devanagari = False
    has_other = False

    for ch in str(text):
        code = ord(ch)

        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            has_latin = True

        elif 0x0900 <= code <= 0x097F:
            has_devanagari = True

        elif ch.isalpha():
            has_other = True

    scripts = sum([
        has_latin,
        has_devanagari,
        has_other
    ])

    if scripts > 1:
        return "mixed"

    if has_devanagari:
        return "devanagari"

    if has_latin:
        return "latin"

    if has_other:
        return "other"

    return "non_alpha"


def retrieve_records(file_path, target_ids):
    records = []

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=CHUNK_SIZE
    ):
        matches = chunk[
            chunk["entity_id"].isin(target_ids)
        ]

        if len(matches):
            records.append(matches)

    if not records:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        )

    return pd.concat(records, ignore_index=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("LARGE-SCALE GROUND TRUTH ANALYSIS")
print("=" * 70)


# ============================================================
# STEP 1 — READ GROUND TRUTH
# ============================================================

print("\nReading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    na_filter=False
)

matched_gt = gt[
    gt["matched_entity_ids"].str.strip() != ""
].copy()

print(f"Total S1 entities:       {len(gt):,}")
print(f"Entities with matches:   {len(matched_gt):,}")


# ============================================================
# STEP 2 — SAMPLE 10,000 S1 ENTITIES
# ============================================================

sample_size = min(
    SAMPLE_S1,
    len(matched_gt)
)

sampled_gt = matched_gt.sample(
    n=sample_size,
    random_state=RANDOM_SEED
)

print(f"Sampled S1 entities:     {len(sampled_gt):,}")


# ============================================================
# STEP 3 — COLLECT IDS
# ============================================================

s1_ids = set(
    sampled_gt["source1_entity_id"]
)

s2_ids = set()
s3_ids = set()

match_count_per_s1 = []

for _, row in sampled_gt.iterrows():

    ids = [
        x.strip()
        for x in row["matched_entity_ids"].split(",")
        if x.strip()
    ]

    match_count_per_s1.append(len(ids))

    for entity_id in ids:

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


print(f"S1 records needed:      {len(s1_ids):,}")
print(f"S2 records needed:      {len(s2_ids):,}")
print(f"S3 records needed:      {len(s3_ids):,}")

print(
    f"Total true pairs:        "
    f"{sum(match_count_per_s1):,}"
)


# ============================================================
# STEP 4 — RETRIEVE RECORDS
# ============================================================

print("\nRetrieving S1...")

s1 = retrieve_records(
    S1_FILE,
    s1_ids
)

print(f"S1 retrieved: {len(s1):,}")


print("\nRetrieving S2...")

s2 = retrieve_records(
    S2_FILE,
    s2_ids
)

print(f"S2 retrieved: {len(s2):,}")


print("\nRetrieving S3...")

s3 = retrieve_records(
    S3_FILE,
    s3_ids
)

print(f"S3 retrieved: {len(s3):,}")


# ============================================================
# STEP 5 — LOOKUPS
# ============================================================

s1_lookup = (
    s1.set_index("entity_id")
    .to_dict("index")
)

s2_lookup = (
    s2.set_index("entity_id")
    .to_dict("index")
)

s3_lookup = (
    s3.set_index("entity_id")
    .to_dict("index")
)


# ============================================================
# STEP 6 — ANALYZE POSITIVE PAIRS
# ============================================================

results = []

for _, row in sampled_gt.iterrows():

    s1_id = row["source1_entity_id"]

    source1 = s1_lookup.get(s1_id)

    if source1 is None:
        continue

    s1_name = source1["business_name"]
    s1_address = source1["business_address"]
    s1_country = source1["country"]

    matched_ids = [
        x.strip()
        for x in row["matched_entity_ids"].split(",")
        if x.strip()
    ]

    for matched_id in matched_ids:

        if matched_id.startswith("S2-"):
            target = s2_lookup.get(matched_id)
            source = "S2"

        else:
            target = s3_lookup.get(matched_id)
            source = "S3"

        if target is None:
            continue

        target_name = target["business_name"]
        target_address = target["business_address"]

        name_sim = similarity(
            s1_name,
            target_name
        )

        address_sim = similarity(
            s1_address,
            target_address
        )

        results.append({

            "s1_id": s1_id,
            "target_id": matched_id,
            "source": source,

            "country": s1_country,

            "name_similarity": name_sim,
            "address_similarity": address_sim,

            "name_script_s1": script_type(s1_name),
            "name_script_target": script_type(target_name),

            "address_missing_s1": not bool(
                s1_address.strip()
            ),

            "address_missing_target": not bool(
                target_address.strip()
            ),

            "s1_name": s1_name,
            "target_name": target_name,

            "s1_address": s1_address,
            "target_address": target_address
        })


df = pd.DataFrame(results)

print(
    f"\nAnalyzed positive pairs: "
    f"{len(df):,}"
)


# ============================================================
# STEP 7 — BASIC STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC SIMILARITY STATISTICS")
print("=" * 70)

print("\nNAME SIMILARITY")

print(
    df["name_similarity"].describe(
        percentiles=[
            .01,
            .05,
            .10,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99
        ]
    )
)

print("\nADDRESS SIMILARITY")

print(
    df["address_similarity"].describe(
        percentiles=[
            .01,
            .05,
            .10,
            .25,
            .50,
            .75,
            .90,
            .95,
            .99
        ]
    )
)


# ============================================================
# STEP 8 — THRESHOLD COUNTS
# ============================================================

print("\n" + "=" * 70)
print("2. SIMILARITY THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    95
]

print("\nThreshold | Name >= T | Address >= T")

for t in thresholds:

    name_count = (
        df["name_similarity"] >= t
    ).sum()

    address_count = (
        df["address_similarity"] >= t
    ).sum()

    print(
        f"{t:9} | "
        f"{name_count:10,} | "
        f"{address_count:13,}"
    )


# ============================================================
# STEP 9 — COMBINED SIGNALS
# ============================================================

print("\n" + "=" * 70)
print("3. COMBINED NAME + ADDRESS SIGNALS")
print("=" * 70)

conditions = [

    ("Name >= 80 AND Address >= 80",
     (df["name_similarity"] >= 80) &
     (df["address_similarity"] >= 80)),

    ("Name < 60 AND Address >= 80",
     (df["name_similarity"] < 60) &
     (df["address_similarity"] >= 80)),

    ("Name >= 80 AND Address < 60",
     (df["name_similarity"] >= 80) &
     (df["address_similarity"] < 60)),

    ("Name < 60 AND Address < 60",
     (df["name_similarity"] < 60) &
     (df["address_similarity"] < 60)),

    ("Name < 40 AND Address >= 70",
     (df["name_similarity"] < 40) &
     (df["address_similarity"] >= 70)),

    ("Name >= 90 AND Address < 40",
     (df["name_similarity"] >= 90) &
     (df["address_similarity"] < 40))
]

for label, condition in conditions:

    count = condition.sum()

    percentage = (
        count / len(df) * 100
        if len(df)
        else 0
    )

    print(
        f"{label:35} "
        f"{count:7,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# STEP 10 — CROSS SCRIPT
# ============================================================

print("\n" + "=" * 70)
print("4. CROSS-SCRIPT ANALYSIS")
print("=" * 70)

cross_script = (
    df["name_script_s1"]
    != df["name_script_target"]
)

print(
    f"Cross-script name pairs: "
    f"{cross_script.sum():,}"
)

print(
    f"Cross-script percentage: "
    f"{cross_script.mean() * 100:.2f}%"
)

print("\nTarget name scripts:")

print(
    df["name_script_target"]
    .value_counts()
)


# ============================================================
# STEP 11 — ADDRESS MISSINGNESS
# ============================================================

print("\n" + "=" * 70)
print("5. ADDRESS MISSINGNESS")
print("=" * 70)

target_address_missing = (
    df["address_missing_target"]
)

print(
    f"Target address missing: "
    f"{target_address_missing.sum():,}"
)

print(
    f"Target address missing %: "
    f"{target_address_missing.mean() * 100:.2f}%"
)


# ============================================================
# STEP 12 — SOURCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("6. SOURCE COMPARISON")
print("=" * 70)

print(
    df.groupby("source")[
        [
            "name_similarity",
            "address_similarity"
        ]
    ].mean()
)


# ============================================================
# STEP 13 — COUNTRY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("7. COUNTRY COMPARISON")
print("=" * 70)

print(
    df.groupby("country")[
        [
            "name_similarity",
            "address_similarity"
        ]
    ].mean()
)


# ============================================================
# STEP 14 — LOW SIMILARITY TRUE MATCHES
# ============================================================

print("\n" + "=" * 70)
print("8. HARDEST TRUE MATCHES")
print("=" * 70)

hardest = df.sort_values(
    by=[
        "name_similarity",
        "address_similarity"
    ]
).head(20)

print(
    hardest[
        [
            "s1_id",
            "target_id",
            "source",
            "country",
            "name_similarity",
            "address_similarity",
            "s1_name",
            "target_name",
            "s1_address",
            "target_address"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 15 — MATCH COUNT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("9. MATCH COUNT PER SOURCE 1")
print("=" * 70)

match_counts = pd.Series(
    match_count_per_s1
)

print(
    match_counts.value_counts()
    .sort_index()
)


# ============================================================
# STEP 16 — SAVE COMPACT RESULTS
# ============================================================

OUTPUT_DIR = (
    "experiments/ground_truth_inspection"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "positive_pairs_10k.tsv"
)

df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nDetailed pairs saved to:\n"
    f"{OUTPUT_FILE}"
)