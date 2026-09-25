import os
import random
import pandas as pd
from rapidfuzz import fuzz

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "data/raw/dataset"

GT_FILE = os.path.join(BASE_DIR, "train", "train_ground_truth.tsv")
S1_FILE = os.path.join(BASE_DIR, "train", "train_source1.tsv")
S2_FILE = os.path.join(BASE_DIR, "train", "train_source2.tsv")
S3_FILE = os.path.join(BASE_DIR, "train", "train_source3.tsv")

OUTPUT_DIR = "experiments/ground_truth_inspection"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "ground_truth_pairs_sample.tsv"
)

SAMPLE_SIZE = 50
CHUNK_SIZE = 100_000
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def similarity(a, b):
    a = clean_text(a)
    b = clean_text(b)

    if not a or not b:
        return 0

    return fuzz.ratio(a, b)


# ============================================================
# STEP 1 — SAMPLE SOURCE 1 ENTITIES FROM GROUND TRUTH
# ============================================================

print("=" * 70)
print("GROUND TRUTH INSPECTION")
print("=" * 70)

random.seed(RANDOM_SEED)

print("\nReading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    na_filter=False
)

print(f"Ground truth rows: {len(gt):,}")


# Separate entities with matches and without matches
matched_gt = gt[
    gt["matched_entity_ids"].str.strip() != ""
].copy()

unmatched_gt = gt[
    gt["matched_entity_ids"].str.strip() == ""
].copy()

print(f"Entities with matches:    {len(matched_gt):,}")
print(f"Entities without matches: {len(unmatched_gt):,}")


# Sample matched Source 1 entities
sample_size = min(SAMPLE_SIZE, len(matched_gt))

sampled_gt = matched_gt.sample(
    n=sample_size,
    random_state=RANDOM_SEED
)

print(f"\nSelected {len(sampled_gt)} matched Source 1 entities.")


# ============================================================
# STEP 2 — COLLECT REQUIRED ENTITY IDS
# ============================================================

s1_ids = set(sampled_gt["source1_entity_id"])

target_s2_ids = set()
target_s3_ids = set()

for value in sampled_gt["matched_entity_ids"]:

    ids = [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]

    for entity_id in ids:

        if entity_id.startswith("S2-"):
            target_s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            target_s3_ids.add(entity_id)


print(f"S1 entities needed: {len(s1_ids):,}")
print(f"S2 entities needed: {len(target_s2_ids):,}")
print(f"S3 entities needed: {len(target_s3_ids):,}")


# ============================================================
# STEP 3 — RETRIEVE ONLY REQUIRED S1 RECORDS
# ============================================================

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

        if len(matches) > 0:
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


print("\nRetrieving Source 1 records...")

s1_records = retrieve_records(
    S1_FILE,
    s1_ids
)

print(f"Retrieved S1 records: {len(s1_records):,}")


# ============================================================
# STEP 4 — RETRIEVE REQUIRED S2 RECORDS
# ============================================================

print("\nRetrieving Source 2 records...")

s2_records = retrieve_records(
    S2_FILE,
    target_s2_ids
)

print(f"Retrieved S2 records: {len(s2_records):,}")


# ============================================================
# STEP 5 — RETRIEVE REQUIRED S3 RECORDS
# ============================================================

print("\nRetrieving Source 3 records...")

s3_records = retrieve_records(
    S3_FILE,
    target_s3_ids
)

print(f"Retrieved S3 records: {len(s3_records):,}")


# ============================================================
# STEP 6 — CREATE LOOKUP DICTIONARIES
# ============================================================

s1_lookup = s1_records.set_index("entity_id").to_dict("index")
s2_lookup = s2_records.set_index("entity_id").to_dict("index")
s3_lookup = s3_records.set_index("entity_id").to_dict("index")


# ============================================================
# STEP 7 — BUILD INSPECTION TABLE
# ============================================================

rows = []

for _, gt_row in sampled_gt.iterrows():

    s1_id = gt_row["source1_entity_id"]

    s1 = s1_lookup.get(s1_id)

    if s1 is None:
        continue

    matched_ids = [
        x.strip()
        for x in gt_row["matched_entity_ids"].split(",")
        if x.strip()
    ]

    for matched_id in matched_ids:

        if matched_id.startswith("S2-"):
            target = s2_lookup.get(matched_id)
        else:
            target = s3_lookup.get(matched_id)

        if target is None:
            continue

        name_sim = similarity(
            s1["business_name"],
            target["business_name"]
        )

        address_sim = similarity(
            s1["business_address"],
            target["business_address"]
        )

        rows.append({

            "source1_entity_id": s1_id,

            "matched_entity_id": matched_id,

            "source": matched_id[:2],

            "country_s1": s1["country"],
            "country_target": target["country"],

            "s1_name": s1["business_name"],
            "target_name": target["business_name"],

            "name_similarity": round(name_sim, 2),

            "s1_address": s1["business_address"],
            "target_address": target["business_address"],

            "address_similarity": round(address_sim, 2),
        })


# ============================================================
# STEP 8 — SAVE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(f"\nGenerated pairs: {len(result):,}")

print(f"\nOutput:")
print(OUTPUT_FILE)

print("\nPreview:")
print(result.head(10).to_string(index=False))