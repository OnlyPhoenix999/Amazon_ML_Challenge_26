from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz.fuzz import ratio

ROOT = Path(__file__).resolve().parents[2]

PAIR_PATH = ROOT / "experiments" / "training_pairs" / "training_pairs_v4.tsv"
TRAIN_SPLIT = ROOT / "experiments" / "splits" / "train_entities.tsv"
GT_PATH = ROOT / "data" / "raw" / "dataset" / "train" / "train_ground_truth.tsv"

OUT_DIR = ROOT / "experiments" / "targeted_negatives_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------
# 1. Load train entities
# ---------------------------------------------------------

train_entities = set(
    pd.read_csv(TRAIN_SPLIT, sep="\t", dtype=str)["entity_id"]
)

print(f"Train entities: {len(train_entities):,}")


# ---------------------------------------------------------
# 2. Load ground truth
# ---------------------------------------------------------

gt = {}

for chunk in pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    chunksize=100000,
):
    for _, row in chunk.iterrows():
        s1 = str(row["source1_entity_id"])

        raw = (
            str(row["matched_entity_ids"])
            if pd.notna(row["matched_entity_ids"])
            else ""
        )

        matched = {
            x.strip()
            for x in raw.split(",")
            if x.strip()
        }

        gt[s1] = matched

print(f"Ground-truth Source1 entities: {len(gt):,}")


# ---------------------------------------------------------
# 3. Load existing ALL_V4 pair pool
# ---------------------------------------------------------

print("Loading training pair pool...")

pairs = pd.read_csv(
    PAIR_PATH,
    sep="\t",
    dtype=str,
)

print(f"Total pair rows: {len(pairs):,}")


# ---------------------------------------------------------
# 4. Restrict to TRAIN entities only
# ---------------------------------------------------------

pairs = pairs[
    pairs["entity_id"].isin(train_entities)
].copy()

print(f"Train-entity pair rows: {len(pairs):,}")


# ---------------------------------------------------------
# 5. Keep only true negatives
# ---------------------------------------------------------

def is_true_negative(row):
    s1 = str(row["entity_id"])
    candidate = str(row["target_entity_id"])

    return candidate not in gt.get(s1, set())


pairs = pairs[pairs.apply(is_true_negative, axis=1)].copy()

print(f"True-negative pool: {len(pairs):,}")


# ---------------------------------------------------------
# 6. Similarity calculations
# ---------------------------------------------------------

print("Calculating targeted similarity signals...")


def safe_ratio(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0

    a = str(a)
    b = str(b)

    if not a or not b:
        return 0.0

    return float(ratio(a, b))


pairs["name_sim"] = pairs.apply(
    lambda r: safe_ratio(
        r["business_name"],
        r["target_business_name"],
    ),
    axis=1,
)

pairs["address_sim"] = pairs.apply(
    lambda r: safe_ratio(
        r["business_address"],
        r["target_business_address"],
    ),
    axis=1,
)


# ---------------------------------------------------------
# 7. Targeted categories
# ---------------------------------------------------------

# A: Extremely similar names but not matches
similar_name = pairs[
    pairs["name_sim"] >= 90
].copy()

# B: Extremely similar addresses but not matches
similar_address = pairs[
    pairs["address_sim"] >= 80
].copy()

# C: Both name and address look convincing
name_address = pairs[
    (pairs["name_sim"] >= 90)
    & (pairs["address_sim"] >= 80)
].copy()

# D: Same/near-same name but address is meaningfully different
same_name_diff_address = pairs[
    (pairs["name_sim"] >= 95)
    & (pairs["address_sim"] < 70)
].copy()

# E: Same/near-same address but name is meaningfully different
same_address_diff_name = pairs[
    (pairs["address_sim"] >= 90)
    & (pairs["name_sim"] < 70)
].copy()


# ---------------------------------------------------------
# 8. Sample categories
# ---------------------------------------------------------

rng = np.random.default_rng(RANDOM_STATE)

TARGET_PER_CATEGORY = 5000


def sample_category(df, name):
    if len(df) == 0:
        print(f"{name:25s}: 0")
        return df.copy()

    n = min(TARGET_PER_CATEGORY, len(df))

    sampled = df.sample(
        n=n,
        random_state=RANDOM_STATE,
    ).copy()

    sampled["negative_category"] = name

    print(
        f"{name:25s}: "
        f"available={len(df):,} "
        f"sampled={len(sampled):,}"
    )

    return sampled


categories = [
    sample_category(similar_name, "similar_name"),
    sample_category(similar_address, "similar_address"),
    sample_category(name_address, "name_address"),
    sample_category(
        same_name_diff_address,
        "same_name_diff_address",
    ),
    sample_category(
        same_address_diff_name,
        "same_address_diff_name",
    ),
]


# ---------------------------------------------------------
# 9. Random control
# ---------------------------------------------------------

random_pool = pairs.copy()

random_n = min(
    TARGET_PER_CATEGORY,
    len(random_pool),
)

random_control = random_pool.sample(
    n=random_n,
    random_state=RANDOM_STATE,
).copy()

random_control["negative_category"] = "random_control"

print(
    f"{'random_control':25s}: "
    f"available={len(random_pool):,} "
    f"sampled={len(random_control):,}"
)

categories.append(random_control)


# ---------------------------------------------------------
# 10. Combine
# ---------------------------------------------------------

targeted = pd.concat(
    categories,
    ignore_index=True,
)

print()
print(f"Raw targeted rows: {len(targeted):,}")


# ---------------------------------------------------------
# 11. Deduplicate pair keys
# ---------------------------------------------------------

PAIR_KEYS = [
    "entity_id",
    "target_entity_id",
]

targeted = (
    targeted
    .drop_duplicates(subset=PAIR_KEYS)
    .reset_index(drop=True)
)

print(
    f"Unique targeted negative pairs: "
    f"{len(targeted):,}"
)


# ---------------------------------------------------------
# 12. Safety: GT leakage check
# ---------------------------------------------------------

leak_count = 0

for _, row in targeted.iterrows():
    s1 = str(row["entity_id"])
    candidate = str(row["target_entity_id"])

    if candidate in gt.get(s1, set()):
        leak_count += 1

print(f"GT leakage pairs: {leak_count:,}")

if leak_count != 0:
    raise RuntimeError(
        "GT LEAKAGE DETECTED — aborting."
    )


# ---------------------------------------------------------
# 13. Duplicate check
# ---------------------------------------------------------

duplicate_pairs = targeted.duplicated(
    subset=PAIR_KEYS
).sum()

print(
    f"Duplicate pair keys after dedup: "
    f"{duplicate_pairs:,}"
)

if duplicate_pairs != 0:
    raise RuntimeError(
        "Duplicate pair keys remain."
    )


# ---------------------------------------------------------
# 14. Category distribution
# ---------------------------------------------------------

print()
print("CATEGORY DISTRIBUTION")
print("=" * 60)

print(
    targeted["negative_category"]
    .value_counts()
    .to_string()
)


# ---------------------------------------------------------
# 15. Similarity distributions
# ---------------------------------------------------------

print()
print("SIMILARITY SUMMARY")
print("=" * 60)

print(
    targeted[
        ["name_sim", "address_sim"]
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    ).round(2)
)


# ---------------------------------------------------------
# 16. Save individual category files
# ---------------------------------------------------------

for category in targeted["negative_category"].unique():

    subset = targeted[
        targeted["negative_category"] == category
    ].copy()

    path = OUT_DIR / f"{category}.tsv"

    subset.to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(f"Wrote {path}")


# ---------------------------------------------------------
# 17. Save union
# ---------------------------------------------------------

union_path = (
    OUT_DIR /
    "targeted_negatives_union.tsv"
)

targeted.to_csv(
    union_path,
    sep="\t",
    index=False,
)

print()
print("=" * 60)
print("TARGETED NEGATIVE MINING COMPLETE")
print("=" * 60)

print(f"Output: {union_path}")
print(f"Rows:   {len(targeted):,}")
print(f"Leakage: {leak_count}")
print(f"Duplicates: {duplicate_pairs}")