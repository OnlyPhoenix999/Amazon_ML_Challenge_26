"""
C-v2: Full ALL_V4 targeted negative mining.

Uses the exact frozen ALL_V4 build_keys() implementation.

Pipeline:
    Full Source1 train entities
        ->
    exact ALL_V4 candidate generation
        ->
    remove GT matches
        ->
    retain deterministic candidate shortlist
        ->
    calculate name/address similarity
        ->
    targeted hard-negative categories

IMPORTANT:
- Does not modify normalization.
- Does not modify ALL_V4.
- Uses TRAIN entities only.
- Final 300-entity holdout is never loaded.
- Ground-truth matches are excluded from negative mining.
"""

from __future__ import annotations

import hashlib
import heapq
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.benchmark_combo2_address_first_v4 import build_keys


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = ROOT / "data" / "raw" / "dataset" / "train"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TRAIN_SPLIT = (
    ROOT
    / "experiments"
    / "splits"
    / "train_entities.tsv"
)

OUT_DIR = ROOT / "experiments" / "targeted_negatives_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FROZEN ALL_V4 CONFIG
# ============================================================

BLOCK_ORDER = (
    "name_first_v2",
    "name_prefix2_v2",
    "address_number_v2",
    "address_first_v4",
)

V4_KEY_MAP = {
    "name_first_v2": "name_first",
    "name_prefix2_v2": "name_prefix2",
    "address_number_v2": "address_number",
    "address_first_v4": "address_first",
}

USECOLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

SEED = 42

# Keep a large but bounded shortlist per Source1.
# Similarity is calculated only after this stage.
CANDIDATES_PER_ENTITY = 1500

# Desired final samples.
TARGET_PER_CATEGORY = 5000


# ============================================================
# HELPERS
# ============================================================

def stable_hash(*parts: str) -> int:
    payload = "|".join(
        [str(SEED), *[str(x) for x in parts]]
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.blake2b(
            payload,
            digest_size=8,
        ).digest(),
        "big",
        signed=False,
    )


def parse_gt(value) -> set[str]:
    if value is None:
        return set()

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return set()

    return {
        x.strip()
        for x in text.split(",")
        if x.strip()
    }


def safe_text(value) -> str:
    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    return text


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return float(ratio(a, b))


# ============================================================
# LOAD TRAIN SOURCE1
# ============================================================

print("=" * 80)
print("C-v2 FULL ALL_V4 TARGETED NEGATIVE MINING")
print("=" * 80)

train_ids = set(
    pd.read_csv(
        TRAIN_SPLIT,
        sep="\t",
        dtype=str,
    )["entity_id"]
)

print(f"Train Source1 entities: {len(train_ids):,}")


# ============================================================
# LOAD SOURCE1 RECORDS
# ============================================================

print("\nLoading train Source1 records...")

source1_records = {}

for chunk in pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype="string",
    chunksize=100_000,
    usecols=USECOLS,
    keep_default_na=False,
):
    for row in chunk.itertuples(index=False, name=None):

        entity_id = str(row[0])

        if entity_id not in train_ids:
            continue

        source1_records[entity_id] = {
            "entity_id": entity_id,
            "business_name": safe_text(row[1]),
            "business_address": safe_text(row[2]),
            "country": safe_text(row[3]),
        }


print(
    f"Loaded Source1 records: "
    f"{len(source1_records):,}"
)


if len(source1_records) != len(train_ids):
    missing = train_ids - set(source1_records)

    raise RuntimeError(
        f"Missing {len(missing)} train Source1 records."
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth for train entities...")

ground_truth = {
    sid: set()
    for sid in train_ids
}

for chunk in pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype="string",
    chunksize=100_000,
    usecols=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
    keep_default_na=False,
):

    subset = chunk[
        chunk["source1_entity_id"].isin(train_ids)
    ]

    for row in subset.itertuples(
        index=False,
        name=None,
    ):
        sid = str(row[0])

        ground_truth[sid].update(
            parse_gt(row[1])
        )


gt_pairs = sum(
    len(v)
    for v in ground_truth.values()
)

print(f"Train GT pairs: {gt_pairs:,}")


# ============================================================
# BUILD EXACT ALL_V4 SOURCE1 INDEXES
# ============================================================

print("\nBuilding exact ALL_V4 indexes...")

indexes = {
    block: defaultdict(list)
    for block in BLOCK_ORDER
}

for sid, record in source1_records.items():

    keys = build_keys(
        record["business_name"],
        record["business_address"],
        record["country"],
        True,
    )

    for block in BLOCK_ORDER:

        actual_key = V4_KEY_MAP[block]

        key = keys.get(actual_key)

        if (
            key
            and isinstance(key, tuple)
            and len(key) >= 2
            and key[0]
            and key[1]
        ):
            indexes[block][
                (
                    str(key[0]),
                    str(key[1]),
                )
            ].append(sid)


print("\nALL_V4 index sizes:")

for block in BLOCK_ORDER:
    print(
        f"  {block:24s}: "
        f"{len(indexes[block]):,}"
    )


# ============================================================
# CANDIDATE SHORTLISTS
# ============================================================

# For every Source1:
# heap entries =
# (block_score, stable_tie, source, target_id,
#  target_name, target_address, target_country,
#  matched_blocks)

candidate_heaps = {
    sid: []
    for sid in train_ids
}


def push_candidate(
    sid,
    item,
):
    heap = candidate_heaps[sid]

    if len(heap) < CANDIDATES_PER_ENTITY:
        heapq.heappush(heap, item)

    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(
            heap,
            item,
        )


# ============================================================
# SCAN SOURCE2 / SOURCE3
# ============================================================

total_rows = 0
total_candidate_occurrences = 0


for source_name, source_path in (
    ("source2", SOURCE2_PATH),
    ("source3", SOURCE3_PATH),
):

    print()
    print("=" * 80)
    print(f"SCANNING {source_name.upper()}")
    print("=" * 80)

    rows_scanned = 0

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype="string",
        chunksize=100_000,
        usecols=USECOLS,
        keep_default_na=False,
    ):

        for row in chunk.itertuples(
            index=False,
            name=None,
        ):

            target_id = str(row[0])
            target_name = safe_text(row[1])
            target_address = safe_text(row[2])
            target_country = safe_text(row[3])

            keys = build_keys(
                target_name,
                target_address,
                target_country,
                True,
            )

            candidate_hits = {}

            for block in BLOCK_ORDER:

                actual_key = V4_KEY_MAP[block]

                key = keys.get(actual_key)

                if not (
                    key
                    and isinstance(key, tuple)
                    and len(key) >= 2
                    and key[0]
                    and key[1]
                ):
                    continue

                lookup_key = (
                    str(key[0]),
                    str(key[1]),
                )

                for sid in indexes[block].get(
                    lookup_key,
                    (),
                ):

                    candidate_hits.setdefault(
                        sid,
                        set(),
                    ).add(block)

            for sid, matched_blocks in candidate_hits.items():

                total_candidate_occurrences += 1

                # Never mine GT positives.
                if target_id in ground_truth[sid]:
                    continue

                block_score = (
                    len(matched_blocks)
                )

                tie = stable_hash(
                    sid,
                    source_name,
                    target_id,
                )

                item = (
                    block_score,
                    tie,
                    source_name,
                    target_id,
                    target_name,
                    target_address,
                    target_country,
                    ",".join(
                        b
                        for b in BLOCK_ORDER
                        if b in matched_blocks
                    ),
                )

                push_candidate(
                    sid,
                    item,
                )

        rows_scanned += len(chunk)
        total_rows += len(chunk)

        print(
            f"\r{source_name}: "
            f"{rows_scanned:,} rows scanned | "
            f"candidate occurrences: "
            f"{total_candidate_occurrences:,}",
            end="",
            flush=True,
        )

    print()


print()
print("=" * 80)
print("ALL_V4 SCAN COMPLETE")
print("=" * 80)

print(
    f"Total target rows scanned: "
    f"{total_rows:,}"
)

print(
    f"Total ALL_V4 candidate occurrences: "
    f"{total_candidate_occurrences:,}"
)


# ============================================================
# CONVERT SHORTLIST TO DATAFRAME
# ============================================================

rows = []

for sid, heap in candidate_heaps.items():

    for (
        block_score,
        tie,
        source_name,
        target_id,
        target_name,
        target_address,
        target_country,
        matched_blocks,
    ) in heap:

        source_record = source1_records[sid]

        rows.append(
            {
                "entity_id": sid,
                "business_name": source_record[
                    "business_name"
                ],
                "business_address": source_record[
                    "business_address"
                ],
                "country": source_record[
                    "country"
                ],
                "target_source": source_name,
                "target_entity_id": target_id,
                "target_business_name": target_name,
                "target_business_address": target_address,
                "target_country": target_country,
                "block_score": block_score,
                "matched_block": matched_blocks,
            }
        )


candidates = pd.DataFrame(rows)

print(
    f"\nCandidate shortlist rows: "
    f"{len(candidates):,}"
)


# ============================================================
# SIMILARITY FEATURES FOR MINING
# ============================================================

print("\nCalculating targeted similarities...")

candidates["name_sim"] = candidates.apply(
    lambda r: similarity(
        r["business_name"],
        r["target_business_name"],
    ),
    axis=1,
)

candidates["address_sim"] = candidates.apply(
    lambda r: similarity(
        r["business_address"],
        r["target_business_address"],
    ),
    axis=1,
)


# ============================================================
# TARGETED CATEGORIES
# ============================================================

categories = {}

categories["similar_name"] = candidates[
    candidates["name_sim"] >= 90
].copy()

categories["similar_address"] = candidates[
    candidates["address_sim"] >= 80
].copy()

categories["name_address"] = candidates[
    (candidates["name_sim"] >= 90)
    & (candidates["address_sim"] >= 80)
].copy()

categories["same_name_diff_address"] = candidates[
    (candidates["name_sim"] >= 95)
    & (candidates["address_sim"] < 70)
].copy()

categories["same_address_diff_name"] = candidates[
    (candidates["address_sim"] >= 90)
    & (candidates["name_sim"] < 70)
].copy()

# High similarity on either side with weak counterpart.
categories["name_strong_address_weak"] = candidates[
    (candidates["name_sim"] >= 90)
    & (candidates["address_sim"] < 50)
].copy()

categories["address_strong_name_weak"] = candidates[
    (candidates["address_sim"] >= 90)
    & (candidates["name_sim"] < 50)
].copy()


# ============================================================
# SAMPLE CATEGORY DATA
# ============================================================

sampled = []

print()
print("=" * 80)
print("TARGETED CATEGORY AVAILABILITY")
print("=" * 80)

for category_name, df in categories.items():

    available = len(df)

    if available == 0:
        print(
            f"{category_name:32s}: 0"
        )
        continue

    n = min(
        TARGET_PER_CATEGORY,
        available,
    )

    subset = df.sample(
        n=n,
        random_state=SEED,
    ).copy()

    subset["negative_category"] = (
        category_name
    )

    sampled.append(subset)

    print(
        f"{category_name:32s}: "
        f"available={available:,} "
        f"sampled={n:,}"
    )


# ============================================================
# RANDOM CONTROL
# ============================================================

random_n = min(
    TARGET_PER_CATEGORY,
    len(candidates),
)

random_control = candidates.sample(
    n=random_n,
    random_state=SEED,
).copy()

random_control[
    "negative_category"
] = "random_control"

sampled.append(
    random_control
)

print(
    f"{'random_control':32s}: "
    f"available={len(candidates):,} "
    f"sampled={random_n:,}"
)


# ============================================================
# COMBINE + DEDUP
# ============================================================

targeted = pd.concat(
    sampled,
    ignore_index=True,
)

print()
print(
    f"Raw targeted rows: "
    f"{len(targeted):,}"
)

targeted = (
    targeted
    .drop_duplicates(
        subset=[
            "entity_id",
            "target_entity_id",
        ]
    )
    .reset_index(drop=True)
)

print(
    f"Unique targeted negative pairs: "
    f"{len(targeted):,}"
)


# ============================================================
# GT LEAKAGE CHECK
# ============================================================

leaks = []

for row in targeted.itertuples():

    sid = str(row.entity_id)
    target_id = str(row.target_entity_id)

    if target_id in ground_truth[sid]:
        leaks.append(
            (sid, target_id)
        )


print(
    f"GT leakage pairs: "
    f"{len(leaks):,}"
)

if leaks:
    raise RuntimeError(
        "GT LEAKAGE DETECTED."
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_count = targeted.duplicated(
    subset=[
        "entity_id",
        "target_entity_id",
    ]
).sum()

print(
    f"Duplicate pair keys: "
    f"{duplicate_count:,}"
)

if duplicate_count:
    raise RuntimeError(
        "Duplicate pair keys remain."
    )


# ============================================================
# DISTRIBUTIONS
# ============================================================

print()
print("=" * 80)
print("CATEGORY DISTRIBUTION")
print("=" * 80)

print(
    targeted[
        "negative_category"
    ].value_counts().to_string()
)

print()
print("=" * 80)
print("SIMILARITY DISTRIBUTION")
print("=" * 80)

print(
    targeted[
        [
            "block_score",
            "name_sim",
            "address_sim",
        ]
    ].describe(
        percentiles=[
            .25,
            .50,
            .75,
            .90,
            .95,
            .99,
        ]
    ).round(2)
)


# ============================================================
# SAVE
# ============================================================

union_path = (
    OUT_DIR
    / "targeted_negatives_union.tsv"
)

targeted.to_csv(
    union_path,
    sep="\t",
    index=False,
)


candidate_path = (
    OUT_DIR
    / "all_v4_train_candidate_shortlist.tsv"
)

candidates.to_csv(
    candidate_path,
    sep="\t",
    index=False,
)


for category_name in (
    targeted[
        "negative_category"
    ].unique()
):

    subset = targeted[
        targeted[
            "negative_category"
        ] == category_name
    ]

    subset.to_csv(
        OUT_DIR
        / f"{category_name}.tsv",
        sep="\t",
        index=False,
    )


print()
print("=" * 80)
print("C-v2 COMPLETE")
print("=" * 80)

print(
    f"Candidate shortlist : "
    f"{candidate_path}"
)

print(
    f"Targeted negatives  : "
    f"{union_path}"
)

print(
    f"Unique negatives    : "
    f"{len(targeted):,}"
)

print(
    f"GT leakage          : "
    f"{len(leaks)}"
)

print(
    f"Duplicate pairs     : "
    f"{duplicate_count}"
)