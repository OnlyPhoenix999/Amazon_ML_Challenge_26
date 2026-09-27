"""
C-v3: Targeted ALL_V4 hard-negative mining.

Strategy:
    1. Use the exact frozen ALL_V4 build_keys().
    2. Scan Source2 + Source3 in streaming mode.
    3. Restrict to the 1,400 TRAIN Source1 entities.
    4. Remove ground-truth matches.
    5. Keep only candidates hitting >=2 distinct ALL_V4 blocks.
    6. Calculate fuzzy similarities only for that targeted pool.
    7. Build hard-negative categories.
    8. Keep a small random 1-block control sample for diagnostics.

This does NOT modify:
    - normalization
    - ALL_V4
    - entity split
    - holdout
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
    ROOT / "experiments" / "splits" / "train_entities.tsv"
)

OUT_DIR = ROOT / "experiments" / "targeted_negatives_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FROZEN ALL_V4
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

# Minimum number of independent ALL_V4 blocks required.
MIN_BLOCKS = 2

# Number of strong candidates retained per Source1.
STRONG_PER_ENTITY = 300

# Small diagnostic sample of one-block negatives.
ONE_BLOCK_CONTROL_PER_ENTITY = 5

TARGET_PER_CATEGORY = 5000


# ============================================================
# HELPERS
# ============================================================

def safe_text(value) -> str:
    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    return text


def parse_gt(value) -> set[str]:
    text = safe_text(value).strip()

    if not text:
        return set()

    return {
        x.strip()
        for x in text.split(",")
        if x.strip()
    }


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


def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return float(ratio(a, b))


# ============================================================
# LOAD TRAIN ENTITY IDS
# ============================================================

print("=" * 80)
print("C-v3 TARGETED ALL_V4 NEGATIVE MINING")
print("=" * 80)

train_ids = set(
    pd.read_csv(
        TRAIN_SPLIT,
        sep="\t",
        dtype=str,
    )["entity_id"]
)

print(
    f"Train Source1 entities: "
    f"{len(train_ids):,}"
)


# ============================================================
# LOAD SOURCE1
# ============================================================

source1 = {}

for chunk in pd.read_csv(
    SOURCE1_PATH,
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

        sid = str(row[0])

        if sid not in train_ids:
            continue

        source1[sid] = {
            "entity_id": sid,
            "business_name": safe_text(row[1]),
            "business_address": safe_text(row[2]),
            "country": safe_text(row[3]),
        }

print(
    f"Loaded Source1 records: "
    f"{len(source1):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading train ground truth...")

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

        ground_truth[
            str(row[0])
        ].update(
            parse_gt(row[1])
        )


print(
    "Train GT pairs:",
    sum(len(v) for v in ground_truth.values()),
)


# ============================================================
# BUILD EXACT ALL_V4 INDEX
# ============================================================

print("\nBuilding exact ALL_V4 indexes...")

indexes = {
    block: defaultdict(list)
    for block in BLOCK_ORDER
}

for sid, record in source1.items():

    keys = build_keys(
        record["business_name"],
        record["business_address"],
        record["country"],
        True,
    )

    for block in BLOCK_ORDER:

        key = keys.get(
            V4_KEY_MAP[block]
        )

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


for block in BLOCK_ORDER:
    print(
        f"{block:24s}: "
        f"{len(indexes[block]):,}"
    )


# ============================================================
# CANDIDATE HEAPS
# ============================================================

# Strong candidates:
# retain candidates with >=2 block hits.
#
# Heap item:
# (block_count, hash, source, target_id,
#  name, address, country, blocks)

strong_heaps = {
    sid: []
    for sid in train_ids
}

# One-block diagnostic control.
one_block_heaps = {
    sid: []
    for sid in train_ids
}


def push_heap(
    heap,
    limit,
    item,
):
    if len(heap) < limit:
        heapq.heappush(heap, item)

    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(
            heap,
            item,
        )


# ============================================================
# STREAM SOURCE2 + SOURCE3
# ============================================================

total_rows = 0
total_candidate_occurrences = 0
strong_occurrences = 0
one_block_occurrences = 0
gt_skipped = 0


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

                key = keys.get(
                    V4_KEY_MAP[block]
                )

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

            for sid, blocks in candidate_hits.items():

                total_candidate_occurrences += 1

                # Never use a GT match as a negative.
                if target_id in ground_truth[sid]:
                    gt_skipped += 1
                    continue

                block_count = len(blocks)

                tie = stable_hash(
                    sid,
                    source_name,
                    target_id,
                )

                item = (
                    block_count,
                    tie,
                    source_name,
                    target_id,
                    target_name,
                    target_address,
                    target_country,
                    ",".join(
                        b
                        for b in BLOCK_ORDER
                        if b in blocks
                    ),
                )

                if block_count >= MIN_BLOCKS:

                    strong_occurrences += 1

                    push_heap(
                        strong_heaps[sid],
                        STRONG_PER_ENTITY,
                        item,
                    )

                else:

                    one_block_occurrences += 1

                    push_heap(
                        one_block_heaps[sid],
                        ONE_BLOCK_CONTROL_PER_ENTITY,
                        item,
                    )

        rows_scanned += len(chunk)
        total_rows += len(chunk)

        print(
            f"\r{source_name}: "
            f"{rows_scanned:,} rows | "
            f"ALL_V4 candidates="
            f"{total_candidate_occurrences:,} | "
            f"multi-block="
            f"{strong_occurrences:,}",
            end="",
            flush=True,
        )

    print()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("ALL_V4 TARGETED SCAN COMPLETE")
print("=" * 80)

print(
    f"Rows scanned              : "
    f"{total_rows:,}"
)

print(
    f"ALL_V4 candidate hits     : "
    f"{total_candidate_occurrences:,}"
)

print(
    f"GT positives skipped      : "
    f"{gt_skipped:,}"
)

print(
    f"Multi-block negatives     : "
    f"{strong_occurrences:,}"
)

print(
    f"One-block control hits    : "
    f"{one_block_occurrences:,}"
)


# ============================================================
# BUILD STRONG CANDIDATE DATAFRAME
# ============================================================

strong_rows = []

for sid, heap in strong_heaps.items():

    for (
        block_count,
        tie,
        source_name,
        target_id,
        target_name,
        target_address,
        target_country,
        matched_blocks,
    ) in heap:

        strong_rows.append(
            {
                **source1[sid],
                "target_source": source_name,
                "target_entity_id": target_id,
                "target_business_name": target_name,
                "target_business_address": target_address,
                "target_country": target_country,
                "block_count": block_count,
                "matched_block": matched_blocks,
            }
        )


strong = pd.DataFrame(strong_rows)

print(
    f"\nStrong candidate shortlist: "
    f"{len(strong):,}"
)


# ============================================================
# FUZZY SIMILARITY
# ============================================================

if len(strong):

    print(
        "Calculating similarities "
        "only on multi-block candidates..."
    )

    strong["name_sim"] = strong.apply(
        lambda r: sim(
            r["business_name"],
            r["target_business_name"],
        ),
        axis=1,
    )

    strong["address_sim"] = strong.apply(
        lambda r: sim(
            r["business_address"],
            r["target_business_address"],
        ),
        axis=1,
    )


# ============================================================
# TARGETED CATEGORIES
# ============================================================

categories = {}

if len(strong):

    categories[
        "similar_name"
    ] = strong[
        strong["name_sim"] >= 90
    ].copy()

    categories[
        "similar_address"
    ] = strong[
        strong["address_sim"] >= 80
    ].copy()

    categories[
        "name_address"
    ] = strong[
        (strong["name_sim"] >= 90)
        & (strong["address_sim"] >= 80)
    ].copy()

    categories[
        "same_name_diff_address"
    ] = strong[
        (strong["name_sim"] >= 95)
        & (strong["address_sim"] < 70)
    ].copy()

    categories[
        "same_address_diff_name"
    ] = strong[
        (strong["address_sim"] >= 90)
        & (strong["name_sim"] < 70)
    ].copy()

    categories[
        "name_strong_address_weak"
    ] = strong[
        (strong["name_sim"] >= 90)
        & (strong["address_sim"] < 50)
    ].copy()

    categories[
        "address_strong_name_weak"
    ] = strong[
        (strong["address_sim"] >= 90)
        & (strong["name_sim"] < 50)
    ].copy()


# ============================================================
# SAMPLE
# ============================================================

sampled = []

print()
print("=" * 80)
print("TARGETED CATEGORY AVAILABILITY")
print("=" * 80)

for category, df in categories.items():

    available = len(df)

    if available == 0:
        print(
            f"{category:32s}: 0"
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

    subset[
        "negative_category"
    ] = category

    sampled.append(subset)

    print(
        f"{category:32s}: "
        f"available={available:,} "
        f"sampled={n:,}"
    )


# ============================================================
# RANDOM CONTROL
# ============================================================

if len(strong):

    n = min(
        TARGET_PER_CATEGORY,
        len(strong),
    )

    random_control = strong.sample(
        n=n,
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
        f"available={len(strong):,} "
        f"sampled={n:,}"
    )


# ============================================================
# COMBINE
# ============================================================

if sampled:

    targeted = pd.concat(
        sampled,
        ignore_index=True,
    )

else:

    targeted = pd.DataFrame()


print(
    f"\nRaw targeted rows: "
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
    f"Unique targeted pairs: "
    f"{len(targeted):,}"
)


# ============================================================
# GT LEAKAGE
# ============================================================

leak_count = 0

for row in targeted.itertuples():

    sid = str(row.entity_id)
    target_id = str(row.target_entity_id)

    if target_id in ground_truth[sid]:
        leak_count += 1


print(
    f"GT leakage: "
    f"{leak_count}"
)

if leak_count:
    raise RuntimeError(
        "GT LEAKAGE DETECTED."
    )


# ============================================================
# DUPLICATES
# ============================================================

duplicate_count = targeted.duplicated(
    subset=[
        "entity_id",
        "target_entity_id",
    ]
).sum()

print(
    f"Duplicate pairs: "
    f"{duplicate_count}"
)


# ============================================================
# DISTRIBUTIONS
# ============================================================

if len(targeted):

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
                "block_count",
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

strong_path = (
    OUT_DIR
    / "multi_block_candidates.tsv"
)

control_path = (
    OUT_DIR
    / "one_block_control.tsv"
)

targeted_path = (
    OUT_DIR
    / "targeted_negatives_union.tsv"
)

strong.to_csv(
    strong_path,
    sep="\t",
    index=False,
)

# Save only a small diagnostic control.
control_rows = []

for sid, heap in one_block_heaps.items():

    for (
        block_count,
        tie,
        source_name,
        target_id,
        target_name,
        target_address,
        target_country,
        matched_blocks,
    ) in heap:

        control_rows.append(
            {
                **source1[sid],
                "target_source": source_name,
                "target_entity_id": target_id,
                "target_business_name": target_name,
                "target_business_address": target_address,
                "target_country": target_country,
                "block_count": block_count,
                "matched_block": matched_blocks,
            }
        )

pd.DataFrame(control_rows).to_csv(
    control_path,
    sep="\t",
    index=False,
)

targeted.to_csv(
    targeted_path,
    sep="\t",
    index=False,
)

print()
print("=" * 80)
print("C-v3 COMPLETE")
print("=" * 80)

print(
    f"Multi-block candidates : "
    f"{strong_path}"
)

print(
    f"One-block control      : "
    f"{control_path}"
)

print(
    f"Targeted negatives     : "
    f"{targeted_path}"
)