from pathlib import Path
from collections import defaultdict
import heapq
import re

import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "raw" / "dataset"
TRAIN = DATA / "train"

S1_PATH = TRAIN / "train_source1.tsv"
S2_PATH = TRAIN / "train_source2.tsv"
S3_PATH = TRAIN / "train_source3.tsv"
GT_PATH = TRAIN / "train_ground_truth.tsv"

OUT = ROOT / "experiments" / "ground_truth_inspection"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_ENTITIES = 2_000

# Keep strongest false candidates per Source1 entity
TOP_K_PER_ENTITY = 10

# Memory-conscious chunking
CHUNK_SIZE = 100_000

RANDOM_STATE = 42


# ============================================================
# NORMALIZATION / BLOCKING
# ============================================================

def norm_text(x):
    """
    Lightweight Unicode-aware normalization.

    IMPORTANT:
    We are NOT transliterating here.
    Cross-script behavior is something we want to observe.
    """
    if pd.isna(x):
        return ""

    x = str(x).lower()

    # Keep Unicode word characters.
    x = re.sub(r"[^\w]+", " ", x, flags=re.UNICODE)

    return re.sub(r"\s+", " ", x).strip()


def first_token(x):
    """
    First normalized token used for exploratory blocking.
    """
    normalized = norm_text(x)

    if not normalized:
        return ""

    return normalized.split()[0]


def block_key(country, name):
    """
    Exploratory blocking key.

    country + first business-name token
    """

    if pd.isna(country):
        country = ""

    country = str(country).strip().lower()

    return (
        country,
        first_token(name)
    )


# ============================================================
# SIMILARITY
# ============================================================

def similarity(a, b):
    return fuzz.ratio(
        str(a) if a is not None else "",
        str(b) if b is not None else ""
    )


def combined_score(name_similarity, address_similarity):
    """
    TEMPORARY DISCOVERY SCORE ONLY.

    This is NOT our final matching score.

    We intentionally give slightly more importance
    to business name than address.
    """

    return (
        0.60 * name_similarity
        + 0.40 * address_similarity
    )


# ============================================================
# TOP-K HEAP
# ============================================================

def push_top_k(heap, item, k):
    """
    Keep only the strongest K candidates.

    heap item:

        (score, unique_id, payload)

    This prevents millions of hard-negative pairs
    from accumulating in RAM.
    """

    score, unique_id, payload = item

    if len(heap) < k:
        heapq.heappush(
            heap,
            item
        )

    elif score > heap[0][0]:
        heapq.heapreplace(
            heap,
            item
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("HARD-NEGATIVE ANALYSIS (OPTIMIZED)")
print("=" * 70)


# ============================================================
# 1. READ GROUND TRUTH
# ============================================================

print("\nReading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)


# ------------------------------------------------------------
# Build:
#
# Source1 ID -> set of true Source2/3 IDs
# ------------------------------------------------------------

matched = {}

for row in gt.itertuples(index=False):

    source1_id = str(
        row.source1_entity_id
    )

    matched_ids = {
        x.strip()
        for x in row.matched_entity_ids.split(",")
        if x.strip()
    }

    matched[source1_id] = matched_ids


all_s1_ids = gt[
    "source1_entity_id"
].astype(str).tolist()


# ============================================================
# 2. SAMPLE SOURCE1 ENTITIES
# ============================================================

sample_ids = (
    pd.Series(all_s1_ids)
    .sample(
        n=min(
            SAMPLE_ENTITIES,
            len(all_s1_ids)
        ),
        random_state=RANDOM_STATE
    )
    .tolist()
)

sample_set = set(sample_ids)

print(
    f"Sampled S1 entities: "
    f"{len(sample_ids):,}"
)


# ============================================================
# 3. LOAD ONLY SAMPLED SOURCE1
# ============================================================

print("\nReading Source1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s1 = s1[
    s1["entity_id"].isin(sample_set)
].copy()


# ------------------------------------------------------------
# CRITICAL OPTIMIZATION
#
# Instead of:
#
# s1[s1["entity_id"] == sid]
#
# repeatedly scanning 2.2M rows,
# build a dictionary once.
# ------------------------------------------------------------

s1_lookup = (
    s1
    .set_index("entity_id")
    .to_dict("index")
)

print(
    f"S1 records retrieved: "
    f"{len(s1_lookup):,}"
)


# ============================================================
# 4. BUILD BLOCK -> SOURCE1 IDS
# ============================================================

block_to_s1 = defaultdict(list)


for source1_id, row in s1_lookup.items():

    key = block_key(
        row["country"],
        row["business_name"]
    )

    block_to_s1[key].append(
        source1_id
    )


print(
    f"Unique S1 blocking keys: "
    f"{len(block_to_s1):,}"
)


# ============================================================
# 5. TOP-K STORAGE
# ============================================================

# Every Source1 entity gets its own small heap.
#
# Instead of potentially storing millions of negatives,
# we retain only the strongest 10.

top_candidates = {
    source1_id: []
    for source1_id in sample_ids
}


# ============================================================
# COUNTERS
# ============================================================

candidate_count = 0

negative_count = 0

serial = 0


# ============================================================
# SOURCE SCANNING FUNCTION
# ============================================================

def process_source(path, source_name):

    global candidate_count
    global negative_count
    global serial

    print(
        f"\nScanning {source_name}..."
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ],
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        # ----------------------------------------------------
        # Create blocking key for every source row.
        # ----------------------------------------------------

        chunk["__block"] = [
            block_key(
                country,
                name
            )
            for country, name in zip(
                chunk["country"],
                chunk["business_name"]
            )
        ]


        # ----------------------------------------------------
        # KEEP ONLY BLOCKS THAT EXIST IN OUR SAMPLED S1.
        #
        # This is the major reduction.
        # ----------------------------------------------------

        relevant = chunk[
            chunk["__block"].isin(
                block_to_s1
            )
        ]


        if relevant.empty:

            if chunk_number % 10 == 0:

                print(
                    f"  chunks {chunk_number:>3} "
                    f"| no relevant blocks"
                )

            continue


        # ----------------------------------------------------
        # Group candidates by blocking key.
        # ----------------------------------------------------

        for block, rows in relevant.groupby(
            "__block",
            sort=False
        ):

            source1_ids = block_to_s1[
                block
            ]


            # ------------------------------------------------
            # Iterate ONLY through relevant rows.
            # ------------------------------------------------

            for row in rows.itertuples(
                index=False
            ):

                candidate_id = str(
                    row.entity_id
                )


                # --------------------------------------------
                # Compare candidate against S1 records
                # sharing the block.
                # --------------------------------------------

                for source1_id in source1_ids:

                    # ----------------------------------------
                    # VERY IMPORTANT:
                    #
                    # If this is a known TRUE MATCH,
                    # it is NOT a hard negative.
                    # ----------------------------------------

                    if candidate_id in matched.get(
                        source1_id,
                        set()
                    ):
                        continue


                    candidate_count += 1

                    negative_count += 1


                    source1 = s1_lookup[
                        source1_id
                    ]


                    # ----------------------------------------
                    # Similarity features
                    # ----------------------------------------

                    name_sim = similarity(
                        source1["business_name"],
                        row.business_name
                    )


                    address_sim = similarity(
                        source1["business_address"],
                        row.business_address
                    )


                    score = combined_score(
                        name_sim,
                        address_sim
                    )


                    serial += 1


                    payload = {

                        "source1_entity_id":
                            source1_id,

                        "source":
                            source_name,

                        "candidate_entity_id":
                            candidate_id,

                        "country":
                            source1["country"],

                        "source1_business_name":
                            source1["business_name"],

                        "candidate_business_name":
                            row.business_name,

                        "source1_business_address":
                            source1["business_address"],

                        "candidate_business_address":
                            row.business_address,

                        "name_similarity":
                            name_sim,

                        "address_similarity":
                            address_sim,

                        "combined_score":
                            score
                    }


                    # ----------------------------------------
                    # Keep only top K for this S1 entity.
                    # ----------------------------------------

                    push_top_k(
                        top_candidates[
                            source1_id
                        ],
                        (
                            score,
                            serial,
                            payload
                        ),
                        TOP_K_PER_ENTITY
                    )


        if chunk_number % 10 == 0:

            kept = sum(
                len(heap)
                for heap in top_candidates.values()
            )

            print(
                f"  chunks {chunk_number:>3} "
                f"| relevant rows {len(relevant):,} "
                f"| negatives checked {negative_count:,} "
                f"| kept {kept:,}"
            )


# ============================================================
# 6. SCAN SOURCE2
# ============================================================

process_source(
    S2_PATH,
    "source2"
)


# ============================================================
# 7. SCAN SOURCE3
# ============================================================

process_source(
    S3_PATH,
    "source3"
)


# ============================================================
# 8. FLATTEN TOP-K RESULTS
# ============================================================

rows = []

for source1_id, heap in top_candidates.items():

    for _, _, payload in heap:

        rows.append(
            payload
        )


hard_negatives = pd.DataFrame(
    rows
)


if not hard_negatives.empty:

    hard_negatives = (
        hard_negatives
        .sort_values(
            [
                "combined_score",
                "name_similarity",
                "address_similarity"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# 9. SAVE
# ============================================================

output_path = (
    OUT /
    "hard_negative_pairs.tsv"
)


hard_negatives.to_csv(
    output_path,
    sep="\t",
    index=False
)


# ============================================================
# 10. BASIC RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Candidate negatives evaluated: "
    f"{negative_count:,}"
)

print(
    f"Hard negatives retained: "
    f"{len(hard_negatives):,}"
)

print(
    f"Output: "
    f"{output_path}"
)


if hard_negatives.empty:

    print(
        "\nNo hard negatives found "
        "with this blocking key."
    )

    raise SystemExit


# ============================================================
# 11. TOP 30 DANGEROUS NEGATIVES
# ============================================================

print(
    "\nTop 30 dangerous hard negatives:"
)


display_columns = [
    "source1_entity_id",
    "source",
    "candidate_entity_id",
    "country",
    "name_similarity",
    "address_similarity",
    "combined_score",
    "source1_business_name",
    "candidate_business_name"
]


print(
    hard_negatives[
        display_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 12. THRESHOLD ANALYSIS
# ============================================================

print(
    "\nThreshold counts:"
)


for threshold in [
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    95
]:

    name_count = int(
        (
            hard_negatives[
                "name_similarity"
            ] >= threshold
        ).sum()
    )


    address_count = int(
        (
            hard_negatives[
                "address_similarity"
            ] >= threshold
        ).sum()
    )


    both_count = int(
        (
            (
                hard_negatives[
                    "name_similarity"
                ] >= threshold
            )
            &
            (
                hard_negatives[
                    "address_similarity"
                ] >= threshold
            )
        ).sum()
    )


    print(
        f"  >= {threshold:2}: "
        f"name={name_count:6,} | "
        f"address={address_count:6,} | "
        f"both={both_count:6,}"
    )


# ============================================================
# 13. COMBINED SCORE ANALYSIS
# ============================================================

print(
    "\nCombined-score counts:"
)


for threshold in [
    40,
    50,
    60,
    70,
    80,
    90
]:

    count = int(
        (
            hard_negatives[
                "combined_score"
            ] >= threshold
        ).sum()
    )

    print(
        f"  >= {threshold:2}: "
        f"{count:,}"
    )


# ============================================================
# 14. DANGER ZONES
# ============================================================

print(
    "\nDanger zones:"
)


danger_zones = {

    "name>=80 & address>=80":
        (
            hard_negatives["name_similarity"] >= 80
        )
        &
        (
            hard_negatives["address_similarity"] >= 80
        ),


    "name>=90 & address<60":
        (
            hard_negatives["name_similarity"] >= 90
        )
        &
        (
            hard_negatives["address_similarity"] < 60
        ),


    "name<60 & address>=80":
        (
            hard_negatives["name_similarity"] < 60
        )
        &
        (
            hard_negatives["address_similarity"] >= 80
        ),


    "name<40 & address>=70":
        (
            hard_negatives["name_similarity"] < 40
        )
        &
        (
            hard_negatives["address_similarity"] >= 70
        ),


    "name>=80 & address<40":
        (
            hard_negatives["name_similarity"] >= 80
        )
        &
        (
            hard_negatives["address_similarity"] < 40
        )
}


for label, mask in danger_zones.items():

    print(
        f"  {label}: "
        f"{int(mask.sum()):,}"
    )


# ============================================================
# 15. SOURCE2 VS SOURCE3
# ============================================================

print(
    "\nBy source:"
)


source_stats = (
    hard_negatives
    .groupby("source")[
        [
            "name_similarity",
            "address_similarity",
            "combined_score"
        ]
    ]
    .agg(
        [
            "count",
            "mean",
            "median"
        ]
    )
    .round(2)
)


print(
    source_stats.to_string()
)


# ============================================================
# 16. US VS INDIA
# ============================================================

print(
    "\nBy country:"
)


country_stats = (
    hard_negatives
    .groupby("country")[
        [
            "name_similarity",
            "address_similarity",
            "combined_score"
        ]
    ]
    .agg(
        [
            "count",
            "mean",
            "median"
        ]
    )
    .round(2)
)


print(
    country_stats.to_string()
)


# ============================================================
# 17. DISTRIBUTION PER SOURCE1
# ============================================================

print(
    "\nHard negatives per sampled S1 entity:"
)


per_entity = (
    hard_negatives
    .groupby("source1_entity_id")
    .size()
    .describe()
    .round(2)
)


print(
    per_entity.to_string()
)


print("\nDone.")