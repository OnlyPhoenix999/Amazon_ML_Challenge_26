"""
Generate a manageable training-pair set from the frozen ALL_V4 blocker.

Important:
- Uses the exact existing V4 build_keys() from:
    src/preprocessing/benchmark_combo2_address_first_v4.py
- Does NOT modify normalization.py.
- Streams Source2 and Source3 in chunks.
- Keeps every recovered positive for sampled Source1 entities.
- Keeps a bounded number of hard negatives per Source1 using blocking-key
  agreement as a cheap hardness signal.
- Does NOT materialize the ~175M candidate population in RAM.

Training philosophy:
- Start with 2,000 S1 entities because that matches the established
  ALL_V4 benchmark population.
- Once correctness is verified, scale the S1 sample and negative budget.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.benchmark_combo2_address_first_v4 import build_keys


TARGET_USECOLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

BLOCK_ORDER = (
    "name_first_v2",
    "name_prefix2_v2",
    "address_number_v2",
    "address_first_v4",
)

# The frozen V4 build_keys() returns these dictionary keys.
# The suffixes above are our canonical names for reporting.
V4_KEY_MAP = {
    "name_first_v2": "name_first",
    "name_prefix2_v2": "name_prefix2",
    "address_number_v2": "address_number",
    "address_first_v4": "address_first",
}


def parse_gt(value: Any) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return set()
    return {x.strip() for x in text.split(",") if x.strip()}


def stable_hash(*parts: str, seed: int = 42) -> int:
    payload = ("|".join((str(seed),) + tuple(parts))).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "big",
        signed=False,
    )


def push_negative(
    heap: list[tuple[int, int, str, str, str, str, str]],
    limit: int,
    *,
    score: int,
    tie: int,
    source: str,
    target_id: str,
    name: str,
    address: str,
    country: str,
) -> None:
    """
    Keep the top `limit` hard negatives.

    Higher block agreement => higher score.
    Stable hash gives deterministic diversity among candidates with equal
    block agreement.
    """
    item = (score, tie, source, target_id, name, address, country)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def reservoir_source1(
    source1_path: Path,
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    import random

    rng = random.Random(seed)
    sample: list[dict[str, str]] = []
    seen = 0

    for chunk in pd.read_csv(
        source1_path,
        sep="\t",
        dtype="string",
        chunksize=100_000,
        usecols=TARGET_USECOLS,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False, name=None):
            seen += 1
            record = {
                "entity_id": str(row[0]),
                "business_name": str(row[1] or ""),
                "business_address": str(row[2] or ""),
                "country": str(row[3] or ""),
            }

            if len(sample) < sample_size:
                sample.append(record)
                continue

            j = rng.randint(1, seen)
            if j <= sample_size:
                sample[j - 1] = record

    return sample


def load_sample_ground_truth(
    gt_path: Path,
    sample_ids: set[str],
) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {sid: set() for sid in sample_ids}

    for chunk in pd.read_csv(
        gt_path,
        sep="\t",
        dtype="string",
        chunksize=100_000,
        usecols=["source1_entity_id", "matched_entity_ids"],
        keep_default_na=False,
    ):
        subset = chunk[chunk["source1_entity_id"].isin(sample_ids)]
        for row in subset.itertuples(index=False, name=None):
            lookup[str(row[0])].update(parse_gt(row[1]))

    return lookup


def build_source1_indexes(
    sample: list[dict[str, str]],
) -> dict[str, dict[tuple[str, str], list[str]]]:
    indexes = {
        block: defaultdict(list)
        for block in BLOCK_ORDER
    }

    for record in sample:
        sid = record["entity_id"]
        keys = build_keys(
            record["business_name"],
            record["business_address"],
            record["country"],
            True,
        )
        for block in BLOCK_ORDER:
            actual_key_name = V4_KEY_MAP[block]
            key = keys.get(actual_key_name)

            if (
                key
                and isinstance(key, tuple)
                and len(key) >= 2
                and key[0]
                and key[1]
            ):
                indexes[block][(str(key[0]), str(key[1]))].append(sid)

    return indexes


def process_target_file(
    path: Path,
    source_name: str,
    source1_indexes,
    ground_truth,
    positives,
    negative_heaps,
    negative_limit: int,
    chunk_size: int,
    seed: int,
) -> tuple[int, int]:
    rows_processed = 0
    candidate_occurrences = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        chunksize=chunk_size,
        usecols=TARGET_USECOLS,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False, name=None):
            target_id = str(row[0])
            name = str(row[1] or "")
            address = str(row[2] or "")
            country = str(row[3] or "")

            key_dict = build_keys(name, address, country, True)

            # One target can collide through several V4 keys with the same S1.
            candidate_hits: dict[str, set[str]] = {}

            for block in BLOCK_ORDER:
                actual_key_name = V4_KEY_MAP[block]
                key = key_dict.get(actual_key_name)

                if (
                    not key
                    or not isinstance(key, tuple)
                    or len(key) < 2
                    or not key[0]
                    or not key[1]
                ):
                    continue

                for sid in source1_indexes[block].get(
                    (str(key[0]), str(key[1])),
                    (),
                ):
                    candidate_hits.setdefault(sid, set()).add(block)

            for sid, matched_blocks in candidate_hits.items():
                candidate_occurrences += 1
                gt = ground_truth[sid]

                pair_key = (source_name, target_id)

                if target_id in gt:
                    previous = positives[sid].get(pair_key)
                    if previous is None:
                        positives[sid][pair_key] = (
                            ",".join(
                                b for b in BLOCK_ORDER if b in matched_blocks
                            ),
                            name,
                            address,
                            country,
                        )
                    else:
                        old_blocks, old_name, old_address, old_country = previous
                        merged = set(old_blocks.split(","))
                        merged.update(matched_blocks)
                        positives[sid][pair_key] = (
                            ",".join(
                                b for b in BLOCK_ORDER if b in merged
                            ),
                            old_name,
                            old_address,
                            old_country,
                        )
                elif negative_limit > 0:
                    block_count = len(matched_blocks)
                    # Multi-key collisions are more valuable hard negatives.
                    # The tie-breaker makes selection deterministic.
                    score = block_count * 10**18
                    tie = stable_hash(
                        sid,
                        source_name,
                        target_id,
                        seed=seed,
                    )
                    push_negative(
                        negative_heaps[sid],
                        negative_limit,
                        score=score,
                        tie=tie,
                        source=source_name,
                        target_id=target_id,
                        name=name,
                        address=address,
                        country=country,
                    )

        rows_processed += len(chunk)
        print(
            f"\r{source_name}: {rows_processed:,} rows scanned | "
            f"candidate occurrences {candidate_occurrences:,}",
            end="",
            flush=True,
        )

    print()
    return rows_processed, candidate_occurrences


def write_outputs(
    output_dir: Path,
    sample: list[dict[str, str]],
    positives,
    negative_heaps,
    ground_truth,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "source1_training_sample.tsv"
    candidate_path = output_dir / "training_pairs_v4.tsv"

    pd.DataFrame(sample).to_csv(
        sample_path,
        sep="\t",
        index=False,
    )

    rows = []
    recovered = 0
    gt_total = 0

    for record in sample:
        sid = record["entity_id"]
        gt = ground_truth[sid]
        gt_total += len(gt)

        pos_map = positives[sid]
        recovered += len(pos_map)

        for (source, target_id), (
            blocks,
            name,
            address,
            country,
        ) in pos_map.items():
            rows.append(
                {
                    **record,
                    "target_source": source,
                    "target_entity_id": target_id,
                    "target_business_name": name,
                    "target_business_address": address,
                    "target_country": country,
                    "label": 1,
                    "matched_block": blocks,
                }
            )

        # Sort hard negatives strongest-first, then keep source/ID order.
        negatives = sorted(
            negative_heaps[sid],
            key=lambda x: (-x[0], -x[1], x[2], x[3]),
        )
        for _, _, source, target_id, name, address, country in negatives:
            rows.append(
                {
                    **record,
                    "target_source": source,
                    "target_entity_id": target_id,
                    "target_business_name": name,
                    "target_business_address": address,
                    "target_country": country,
                    "label": 0,
                    "matched_block": "hard_negative",
                }
            )

    pd.DataFrame(rows).to_csv(
        candidate_path,
        sep="\t",
        index=False,
    )

    stats = {
        "sampled_source1_entities": len(sample),
        "ground_truth_pairs": gt_total,
        "recovered_positive_pairs": recovered,
        "candidate_recall_percent": (
            100.0 * recovered / gt_total if gt_total else 100.0
        ),
        "positive_rows": sum(1 for r in rows if r["label"] == 1),
        "negative_rows": sum(1 for r in rows if r["label"] == 0),
        "total_training_pairs": len(rows),
    }

    (output_dir / "training_pairs_v4_metadata.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--negatives-per-entity", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "dataset" / "train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "training_pairs",
    )
    args = parser.parse_args()

    source1 = args.train_dir / "train_source1.tsv"
    source2 = args.train_dir / "train_source2.tsv"
    source3 = args.train_dir / "train_source3.tsv"
    gt_path = args.train_dir / "train_ground_truth.tsv"

    for path in (source1, source2, source3, gt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("=" * 78)
    print("ALL_V4 TRAINING-PAIR GENERATION")
    print("=" * 78)
    print(f"Source1 sample size     : {args.sample_size:,}")
    print(f"Negatives / Source1    : {args.negatives_per_entity:,}")
    print(f"Target chunk size      : {args.chunk_size:,}")
    print("Blocker                 : FROZEN ALL_V4")

    sample = reservoir_source1(
        source1,
        args.sample_size,
        args.seed,
    )
    sample_ids = {x["entity_id"] for x in sample}

    gt = load_sample_ground_truth(gt_path, sample_ids)
    s1_indexes = build_source1_indexes(sample)

    print("\nSource1 V4 index sizes:")
    for block in BLOCK_ORDER:
        print(f"  {block:22s}: {len(s1_indexes[block]):,}")

    if not any(len(s1_indexes[b]) for b in BLOCK_ORDER):
        raise RuntimeError(
            "ALL_V4 Source1 indexes are empty. "
            "Check V4_KEY_MAP / build_keys() before scanning targets."
        )

    positives = {sid: {} for sid in sample_ids}
    negative_heaps = {sid: [] for sid in sample_ids}

    total_candidates = 0

    for source_name, path in (
        ("source2", source2),
        ("source3", source3),
    ):
        _, candidates = process_target_file(
            path=path,
            source_name=source_name,
            source1_indexes=s1_indexes,
            ground_truth=gt,
            positives=positives,
            negative_heaps=negative_heaps,
            negative_limit=args.negatives_per_entity,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )
        total_candidates += candidates

    stats = write_outputs(
        args.output_dir,
        sample,
        positives,
        negative_heaps,
        gt,
    )
    stats["candidate_occurrences_scanned"] = total_candidates

    print("=" * 78)
    print("TRAINING PAIR GENERATION COMPLETE")
    print("=" * 78)
    for k, v in stats.items():
        print(f"{k:34s}: {v}")


if __name__ == "__main__":
    main()
