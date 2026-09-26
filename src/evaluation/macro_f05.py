"""
Official-style entity-level Macro F0.5 evaluator.

For each Source1:
- zero true matches:
    empty prediction => 1.0
    any predicted match => 0.0
- nonzero true matches:
    F0.5 = 1.25 * precision * recall / (0.25 * precision + recall)

Final metric:
    arithmetic mean of entity-level F0.5 over all Source1 entities.
"""

from __future__ import annotations

import pandas as pd


def f05(precision: float, recall: float) -> float:
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return 1.25 * precision * recall / denom


def parse_ids(value) -> set[str]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return set()
    return {x.strip() for x in text.split(",") if x.strip()}


def macro_f05_from_predictions(
    source1_ids,
    ground_truth,
    predictions,
    threshold: float,
):
    """
    predictions columns required:
        source1_entity_id
        score

    Every prediction row represents a candidate pair. Rows with score >=
    threshold are treated as predicted matches.

    ground_truth columns:
        source1_entity_id
        matched_entity_ids
    """
    gt_map = {
        str(row["source1_entity_id"]): parse_ids(
            row["matched_entity_ids"]
        )
        for _, row in ground_truth.iterrows()
    }

    grouped_predictions = {}
    selected = predictions[predictions["score"] >= threshold]

    for sid, group in selected.groupby("source1_entity_id"):
        grouped_predictions[str(sid)] = {
            str(target_id)
            for target_id in group["target_entity_id"]
        }

    values = []

    for sid in source1_ids:
        sid = str(sid)
        truth = gt_map.get(sid, set())
        pred = grouped_predictions.get(sid, set())

        if not truth:
            values.append(1.0 if not pred else 0.0)
            continue

        tp = len(truth & pred)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(truth)
        values.append(f05(precision, recall))

    return sum(values) / len(values) if values else 0.0


def tune_threshold(
    source1_ids,
    ground_truth,
    predictions,
    thresholds=None,
):
    if thresholds is None:
        thresholds = [x / 1000 for x in range(1, 1000)]

    best_threshold = None
    best_score = -1.0

    for threshold in thresholds:
        score = macro_f05_from_predictions(
            source1_ids,
            ground_truth,
            predictions,
            threshold,
        )
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
