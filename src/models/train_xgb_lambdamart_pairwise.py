"""
Controlled XGBoost LambdaMART experiment: rank:pairwise.

Run from repository root:
    python src/models/train_xgb_lambdamart_pairwise.py

This uses:
- ALL_V4 candidate pairs represented by training_features_v4.tsv
- exactly the established 39 features
- train_entities.tsv / dev_entities.tsv / holdout_entities.tsv
- train-only fitting
- DEV-only threshold selection
- the project's raw train_ground_truth.tsv for the final entity-level metric

The holdout is never used for threshold selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


FEATURE_COLUMNS = [
    "name_ratio", "name_wratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_exact", "name_length_diff", "name_length_ratio", "name_token_count_diff",
    "address_ratio", "address_wratio", "address_token_sort_ratio",
    "address_token_set_ratio", "address_exact", "address_length_diff",
    "address_length_ratio", "address_token_count_diff",
    "norm_name_ratio", "norm_name_wratio", "norm_name_token_sort_ratio",
    "norm_name_token_set_ratio", "norm_name_exact",
    "name_no_suffix_ratio", "name_no_suffix_wratio", "name_no_suffix_exact",
    "name_first_token_same", "name_last_token_same",
    "norm_address_ratio", "norm_address_wratio", "norm_address_token_sort_ratio",
    "norm_address_token_set_ratio", "norm_address_exact",
    "address_number_match", "same_country", "source_is_s2", "source_is_s3",
    "address1_missing", "address2_missing", "name_script_same", "address_script_same",
]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = ROOT / "data" / "raw" / "dataset" / "train" / "train_ground_truth.tsv"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/training_features_v4.tsv")
    p.add_argument("--train-entities", default="experiments/splits/train_entities.tsv")
    p.add_argument("--dev-entities", default="experiments/splits/dev_entities.tsv")
    p.add_argument("--holdout-entities", default="experiments/splits/holdout_entities.tsv")
    p.add_argument("--ground-truth", default=str(DEFAULT_GT))
    p.add_argument("--output-dir", default="experiments/models/lambdamart_pairwise")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--early-stopping-rounds", type=int, default=40)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def load_ids(path):
    return set(pd.read_csv(path, sep="\t", dtype=str)["entity_id"])


def load_gt(path, entity_ids):
    gt = {x: set() for x in entity_ids}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=250000):
        chunk = chunk.fillna("")
        chunk = chunk[chunk["source1_entity_id"].isin(entity_ids)]
        for row in chunk.itertuples(index=False):
            if row.source1_entity_id in gt:
                gt[row.source1_entity_id].update(
                    x.strip() for x in str(row.matched_entity_ids).split(",") if x.strip()
                )
    return gt


def prepare(df, entity_ids):
    part = df[df.entity_id.isin(entity_ids)].copy()
    part.sort_values("entity_id", kind="stable", inplace=True)
    groups = part.groupby("entity_id", sort=False).size().to_numpy(dtype=np.int32)
    return part, groups


def entity_f05(true_targets, predicted_targets):
    true_targets = set(true_targets)
    predicted_targets = set(predicted_targets)

    if not true_targets:
        return 1.0 if not predicted_targets else 0.0

    tp = len(true_targets & predicted_targets)
    fp = len(predicted_targets - true_targets)
    fn = len(true_targets - predicted_targets)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    if precision == 0.0 or recall == 0.0:
        return 0.0

    return 1.25 * precision * recall / (0.25 * precision + recall)


def evaluate(predictions, gt, threshold):
    scores = []
    total_predicted = 0

    for entity_id, group in predictions.groupby("entity_id", sort=False):
        selected = set(
            group.loc[group["score"] >= threshold, "target_entity_id"].astype(str)
        )
        total_predicted += len(selected)
        scores.append(entity_f05(gt[entity_id], selected))

    return float(np.mean(scores)), total_predicted / len(gt)


def pair_metrics(predictions, gt, threshold):
    tp = fp = fn = 0
    for entity_id, group in predictions.groupby("entity_id", sort=False):
        predicted = set(
            group.loc[group["score"] >= threshold, "target_entity_id"].astype(str)
        )
        truth = gt[entity_id]
        tp += len(predicted & truth)
        fp += len(predicted - truth)
        fn += len(truth - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, tp, fp, fn


def select_threshold(dev_predictions, dev_gt):
    # Same threshold family used by the existing entity-level ranking evaluator.
    best = None
    for threshold in np.round(np.arange(0.55, 0.851, 0.01), 2):
        score, avg = evaluate(dev_predictions, dev_gt, float(threshold))
        row = (score, float(threshold), avg)
        if best is None or score > best[0]:
            best = row
    return best


def main():
    a = args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("Loading feature table...")
    df = pd.read_csv(a.features, sep="\t")
    df["entity_id"] = df["entity_id"].astype(str)
    df["target_entity_id"] = df["target_entity_id"].astype(str)

    train_ids = load_ids(a.train_entities)
    dev_ids = load_ids(a.dev_entities)
    holdout_ids = load_ids(a.holdout_entities)

    if train_ids & dev_ids or train_ids & holdout_ids or dev_ids & holdout_ids:
        raise RuntimeError("Entity overlap detected between splits.")

    train_df, train_group = prepare(df, train_ids)
    dev_df, dev_group = prepare(df, dev_ids)
    holdout_df, holdout_group = prepare(df, holdout_ids)

    print(f"Train:   {len(train_ids):,} entities, {len(train_df):,} rows")
    print(f"Dev:     {len(dev_ids):,} entities, {len(dev_df):,} rows")
    print(f"Holdout: {len(holdout_ids):,} entities, {len(holdout_df):,} rows")

    train_gt = load_gt(a.ground_truth, train_ids)
    dev_gt = load_gt(a.ground_truth, dev_ids)
    holdout_gt = load_gt(a.ground_truth, holdout_ids)

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        eval_metric="ndcg@10",
        n_estimators=a.n_estimators,
        learning_rate=a.learning_rate,
        max_depth=a.max_depth,
        min_child_weight=a.min_child_weight,
        subsample=a.subsample,
        colsample_bytree=a.colsample_bytree,
        tree_method="hist",
        device=a.device,
        random_state=42,
        n_jobs=a.n_jobs,
        early_stopping_rounds=a.early_stopping_rounds,
    )

    print("\nTraining XGBRanker objective=rank:pairwise...")
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df["label"].astype(int),
        group=train_group,
        eval_set=[(dev_df[FEATURE_COLUMNS], dev_df["label"].astype(int))],
        eval_group=[dev_group],
        verbose=50,
    )

    print("\nSelecting threshold on DEV only...")
    dev_pred = dev_df[["entity_id", "target_entity_id"]].copy()
    dev_pred["score"] = model.predict(dev_df[FEATURE_COLUMNS])

    dev_score, threshold, dev_avg = select_threshold(dev_pred, dev_gt)
    dev_precision, dev_recall, dev_tp, dev_fp, dev_fn = pair_metrics(
        dev_pred, dev_gt, threshold
    )

    print(f"DEV threshold:       {threshold:.2f}")
    print(f"DEV Entity Macro F0.5: {dev_score:.6f}")
    print(f"DEV pair precision:   {dev_precision:.6f}")
    print(f"DEV pair recall:      {dev_recall:.6f}")

    print("\nEvaluating untouched HOLDOUT...")
    hold_pred = holdout_df[["entity_id", "target_entity_id"]].copy()
    hold_pred["score"] = model.predict(holdout_df[FEATURE_COLUMNS])

    hold_score, hold_avg = evaluate(hold_pred, holdout_gt, threshold)
    hold_precision, hold_recall, hold_tp, hold_fp, hold_fn = pair_metrics(
        hold_pred, holdout_gt, threshold
    )

    model_path = out / "xgb_lambdamart_rank_pairwise.json"
    dev_path = out / "dev_predictions.tsv"
    hold_path = out / "holdout_predictions.tsv"
    summary_path = out / "experiment_summary.json"

    model.save_model(model_path)
    dev_pred.to_csv(dev_path, sep="\t", index=False)
    hold_pred.to_csv(hold_path, sep="\t", index=False)

    summary = {
        "model": "XGBRanker",
        "objective": "rank:pairwise",
        "feature_count": 39,
        "candidate_blocker": "ALL_V4",
        "train_entities": len(train_ids),
        "dev_entities": len(dev_ids),
        "holdout_entities": len(holdout_ids),
        "dev_threshold": threshold,
        "dev_entity_macro_f05": dev_score,
        "dev_pair_precision": dev_precision,
        "dev_pair_recall": dev_recall,
        "holdout_entity_macro_f05": hold_score,
        "holdout_pair_precision": hold_precision,
        "holdout_pair_recall": hold_recall,
        "holdout_avg_predictions_per_entity": hold_avg,
        "baseline_holdout_entity_macro_f05": 0.919182,
        "holdout_untouched_for_selection": True,
        "runtime_seconds": time.time() - started,
        "device": a.device,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n==============================")
    print("LAMBDA MART PAIRWISE RESULT")
    print("==============================")
    print(f"DEV threshold:        {threshold:.2f}")
    print(f"DEV Macro F0.5:       {dev_score:.6f}")
    print(f"HOLDOUT Macro F0.5:   {hold_score:.6f}")
    print(f"Baseline:             0.919182")
    print(f"Runtime:              {time.time() - started:.1f}s")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
