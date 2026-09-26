"""
XGBoost LambdaMART experiment for Amazon ML Challenge 2026.

Controlled experiment:
- ALL_V4 candidate pairs are already represented in training_features_v4.tsv.
- Exactly the established 39 handcrafted features are used.
- Entity split is fixed: 1400 train / 300 dev / 300 holdout.
- The holdout is never used for threshold/model selection.
- XGBRanker with objective=rank:ndcg is the only model change.

Run from repository root:
    python src/models/train_xgb_lambdamart.py

Optional:
    python src/models/train_xgb_lambdamart.py --device cuda
    python src/models/train_xgb_lambdamart.py --n-estimators 500

Outputs are written under experiments/models/lambdamart/.
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
    "name_ratio",
    "name_wratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_exact",
    "name_length_diff",
    "name_length_ratio",
    "name_token_count_diff",
    "address_ratio",
    "address_wratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_exact",
    "address_length_diff",
    "address_length_ratio",
    "address_token_count_diff",
    "norm_name_ratio",
    "norm_name_wratio",
    "norm_name_token_sort_ratio",
    "norm_name_token_set_ratio",
    "norm_name_exact",
    "name_no_suffix_ratio",
    "name_no_suffix_wratio",
    "name_no_suffix_exact",
    "name_first_token_same",
    "name_last_token_same",
    "norm_address_ratio",
    "norm_address_wratio",
    "norm_address_token_sort_ratio",
    "norm_address_token_set_ratio",
    "norm_address_exact",
    "address_number_match",
    "same_country",
    "source_is_s2",
    "source_is_s3",
    "address1_missing",
    "address2_missing",
    "name_script_same",
    "address_script_same",
]

ID_COLUMNS = ["entity_id", "target_entity_id"]
LABEL_COLUMN = "label"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="data/processed/training_features_v4.tsv",
    )
    parser.add_argument(
        "--train-entities",
        default="experiments/splits/train_entities.tsv",
    )
    parser.add_argument(
        "--dev-entities",
        default="experiments/splits/dev_entities.tsv",
    )
    parser.add_argument(
        "--holdout-entities",
        default="experiments/splits/holdout_entities.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/models/lambdamart",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--threshold-grid-size", type=int, default=401)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def load_ids(path: str) -> set[str]:
    df = pd.read_csv(path, sep="\t", usecols=["entity_id"])
    return set(df["entity_id"].astype(str))


def prepare_split(df: pd.DataFrame, entity_ids: set[str]):
    part = df[df["entity_id"].isin(entity_ids)].copy()
    # XGBRanker requires rows belonging to each query/entity to be contiguous.
    part.sort_values("entity_id", inplace=True, kind="stable")
    groups = part.groupby("entity_id", sort=False).size().to_numpy(dtype=np.int32)
    return part, groups


def validate_inputs(df: pd.DataFrame, train_ids, dev_ids, holdout_ids):
    required = set(ID_COLUMNS + [LABEL_COLUMN] + FEATURE_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if train_ids & dev_ids:
        raise ValueError("Train/dev entity overlap detected.")
    if train_ids & holdout_ids:
        raise ValueError("Train/holdout entity overlap detected.")
    if dev_ids & holdout_ids:
        raise ValueError("Dev/holdout entity overlap detected.")

    all_split_ids = train_ids | dev_ids | holdout_ids
    present_ids = set(df["entity_id"].astype(str))
    missing_entities = sorted(all_split_ids - present_ids)
    if missing_entities:
        raise ValueError(
            f"{len(missing_entities)} split entities are absent from feature TSV."
        )


def entity_fbeta(
    scored: pd.DataFrame,
    threshold: float,
    beta: float = 0.5,
):
    """Entity-level Macro F_beta over recoverable ALL_V4 candidates."""
    beta2 = beta * beta
    scores = []

    for _, group in scored.groupby("entity_id", sort=False):
        truth = set(
            group.loc[group["label"] == 1, "target_entity_id"].astype(str)
        )
        predicted = set(
            group.loc[group["score"] >= threshold, "target_entity_id"].astype(str)
        )

        if not truth and not predicted:
            scores.append(1.0)
            continue
        if not truth or not predicted:
            scores.append(0.0)
            continue

        tp = len(truth & predicted)
        precision = tp / len(predicted)
        recall = tp / len(truth)

        if precision == 0.0 or recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(
                (1.0 + beta2) * precision * recall
                / (beta2 * precision + recall)
            )

    return float(np.mean(scores))


def pair_metrics(scored: pd.DataFrame, threshold: float):
    predicted = scored["score"] >= threshold
    truth = scored["label"].astype(int) == 1

    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_matches": int(predicted.sum()),
        "ground_truth_recoverable_matches": int(truth.sum()),
    }


def select_dev_threshold(scored: pd.DataFrame, grid_size: int):
    # Quantile grid keeps threshold search cheap while still covering the
    # actual score distribution. Threshold selection uses DEV only.
    quantiles = np.linspace(0.0, 1.0, grid_size)
    thresholds = np.unique(np.quantile(scored["score"].to_numpy(), quantiles))

    best_score = -1.0
    best_threshold = None

    for threshold in thresholds:
        score = entity_fbeta(scored, threshold)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    print("Loading feature table...")
    df = pd.read_csv(args.features, sep="\t")
    df["entity_id"] = df["entity_id"].astype(str)
    df["target_entity_id"] = df["target_entity_id"].astype(str)

    train_ids = load_ids(args.train_entities)
    dev_ids = load_ids(args.dev_entities)
    holdout_ids = load_ids(args.holdout_entities)

    validate_inputs(df, train_ids, dev_ids, holdout_ids)

    train_df, train_group = prepare_split(df, train_ids)
    dev_df, dev_group = prepare_split(df, dev_ids)
    holdout_df, holdout_group = prepare_split(df, holdout_ids)

    print(
        f"Train:   {train_df['entity_id'].nunique()} entities, "
        f"{len(train_df):,} candidate rows, "
        f"{int(train_df[LABEL_COLUMN].sum()):,} positives"
    )
    print(
        f"Dev:     {dev_df['entity_id'].nunique()} entities, "
        f"{len(dev_df):,} candidate rows, "
        f"{int(dev_df[LABEL_COLUMN].sum()):,} positives"
    )
    print(
        f"Holdout: {holdout_df['entity_id'].nunique()} entities, "
        f"{len(holdout_df):,} candidate rows, "
        f"{int(holdout_df[LABEL_COLUMN].sum()):,} recoverable positives"
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[LABEL_COLUMN].astype(int)
    X_dev = dev_df[FEATURE_COLUMNS]
    y_dev = dev_df[LABEL_COLUMN].astype(int)
    X_holdout = holdout_df[FEATURE_COLUMNS]

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method="hist",
        device=args.device,
        random_state=42,
        n_jobs=args.n_jobs,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    print("\nTraining XGBRanker...")
    model.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_dev, y_dev)],
        eval_group=[dev_group],
        verbose=50,
    )

    print("\nGenerating DEV scores...")
    dev_scores = model.predict(X_dev)

    dev_scored = dev_df[ID_COLUMNS + [LABEL_COLUMN]].copy()
    dev_scored["score"] = dev_scores

    threshold, dev_macro_f05 = select_dev_threshold(
        dev_scored,
        args.threshold_grid_size,
    )
    dev_pair = pair_metrics(dev_scored, threshold)

    print(f"\nSelected DEV threshold: {threshold:.8f}")
    print(f"DEV Entity Macro F0.5:   {dev_macro_f05:.6f}")
    print(f"DEV pair precision:      {dev_pair['pair_precision']:.6f}")
    print(f"DEV pair recall:         {dev_pair['pair_recall']:.6f}")

    # Holdout is evaluated exactly once using the threshold selected on DEV.
    print("\nGenerating HOLDOUT scores (untouched)...")
    holdout_scores = model.predict(X_holdout)

    holdout_scored = holdout_df[ID_COLUMNS + [LABEL_COLUMN]].copy()
    holdout_scored["score"] = holdout_scores

    holdout_macro_f05 = entity_fbeta(holdout_scored, threshold)
    holdout_pair = pair_metrics(holdout_scored, threshold)

    model_path = output_dir / "xgb_lambdamart_rank_ndcg.json"
    dev_path = output_dir / "dev_predictions.tsv"
    holdout_path = output_dir / "holdout_predictions.tsv"
    summary_path = output_dir / "experiment_summary.json"

    model.save_model(model_path)
    dev_scored.to_csv(dev_path, sep="\t", index=False)
    holdout_scored.to_csv(holdout_path, sep="\t", index=False)

    elapsed = time.time() - started

    summary = {
        "model": "XGBRanker",
        "objective": "rank:ndcg",
        "features": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "candidate_blocker": "ALL_V4",
        "train_entities": len(train_ids),
        "dev_entities": len(dev_ids),
        "holdout_entities": len(holdout_ids),
        "train_rows": len(train_df),
        "dev_rows": len(dev_df),
        "holdout_rows": len(holdout_df),
        "best_iteration": (
            int(model.best_iteration)
            if getattr(model, "best_iteration", None) is not None
            else None
        ),
        "best_eval_ndcg_at_10": (
            float(model.best_score)
            if getattr(model, "best_score", None) is not None
            else None
        ),
        "dev_threshold": threshold,
        "dev_entity_macro_f05": dev_macro_f05,
        "dev_pair_metrics": dev_pair,
        "holdout_entity_macro_f05": holdout_macro_f05,
        "holdout_pair_metrics": holdout_pair,
        "holdout_is_untouched_for_selection": True,
        "missed_pairs_file_used": False,
        "missed_pairs_note": (
            "missed_pairs_v4.tsv contains entities outside this 1400/300/300 split "
            "and is not used in this controlled candidate-scoring experiment."
        ),
        "runtime_seconds": elapsed,
        "device": args.device,
        "xgboost_version": xgb.__version__,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
        "baseline_holdout_entity_macro_f05": 0.919182,
    }

    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n==============================")
    print("LAMBDA MART RESULT")
    print("==============================")
    print(f"DEV threshold:           {threshold:.8f}")
    print(f"DEV Macro F0.5:          {dev_macro_f05:.6f}")
    print(f"HOLDOUT Macro F0.5:      {holdout_macro_f05:.6f}")
    print("Baseline Macro F0.5:     0.919182")
    print(f"Runtime:                 {elapsed:.1f}s")
    print(f"\nSaved model:             {model_path}")
    print(f"Saved DEV predictions:   {dev_path}")
    print(f"Saved HOLDOUT results:   {holdout_path}")
    print(f"Saved summary:            {summary_path}")


if __name__ == "__main__":
    main()
