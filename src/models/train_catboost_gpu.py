"""
Train the CatBoost GPU A/B baseline on the exact same feature table and
Source1-grouped split as the XGBoost experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.macro_f05 import tune_threshold


META_COLUMNS = {
    "entity_id",
    "target_source",
    "target_entity_id",
    "label",
    "matched_block",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--source1-sample", required=True)
    parser.add_argument("--output-dir", default="experiments/models/catboost_gpu")
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features, sep="\t")

    feature_cols = [
        c for c in df.columns if c not in META_COLUMNS
    ]

    sample_ids = (
        pd.read_csv(args.source1_sample, sep="\t", dtype="string")["entity_id"]
        .astype(str)
        .drop_duplicates()
        .to_numpy()
    )
    sample_frame = pd.DataFrame({"entity_id": sample_ids})

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.valid_size,
        random_state=args.seed,
    )
    train_groups_idx, valid_groups_idx = next(
        splitter.split(
            sample_frame,
            groups=sample_frame["entity_id"],
        )
    )

    train_groups = set(sample_frame.iloc[train_groups_idx]["entity_id"])
    valid_groups = set(sample_frame.iloc[valid_groups_idx]["entity_id"])

    df_groups = df["entity_id"].astype(str)
    train_mask = df_groups.isin(train_groups)
    valid_mask = df_groups.isin(valid_groups)

    X_train = df.loc[train_mask, feature_cols].astype("float32")
    y_train = df.loc[train_mask, "label"].astype("int8")
    X_valid = df.loc[valid_mask, feature_cols].astype("float32")
    y_valid = df.loc[valid_mask, "label"].astype("int8")

    train_idx = df.index[train_mask].to_numpy()
    valid_idx = df.index[valid_mask].to_numpy()

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=1200,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=5,
        random_seed=args.seed,
        task_type="GPU",
        devices="0",
        verbose=100,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=60,
    )

    pred = model.predict_proba(X.iloc[valid_idx])[:, 1]

    valid_meta = df.iloc[valid_idx][
        ["entity_id", "target_source", "target_entity_id", "label"]
    ].copy()
    valid_meta["score"] = pred

    valid_meta.to_csv(
        outdir / "validation_predictions.tsv",
        sep="\t",
        index=False,
    )

    gt = pd.read_csv(
        args.ground_truth,
        sep="\t",
        dtype="string",
    )

    validation_ids = [str(x) for x in sample_frame.iloc[valid_groups_idx]["entity_id"]]

    best_threshold, best_f05 = tune_threshold(
        validation_ids,
        gt[gt["source1_entity_id"].astype(str).isin(validation_ids)],
        valid_meta.rename(columns={"entity_id": "source1_entity_id"}),
    )

    metrics = {
        "best_threshold": best_threshold,
        "macro_f05": best_f05,
        "train_rows": len(train_idx),
        "validation_rows": len(valid_idx),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "best_iteration": model.get_best_iteration(),
    }

    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    model.save_model(str(outdir / "catboost_gpu.cbm"))

    print("=" * 72)
    print("CATBOOST GPU RESULT")
    print("=" * 72)
    print(f"Best threshold : {best_threshold:.4f}")
    print(f"Macro F0.5     : {best_f05:.6f}")
