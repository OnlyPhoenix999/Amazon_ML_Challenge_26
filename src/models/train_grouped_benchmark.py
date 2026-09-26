from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, fbeta_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "training_features_v4.tsv"
OUTPUT_DIR = ROOT / "experiments" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "entity_id"
TARGET_ID_COL = "target_entity_id"
LABEL_COL = "label"

DROP_COLUMNS = [
    ID_COL,
    TARGET_ID_COL,
    LABEL_COL,
]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def entity_macro_f05(entity_ids, target_ids, labels, probabilities, threshold):
    tmp = pd.DataFrame({
        "entity_id": entity_ids.to_numpy(),
        "target_entity_id": target_ids.to_numpy(),
        "label": labels.to_numpy(),
        "probability": probabilities,
    })

    scores = []

    for entity_id, group in tmp.groupby("entity_id", sort=False):
        true_targets = set(
            group.loc[group["label"] == 1, "target_entity_id"]
        )

        predicted_targets = set(
            group.loc[group["probability"] >= threshold, "target_entity_id"]
        )

        if not true_targets:
            scores.append(1.0 if not predicted_targets else 0.0)
            continue

        tp = len(true_targets & predicted_targets)
        fp = len(predicted_targets - true_targets)
        fn = len(true_targets - predicted_targets)

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        if precision == 0.0 and recall == 0.0:
            f05 = 0.0
        else:
            f05 = (
                1.25 * precision * recall
                / (0.25 * precision + recall)
            )

        scores.append(f05)

    return float(np.mean(scores))


def pair_metrics(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        pred,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_true,
        pred,
        beta=0.5,
        zero_division=0,
    )

    return precision, recall, f05


def search_threshold(
    entity_ids,
    target_ids,
    labels,
    probabilities,
):
    results = []

    thresholds = np.round(
        np.arange(0.05, 0.951, 0.01),
        2,
    )

    for threshold in thresholds:
        entity_f05 = entity_macro_f05(
            entity_ids,
            target_ids,
            labels,
            probabilities,
            threshold,
        )

        pair_precision, pair_recall, pair_f05 = pair_metrics(
            labels,
            probabilities,
            threshold,
        )

        results.append({
            "threshold": threshold,
            "entity_macro_f05": entity_f05,
            "pair_precision": pair_precision,
            "pair_recall": pair_recall,
            "pair_f05": pair_f05,
        })

    results_df = pd.DataFrame(results)

    best_idx = results_df["entity_macro_f05"].idxmax()

    return (
        results_df.loc[best_idx].to_dict(),
        results_df,
    )


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    val_meta,
):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    best, threshold_results = search_threshold(
        val_meta["entity_id"],
        val_meta["target_entity_id"],
        y_val,
        probabilities,
    )

    print(f"Best threshold:      {best['threshold']:.2f}")
    print(f"Entity Macro F0.5:    {best['entity_macro_f05']:.6f}")
    print(f"Pair precision:      {best['pair_precision']:.6f}")
    print(f"Pair recall:         {best['pair_recall']:.6f}")
    print(f"Pair F0.5:            {best['pair_f05']:.6f}")

    threshold_results.to_csv(
        OUTPUT_DIR / f"{name.lower()}_thresholds.tsv",
        sep="\t",
        index=False,
    )

    prediction_df = val_meta.copy()
    prediction_df["label"] = y_val.to_numpy()
    prediction_df["probability"] = probabilities

    prediction_df.to_csv(
        OUTPUT_DIR / f"{name.lower()}_validation_predictions.tsv",
        sep="\t",
        index=False,
    )

    return best


def main():
    print(f"Loading: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        dtype={
            "entity_id": str,
            "target_entity_id": str,
            "label": int,
        },
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    feature_columns = [
        c for c in df.columns
        if c not in DROP_COLUMNS
    ]

    X = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y = df[LABEL_COL].astype(int)

    groups = df[ID_COL]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    val_meta = df.iloc[val_idx][
        [ID_COL, TARGET_ID_COL]
    ].copy()

    print()
    print(f"Features:             {len(feature_columns):,}")
    print(f"Training rows:        {len(train_idx):,}")
    print(f"Validation rows:      {len(val_idx):,}")
    print(
        f"Training S1 entities: "
        f"{df.iloc[train_idx][ID_COL].nunique():,}"
    )
    print(
        f"Validation S1 entities: "
        f"{df.iloc[val_idx][ID_COL].nunique():,}"
    )
    print(
        f"Training positives:   {y_train.sum():,}"
    )
    print(
        f"Validation positives: {y_val.sum():,}"
    )

    results = {}

    # ------------------------------------------------------------
    # LightGBM CPU
    # ------------------------------------------------------------

    lightgbm_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    results["lightgbm"] = evaluate_model(
        "LightGBM",
        lightgbm_model,
        X_train,
        y_train,
        X_val,
        y_val,
        val_meta,
    )

    # ------------------------------------------------------------
    # XGBoost GPU
    # ------------------------------------------------------------

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )

    try:
        results["xgboost"] = evaluate_model(
            "XGBoost",
            xgb_model,
            X_train,
            y_train,
            X_val,
            y_val,
            val_meta,
        )
    except Exception as exc:
        print()
        print("XGBoost GPU failed:")
        print(repr(exc))

    # ------------------------------------------------------------
    # CatBoost GPU
    # ------------------------------------------------------------

    catboost_model = CatBoostClassifier(
        loss_function="Logloss",
        iterations=500,
        learning_rate=0.05,
        depth=8,
        random_seed=RANDOM_STATE,
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )

    try:
        results["catboost"] = evaluate_model(
            "CatBoost",
            catboost_model,
            X_train,
            y_train,
            X_val,
            y_val,
            val_meta,
        )
    except Exception as exc:
        print()
        print("CatBoost GPU failed:")
        print(repr(exc))

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        print(
            f"{name:10s} | "
            f"threshold={result['threshold']:.2f} | "
            f"entity_macro_f05={result['entity_macro_f05']:.6f} | "
            f"pair_precision={result['pair_precision']:.6f} | "
            f"pair_recall={result['pair_recall']:.6f} | "
            f"pair_f05={result['pair_f05']:.6f}"
        )

    pd.DataFrame(results).T.to_csv(
        OUTPUT_DIR / "grouped_model_summary.tsv",
        sep="\t",
    )


if __name__ == "__main__":
    main()
