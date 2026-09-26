from pathlib import Path
import time
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit(
        "LightGBM is not installed. Run: pip install lightgbm"
    )

FEATURE_FILE = Path("data/processed/training_features_v4.tsv")
GT_FILE = Path("data/raw/dataset/train/train_ground_truth.tsv")
TRAIN_IDS = Path("experiments/splits/train_entities.tsv")
DEV_IDS = Path("experiments/splits/dev_entities.tsv")
HOLDOUT_IDS = Path("experiments/splits/holdout_entities.tsv")
OUT_DIR = Path("experiments/models/lightgbm_lambdarank")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
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


def load_ids(path):
    return set(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
        )["entity_id"].astype(str)
    )


def load_gt(path):
    gt = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000,
    ):
        for _, r in chunk.iterrows():

            # IMPORTANT:
            # Ground-truth file uses source1_entity_id,
            # while feature table uses entity_id.
            s1 = str(r["source1_entity_id"])

            raw = (
                str(r["matched_entity_ids"])
                if pd.notna(r["matched_entity_ids"])
                else ""
            )

            ids = {
                x.strip()
                for x in raw.split(",")
                if x.strip()
            }

            gt[s1] = ids

    return gt


def f05(p, r):
    if p == 0 and r == 0:
        return 0.0

    return (
        1.25 * p * r
        / (0.25 * p + r)
    )


def entity_macro_f05(pred, gt):

    entities = set(gt) | set(pred)

    scores = []

    for s1 in entities:

        actual = gt.get(s1, set())
        predicted = pred.get(s1, set())

        if not actual:
            scores.append(
                1.0
                if not predicted
                else 0.0
            )
            continue

        tp = len(
            actual & predicted
        )

        precision = (
            tp / len(predicted)
            if predicted
            else 0.0
        )

        recall = (
            tp / len(actual)
        )

        scores.append(
            f05(
                precision,
                recall,
            )
        )

    return float(
        np.mean(scores)
    )


def pair_metrics(pred, gt):

    tp = 0
    fp = 0
    fn = 0

    for s1 in set(gt) | set(pred):

        actual = gt.get(
            s1,
            set()
        )

        predicted = pred.get(
            s1,
            set()
        )

        tp += len(
            actual & predicted
        )

        fp += len(
            predicted - actual
        )

        fn += len(
            actual - predicted
        )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    return precision, recall


def make_groups(df):

    return (
        df.groupby(
            "entity_id",
            sort=False
        )
        .size()
        .to_numpy()
    )


def predict_pairs(
    df,
    scores,
    threshold
):

    mask = (
        scores >= threshold
    )

    pred = {}

    selected = df.loc[
        mask,
        [
            "entity_id",
            "target_entity_id",
        ],
    ]

    for s1, group in selected.groupby(
        "entity_id"
    ):

        pred[str(s1)] = set(
            group[
                "target_entity_id"
            ].astype(str)
        )

    return pred


print(
    "Loading feature table..."
)

t0 = time.time()

df = pd.read_csv(
    FEATURE_FILE,
    sep="\t",
)

df["entity_id"] = (
    df["entity_id"]
    .astype(str)
)

df["target_entity_id"] = (
    df["target_entity_id"]
    .astype(str)
)


train_ids = load_ids(
    TRAIN_IDS
)

dev_ids = load_ids(
    DEV_IDS
)

holdout_ids = load_ids(
    HOLDOUT_IDS
)


train = df[
    df["entity_id"]
    .isin(train_ids)
].copy()

dev = df[
    df["entity_id"]
    .isin(dev_ids)
].copy()

holdout = df[
    df["entity_id"]
    .isin(holdout_ids)
].copy()


print(
    f"Train:   {len(train_ids):,} entities, "
    f"{len(train):,} rows"
)

print(
    f"Dev:     {len(dev_ids):,} entities, "
    f"{len(dev):,} rows"
)

print(
    f"Holdout: {len(holdout_ids):,} entities, "
    f"{len(holdout):,} rows"
)


X_train = train[
    FEATURES
]

y_train = train[
    "label"
].astype(int)

X_dev = dev[
    FEATURES
]

y_dev = dev[
    "label"
].astype(int)


groups_train = make_groups(
    train
)

groups_dev = make_groups(
    dev
)


print(
    "Training LightGBM LambdaRank..."
)


model = lgb.LGBMRanker(

    objective="lambdarank",

    metric="ndcg",

    ndcg_at=[10],

    learning_rate=0.05,

    n_estimators=1000,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=20,

    subsample=0.9,

    colsample_bytree=0.9,

    reg_lambda=1.0,

    random_state=42,

    verbosity=-1,
)


model.fit(

    X_train,

    y_train,

    group=groups_train,

    eval_set=[
        (X_dev, y_dev)
    ],

    eval_group=[
        groups_dev
    ],

    callbacks=[

        lgb.early_stopping(
            50,
            verbose=True
        ),

        lgb.log_evaluation(
            50
        ),
    ],
)


dev_scores = model.predict(
    X_dev
)

holdout_scores = model.predict(
    holdout[FEATURES]
)


gt = load_gt(
    GT_FILE
)


dev_gt = {
    k: gt[k]
    for k in dev_ids
    if k in gt
}


holdout_gt = {
    k: gt[k]
    for k in holdout_ids
    if k in gt
}


print(
    "\nSelecting threshold on DEV only..."
)


# LightGBM LambdaRank produces
# ranking scores, NOT probabilities.
# Therefore we search thresholds
# over the actual score distribution.

unique_scores = np.unique(
    dev_scores
)

if len(unique_scores) > 300:

    thresholds = np.quantile(
        dev_scores,
        np.linspace(
            0.50,
            0.999,
            500,
        ),
    )

else:

    thresholds = unique_scores


best = None


for threshold in thresholds:

    pred = predict_pairs(
        dev,
        dev_scores,
        float(threshold),
    )

    score = entity_macro_f05(
        pred,
        dev_gt,
    )

    precision, recall = pair_metrics(
        pred,
        dev_gt,
    )

    if (
        best is None
        or score > best[0]
    ):

        best = (
            score,
            float(threshold),
            precision,
            recall,
            len(pred),
        )


(
    dev_score,
    threshold,
    dev_precision,
    dev_recall,
    dev_pred_entities,
) = best


print(
    f"DEV threshold:       "
    f"{threshold:.8f}"
)

print(
    f"DEV Entity Macro F0.5: "
    f"{dev_score:.6f}"
)

print(
    f"DEV pair precision:   "
    f"{dev_precision:.6f}"
)

print(
    f"DEV pair recall:      "
    f"{dev_recall:.6f}"
)

print(
    f"DEV entities with predictions: "
    f"{dev_pred_entities}"
)


print(
    "\nEvaluating untouched HOLDOUT..."
)


holdout_pred = predict_pairs(
    holdout,
    holdout_scores,
    threshold,
)


holdout_score = entity_macro_f05(
    holdout_pred,
    holdout_gt,
)


holdout_precision, holdout_recall = (
    pair_metrics(
        holdout_pred,
        holdout_gt,
    )
)


print(
    "\n=============================="
)

print(
    "LIGHTGBM LAMBDARANK RESULT"
)

print(
    "=============================="
)

print(
    f"DEV threshold:        "
    f"{threshold:.8f}"
)

print(
    f"DEV Macro F0.5:       "
    f"{dev_score:.6f}"
)

print(
    f"HOLDOUT Macro F0.5:   "
    f"{holdout_score:.6f}"
)

print(
    f"HOLDOUT pair precision: "
    f"{holdout_precision:.6f}"
)

print(
    f"HOLDOUT pair recall:    "
    f"{holdout_recall:.6f}"
)

print(
    "Baseline:             "
    "0.919182"
)

print(
    f"Runtime:              "
    f"{time.time() - t0:.1f}s"
)


# Save model
model.booster_.save_model(
    str(
        OUT_DIR /
        "model.txt"
    )
)


# Save DEV predictions
pd.DataFrame({

    "entity_id":
        dev["entity_id"].values,

    "target_entity_id":
        dev[
            "target_entity_id"
        ].values,

    "label":
        dev["label"].values,

    "score":
        dev_scores,

}).to_csv(

    OUT_DIR /
    "dev_predictions.tsv",

    sep="\t",

    index=False,
)


# Save HOLDOUT predictions
pd.DataFrame({

    "entity_id":
        holdout[
            "entity_id"
        ].values,

    "target_entity_id":
        holdout[
            "target_entity_id"
        ].values,

    "label":
        holdout["label"].values,

    "score":
        holdout_scores,

}).to_csv(

    OUT_DIR /
    "holdout_predictions.tsv",

    sep="\t",

    index=False,
)


with open(
    OUT_DIR /
    "summary.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "LightGBM LambdaRank\n"
    )

    f.write(
        f"dev_threshold="
        f"{threshold:.8f}\n"
    )

    f.write(
        f"dev_macro_f05="
        f"{dev_score:.6f}\n"
    )

    f.write(
        f"holdout_macro_f05="
        f"{holdout_score:.6f}\n"
    )

    f.write(
        f"holdout_pair_precision="
        f"{holdout_precision:.6f}\n"
    )

    f.write(
        f"holdout_pair_recall="
        f"{holdout_recall:.6f}\n"
    )

    f.write(
        "baseline_macro_f05="
        "0.919182\n"
    )


print(
    f"\nSaved to: {OUT_DIR}"
)