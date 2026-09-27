
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]

FEATURES = ROOT / "data" / "processed" / "training_features_address_intelligence.tsv"
TRAIN_SPLIT = ROOT / "experiments" / "splits" / "train_entities.tsv"
DEV_SPLIT = ROOT / "experiments" / "splits" / "dev_entities.tsv"

TARGETED_DIR = ROOT / "experiments" / "targeted_negatives_v2"

OUT = ROOT / "experiments" / "models"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("C1 — TARGETED NEGATIVE OVERSAMPLING")
print("=" * 70)

# ---------------------------------------------------------
# Load 49-feature table
# ---------------------------------------------------------

df = pd.read_csv(FEATURES, sep="\t", dtype=str)

print(f"Feature rows: {len(df):,}")

feature_cols = [
    c for c in df.columns
    if c not in ["entity_id", "target_entity_id", "label"]
]

for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["label"] = pd.to_numeric(df["label"])

# ---------------------------------------------------------
# Train / dev split
# ---------------------------------------------------------

train_ids = set(
    pd.read_csv(
        TRAIN_SPLIT,
        sep="\t",
        dtype=str,
    )["entity_id"]
)

dev_ids = set(
    pd.read_csv(
        DEV_SPLIT,
        sep="\t",
        dtype=str,
    )["entity_id"]
)

train = df[df["entity_id"].isin(train_ids)].copy()
dev = df[df["entity_id"].isin(dev_ids)].copy()

print(f"Train rows: {len(train):,}")
print(f"Dev rows:   {len(dev):,}")

# ---------------------------------------------------------
# Load targeted negatives
# ---------------------------------------------------------

targeted_files = [
    TARGETED_DIR / "similar_name.tsv",
    TARGETED_DIR / "similar_address.tsv",
]

targeted_ids = set()

for path in targeted_files:
    if not path.exists():
        continue

    t = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=["entity_id", "target_entity_id"],
    )

    for row in t.itertuples(index=False):
        targeted_ids.add(
            (str(row[0]), str(row[1]))
        )

print(
    f"Unique targeted pairs: "
    f"{len(targeted_ids):,}"
)

# ---------------------------------------------------------
# Identify targeted negatives inside 49-feature train set
# ---------------------------------------------------------

keys = list(
    zip(
        train["entity_id"].astype(str),
        train["target_entity_id"].astype(str),
    )
)

target_mask = np.array(
    [k in targeted_ids for k in keys]
)

targeted = train[
    target_mask
    & (train["label"] == 0)
].copy()

print(
    f"Targeted negatives found: "
    f"{len(targeted):,}"
)

# ---------------------------------------------------------
# CONTROL:
# oversample targeted negatives 3x
# ---------------------------------------------------------

OVERSAMPLE = 3

targeted_extra = pd.concat(
    [targeted] * OVERSAMPLE,
    ignore_index=True,
)

train_c1 = pd.concat(
    [
        train,
        targeted_extra,
    ],
    ignore_index=True,
)

print(
    f"Original train rows: "
    f"{len(train):,}"
)

print(
    f"C1 train rows:       "
    f"{len(train_c1):,}"
)

print(
    f"Extra targeted rows: "
    f"{len(targeted_extra):,}"
)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

X_train = train_c1[feature_cols]
y_train = train_c1["label"]

X_dev = dev[feature_cols]
y_dev = dev["label"]

model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cuda",
    eval_metric="logloss",
    random_state=42,
    n_jobs=4,
)

print("\nTraining C1...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_dev, y_dev)],
    verbose=False,
)

# ---------------------------------------------------------
# SAVE TRAINED C1 MODEL
# ---------------------------------------------------------

model_path = OUT / "c1_targeted_xgb.json"

model.save_model(str(model_path))

print(f"Saved model: {model_path}")

# ---------------------------------------------------------
# Save DEV predictions
# ---------------------------------------------------------

dev["probability"] = model.predict_proba(
    X_dev
)[:, 1]

pred_path = (
    OUT / "c1_targeted_dev_predictions.tsv"
)

dev[
    [
        "entity_id",
        "target_entity_id",
        "label",
        "probability",
    ]
].to_csv(
    pred_path,
    sep="\t",
    index=False,
)

print(f"\nSaved: {pred_path}")

# ---------------------------------------------------------
# Entity Macro F0.5
# ---------------------------------------------------------

gt_path = (
    ROOT
    / "data"
    / "raw"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

gt = {}

for chunk in pd.read_csv(
    gt_path,
    sep="\t",
    dtype=str,
    chunksize=100000,
):

    for _, r in chunk.iterrows():

        sid = str(r["source1_entity_id"])

        if sid not in dev_ids:
            continue

        raw = (
            str(r["matched_entity_ids"])
            if pd.notna(r["matched_entity_ids"])
            else ""
        )

        gt[sid] = {
            x.strip()
            for x in raw.split(",")
            if x.strip()
        }


def entity_f05(pred, actual):

    if not actual:
        return 1.0 if not pred else 0.0

    tp = len(pred & actual)
    fp = len(pred - actual)
    fn = len(actual - pred)

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

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def evaluate(threshold):

    scores = []

    for sid in dev_ids:

        group = dev[
            dev["entity_id"] == sid
        ]

        pred = set(
            group.loc[
                group["probability"] >= threshold,
                "target_entity_id",
            ]
        )

        scores.append(
            entity_f05(
                pred,
                gt.get(sid, set()),
            )
        )

    return float(np.mean(scores))


# ---------------------------------------------------------
# Threshold search
# ---------------------------------------------------------

best_score = -1
best_threshold = None

for threshold in np.arange(
    0.50,
    0.86,
    0.01,
):

    score = evaluate(
        round(float(threshold), 2)
    )

    if score > best_score:

        best_score = score
        best_threshold = round(
            float(threshold),
            2,
        )

print()
print("=" * 70)
print("C1 RESULT")
print("=" * 70)

print(
    f"Best DEV threshold : "
    f"{best_threshold}"
)

print(
    f"DEV Macro F0.5      : "
    f"{best_score:.6f}"
)

print(
    "\nReference Address Intelligence DEV: "
    "0.947557"
)

print(
    "Reference Address Intelligence HOLDOUT: "
    "0.938234"
)

print("=" * 70)

