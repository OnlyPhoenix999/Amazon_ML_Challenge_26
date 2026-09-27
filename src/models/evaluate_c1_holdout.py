from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]

FEATURES = (
    ROOT
    / "data"
    / "processed"
    / "training_features_address_intelligence.tsv"
)

HOLDOUT_SPLIT = (
    ROOT
    / "experiments"
    / "splits"
    / "holdout_entities.tsv"
)

MODEL = (
    ROOT
    / "experiments"
    / "models"
    / "c1_targeted_xgb.json"
)

GT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

THRESHOLD = 0.64

print("=" * 70)
print("C1 — HOLDOUT EVALUATION")
print("=" * 70)

# ---------------------------------------------------------
# Load features
# ---------------------------------------------------------

df = pd.read_csv(
    FEATURES,
    sep="\t",
    dtype=str,
)

print(f"Feature rows: {len(df):,}")

feature_cols = [
    c
    for c in df.columns
    if c not in [
        "entity_id",
        "target_entity_id",
        "label",
    ]
]

for c in feature_cols:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )

# ---------------------------------------------------------
# Load holdout entities
# ---------------------------------------------------------

holdout_ids = set(
    pd.read_csv(
        HOLDOUT_SPLIT,
        sep="\t",
        dtype=str,
    )["entity_id"]
)

print(
    f"Holdout entities: "
    f"{len(holdout_ids):,}"
)

holdout = df[
    df["entity_id"].isin(holdout_ids)
].copy()

print(
    f"Holdout feature rows: "
    f"{len(holdout):,}"
)

# ---------------------------------------------------------
# Load C1 model
# ---------------------------------------------------------

print("\nLoading C1 model...")

model = xgb.XGBClassifier()

model.load_model(
    str(MODEL)
)

print("C1 model loaded.")

# ---------------------------------------------------------
# Predict holdout
# ---------------------------------------------------------

print("\nGenerating holdout predictions...")

X_holdout = holdout[feature_cols]

holdout["probability"] = model.predict_proba(
    X_holdout
)[:, 1]

# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------

print("Loading ground truth...")

gt = {}

for chunk in pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    chunksize=100000,
):

    for _, r in chunk.iterrows():

        sid = str(
            r["source1_entity_id"]
        )

        if sid not in holdout_ids:
            continue

        raw = (
            str(r["matched_entity_ids"])
            if pd.notna(
                r["matched_entity_ids"]
            )
            else ""
        )

        gt[sid] = {
            x.strip()
            for x in raw.split(",")
            if x.strip()
        }

print(
    f"Ground-truth entities loaded: "
    f"{len(gt):,}"
)

# ---------------------------------------------------------
# Entity F0.5
# ---------------------------------------------------------

def entity_f05(pred, actual):

    if not actual:
        return (
            1.0
            if not pred
            else 0.0
        )

    tp = len(
        pred & actual
    )

    fp = len(
        pred - actual
    )

    fn = len(
        actual - pred
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

    if precision == 0 and recall == 0:
        return 0.0

    return (
        1.25
        * precision
        * recall
        / (
            0.25 * precision
            + recall
        )
    )


# ---------------------------------------------------------
# Evaluate C1
# ---------------------------------------------------------

scores = []

total_tp = 0
total_fp = 0
total_fn = 0

prediction_counts = []

for sid in holdout_ids:

    group = holdout[
        holdout["entity_id"] == sid
    ]

    pred = set(
        group.loc[
            group["probability"] >= THRESHOLD,
            "target_entity_id",
        ]
    )

    actual = gt.get(
        sid,
        set(),
    )

    tp = len(
        pred & actual
    )

    fp = len(
        pred - actual
    )

    fn = len(
        actual - pred
    )

    total_tp += tp
    total_fp += fp
    total_fn += fn

    prediction_counts.append(
        len(pred)
    )

    scores.append(
        entity_f05(
            pred,
            actual,
        )
    )

macro_f05 = float(
    np.mean(scores)
)

pair_precision = (
    total_tp
    / (total_tp + total_fp)
    if total_tp + total_fp
    else 0.0
)

pair_recall = (
    total_tp
    / (total_tp + total_fn)
    if total_tp + total_fn
    else 0.0
)

# ---------------------------------------------------------
# Result
# ---------------------------------------------------------

print()
print("=" * 70)
print("C1 HOLDOUT RESULT")
print("=" * 70)

print(
    f"Holdout entities      : "
    f"{len(holdout_ids):,}"
)

print(
    f"Threshold             : "
    f"{THRESHOLD}"
)

print(
    f"TP                    : "
    f"{total_tp:,}"
)

print(
    f"FP                    : "
    f"{total_fp:,}"
)

print(
    f"FN                    : "
    f"{total_fn:,}"
)

print(
    f"Pair precision        : "
    f"{pair_precision:.6f}"
)

print(
    f"Pair recall           : "
    f"{pair_recall:.6f}"
)

print(
    f"Avg predictions/entity: "
    f"{np.mean(prediction_counts):.3f}"
)

print(
    f"C1 HOLDOUT Macro F0.5 : "
    f"{macro_f05:.6f}"
)

print()
print(
    "Address Intelligence "
    "HOLDOUT: 0.938234"
)

print(
    f"C1 improvement: "
    f"{macro_f05 - 0.938234:+.6f}"
)

print("=" * 70)