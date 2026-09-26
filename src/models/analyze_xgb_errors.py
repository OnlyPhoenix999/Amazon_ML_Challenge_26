from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PRED_PATH = ROOT / "experiments" / "models" / "xgboost_validation_predictions.tsv"
PAIRS_PATH = ROOT / "experiments" / "training_pairs" / "training_pairs_v4.tsv"
OUT_DIR = ROOT / "experiments" / "models"

THRESHOLD = 0.66


def f05(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return 1.25 * precision * recall / (0.25 * precision + recall)


def main():
    predictions = pd.read_csv(
        PRED_PATH,
        sep="\t",
        dtype=str,
    )

    predictions["label"] = predictions["label"].astype(int)
    predictions["probability"] = predictions["probability"].astype(float)

    pairs = pd.read_csv(
        PAIRS_PATH,
        sep="\t",
        dtype=str,
    ).fillna("")

    key_cols = [
        "entity_id",
        "target_entity_id",
    ]

    pair_cols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "target_source",
        "target_entity_id",
        "target_business_name",
        "target_business_address",
        "target_country",
        "matched_block",
    ]

    joined = predictions.merge(
        pairs[pair_cols],
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    joined["predicted"] = (
        joined["probability"] >= THRESHOLD
    ).astype(int)

    joined["error_type"] = np.select(
        [
            (joined["label"] == 0) & (joined["predicted"] == 1),
            (joined["label"] == 1) & (joined["predicted"] == 0),
            (joined["label"] == 1) & (joined["predicted"] == 1),
            (joined["label"] == 0) & (joined["predicted"] == 0),
        ],
        [
            "false_positive",
            "false_negative",
            "true_positive",
            "true_negative",
        ],
        default="unknown",
    )

    # ----------------------------------------------------------
    # Overall pair-level error counts
    # ----------------------------------------------------------

    print("=" * 70)
    print("XGBOOST ERROR ANALYSIS")
    print("=" * 70)
    print(f"Validation rows: {len(joined):,}")
    print(f"Threshold:       {THRESHOLD:.2f}")
    print()

    print("Error counts:")
    print(joined["error_type"].value_counts())

    # ----------------------------------------------------------
    # Entity-level validation summary
    # ----------------------------------------------------------

    entity_rows = []

    for entity_id, group in joined.groupby(
        "entity_id",
        sort=False,
    ):
        true_targets = set(
            group.loc[group["label"] == 1, "target_entity_id"]
        )

        predicted_targets = set(
            group.loc[
                group["predicted"] == 1,
                "target_entity_id",
            ]
        )

        tp = len(true_targets & predicted_targets)
        fp = len(predicted_targets - true_targets)
        fn = len(true_targets - predicted_targets)

        entity_rows.append({
            "entity_id": entity_id,
            "true_count": len(true_targets),
            "predicted_count": len(predicted_targets),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f05": f05(tp, fp, fn),
        })

    entity_summary = pd.DataFrame(entity_rows)

    print()
    print("Entity-level summary:")
    print(
        f"Entities: {len(entity_summary):,}"
    )
    print(
        f"Entities with FP: "
        f"{(entity_summary['fp'] > 0).sum():,}"
    )
    print(
        f"Entities with FN: "
        f"{(entity_summary['fn'] > 0).sum():,}"
    )
    print(
        f"Mean entity F0.5: "
        f"{entity_summary['f05'].mean():.6f}"
    )

    # ----------------------------------------------------------
    # False positives
    # ----------------------------------------------------------

    fp = joined[
        joined["error_type"] == "false_positive"
    ].sort_values(
        "probability",
        ascending=False,
    )

    fn = joined[
        joined["error_type"] == "false_negative"
    ].sort_values(
        "probability",
        ascending=True,
    )

    fp_high = fp[fp["probability"] >= 0.90]
    fn_low = fn[fn["probability"] <= 0.10]

    print()
    print(f"False positives:              {len(fp):,}")
    print(f"High-confidence FP >= 0.90:   {len(fp_high):,}")
    print(f"False negatives:              {len(fn):,}")
    print(f"Low-confidence FN <= 0.10:    {len(fn_low):,}")

    # ----------------------------------------------------------
    # Source breakdown
    # ----------------------------------------------------------

    print()
    print("False positives by source:")
    print(
        fp["target_source"]
        .value_counts()
        .to_string()
    )

    print()
    print("False negatives by source:")
    print(
        fn["target_source"]
        .value_counts()
        .to_string()
    )

    # ----------------------------------------------------------
    # Blocking-pattern breakdown
    # ----------------------------------------------------------

    fp["block_count"] = (
        fp["matched_block"]
        .str.split(",")
        .str.len()
    )

    fn["block_count"] = (
        fn["matched_block"]
        .str.split(",")
        .str.len()
    )

    print()
    print("False positives by number of matched blocks:")
    print(
        fp["block_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("False negatives by number of matched blocks:")
    print(
        fn["block_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ----------------------------------------------------------
    # Save useful inspection files
    # ----------------------------------------------------------

    inspection_cols = [
        "entity_id",
        "target_entity_id",
        "label",
        "probability",
        "target_source",
        "country",
        "target_country",
        "business_name",
        "target_business_name",
        "business_address",
        "target_business_address",
        "matched_block",
    ]

    fp[
        inspection_cols
    ].head(500).to_csv(
        OUT_DIR / "xgboost_top_false_positives.tsv",
        sep="\t",
        index=False,
    )

    fn[
        inspection_cols
    ].head(500).to_csv(
        OUT_DIR / "xgboost_top_false_negatives.tsv",
        sep="\t",
        index=False,
    )

    fp_high[
        inspection_cols
    ].to_csv(
        OUT_DIR / "xgboost_high_confidence_false_positives.tsv",
        sep="\t",
        index=False,
    )

    fn_low[
        inspection_cols
    ].to_csv(
        OUT_DIR / "xgboost_low_confidence_false_negatives.tsv",
        sep="\t",
        index=False,
    )

    entity_summary.sort_values(
        ["fp", "fn"],
        ascending=False,
    ).to_csv(
        OUT_DIR / "xgboost_entity_error_summary.tsv",
        sep="\t",
        index=False,
    )

    print()
    print("Saved:")
    print("  xgboost_top_false_positives.tsv")
    print("  xgboost_top_false_negatives.tsv")
    print("  xgboost_high_confidence_false_positives.tsv")
    print("  xgboost_low_confidence_false_negatives.tsv")
    print("  xgboost_entity_error_summary.tsv")


if __name__ == "__main__":
    main()
