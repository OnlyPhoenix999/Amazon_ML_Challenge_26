import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.similarity_features import build_pair_features


PRED_PATH = ROOT / "experiments/models/fresh_base39_holdout_predictions.tsv"

S1_PATH = ROOT / "data/raw/dataset/train/train_source1.tsv"
S2_PATH = ROOT / "data/raw/dataset/train/train_source2.tsv"
S3_PATH = ROOT / "data/raw/dataset/train/train_source3.tsv"

OUT_DIR = ROOT / "experiments/error_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.69


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


def clean(value):
    if pd.isna(value):
        return ""
    return str(value)


def load_source(path, needed_ids):
    """Load only rows belonging to the FP/FN entities."""
    pieces = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000,
    ):
        mask = chunk["entity_id"].isin(needed_ids)

        if mask.any():
            pieces.append(chunk.loc[mask])

    if not pieces:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        )

    return pd.concat(pieces, ignore_index=True)


def main():

    print("Loading fresh holdout predictions...")

    pred = pd.read_csv(
        PRED_PATH,
        sep="\t",
        dtype={
            "entity_id": str,
            "target_entity_id": str,
            "label": int,
            "probability": float,
        },
    )

    pred["predicted"] = pred["probability"] >= THRESHOLD

    fp = pred[
        (pred["label"] == 0)
        & (pred["predicted"])
    ].copy()

    fn = pred[
        (pred["label"] == 1)
        & (~pred["predicted"])
    ].copy()

    print()
    print("========================================")
    print("FRESH HOLDOUT ERROR SUMMARY")
    print("========================================")
    print(f"Threshold:              {THRESHOLD}")
    print(f"Holdout rows:           {len(pred):,}")
    print(f"Entities:               {pred.entity_id.nunique():,}")
    print(f"Predicted positives:    {pred.predicted.sum():,}")
    print(f"TP:                     {(pred.label.eq(1) & pred.predicted).sum():,}")
    print(f"FP:                     {len(fp):,}")
    print(f"FN:                     {len(fn):,}")
    print()

    # --------------------------------------------------
    # Only load records needed by FP/FN pairs
    # --------------------------------------------------

    needed_s1 = set(fp["entity_id"]) | set(fn["entity_id"])

    needed_s2 = set()
    needed_s3 = set()

    for target in pd.concat(
        [
            fp["target_entity_id"],
            fn["target_entity_id"],
        ]
    ):
        if target.startswith("S2-"):
            needed_s2.add(target)
        elif target.startswith("S3-"):
            needed_s3.add(target)

    print("Loading relevant source records...")
    print(f"S1 records needed: {len(needed_s1)}")
    print(f"S2 records needed: {len(needed_s2)}")
    print(f"S3 records needed: {len(needed_s3)}")

    s1 = load_source(S1_PATH, needed_s1)
    s2 = load_source(S2_PATH, needed_s2)
    s3 = load_source(S3_PATH, needed_s3)

    s1_map = s1.set_index("entity_id").to_dict("index")
    s2_map = s2.set_index("entity_id").to_dict("index")
    s3_map = s3.set_index("entity_id").to_dict("index")

    # --------------------------------------------------
    # Build detailed error rows
    # --------------------------------------------------

    errors = []

    combined = pd.concat(
        [
            fp.assign(error_type="FALSE_POSITIVE"),
            fn.assign(error_type="FALSE_NEGATIVE"),
        ],
        ignore_index=True,
    )

    for _, row in combined.iterrows():

        source_id = row["entity_id"]
        target_id = row["target_entity_id"]

        source_record = s1_map.get(source_id, {})

        if target_id.startswith("S2-"):
            target_record = s2_map.get(target_id, {})
            target_source = "S2"
        else:
            target_record = s3_map.get(target_id, {})
            target_source = "S3"

        name1 = clean(source_record.get("business_name"))
        address1 = clean(source_record.get("business_address"))
        country1 = clean(source_record.get("country"))

        name2 = clean(target_record.get("business_name"))
        address2 = clean(target_record.get("business_address"))
        country2 = clean(target_record.get("country"))

        features = build_pair_features(
            name1=name1,
            name2=name2,
            address1=address1,
            address2=address2,
            country1=country1,
            country2=country2,
            source=target_source,
        )

        output = {
            "error_type": row["error_type"],
            "entity_id": source_id,
            "target_entity_id": target_id,
            "probability": row["probability"],
            "label": row["label"],
            "target_source": target_source,

            "business_name": name1,
            "business_address": address1,
            "country": country1,

            "target_business_name": name2,
            "target_business_address": address2,
            "target_country": country2,
        }

        for feature in FEATURE_COLUMNS:
            output[feature] = features[feature]

        errors.append(output)

    result = pd.DataFrame(errors)

    # --------------------------------------------------
    # Save complete machine-readable report
    # --------------------------------------------------

    all_path = OUT_DIR / "fresh_holdout_fp_fn_full.tsv"
    result.to_csv(
        all_path,
        sep="\t",
        index=False,
    )

    fp_result = result[
        result["error_type"] == "FALSE_POSITIVE"
    ].sort_values(
        "probability",
        ascending=False,
    )

    fn_result = result[
        result["error_type"] == "FALSE_NEGATIVE"
    ].sort_values(
        "probability",
        ascending=True,
    )

    fp_path = OUT_DIR / "fresh_holdout_false_positives.tsv"
    fn_path = OUT_DIR / "fresh_holdout_false_negatives.tsv"

    fp_result.to_csv(
        fp_path,
        sep="\t",
        index=False,
    )

    fn_result.to_csv(
        fn_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------
    # Human-readable console output
    # --------------------------------------------------

    display_columns = [
        "entity_id",
        "target_entity_id",
        "probability",
        "business_name",
        "business_address",
        "country",
        "target_business_name",
        "target_business_address",
        "target_country",
        "name_ratio",
        "norm_name_ratio",
        "name_no_suffix_ratio",
        "address_ratio",
        "norm_address_ratio",
        "address_number_match",
        "same_country",
    ]

    print()
    print("========================================")
    print("FALSE POSITIVES")
    print("========================================")

    print(
        fp_result[display_columns]
        .to_string(index=False)
    )

    print()
    print("========================================")
    print("FALSE NEGATIVES")
    print("========================================")

    print(
        fn_result[display_columns]
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Aggregate diagnostics
    # --------------------------------------------------

    print()
    print("========================================")
    print("FP FEATURE SUMMARY")
    print("========================================")

    if len(fp_result):
        print(
            fp_result[
                [
                    "probability",
                    "name_ratio",
                    "norm_name_ratio",
                    "name_no_suffix_ratio",
                    "address_ratio",
                    "norm_address_ratio",
                    "address_number_match",
                    "same_country",
                ]
            ].describe().round(3).to_string()
        )

    print()
    print("========================================")
    print("FN FEATURE SUMMARY")
    print("========================================")

    if len(fn_result):
        print(
            fn_result[
                [
                    "probability",
                    "name_ratio",
                    "norm_name_ratio",
                    "name_no_suffix_ratio",
                    "address_ratio",
                    "norm_address_ratio",
                    "address_number_match",
                    "same_country",
                ]
            ].describe().round(3).to_string()
        )

    print()
    print("========================================")
    print("FILES WRITTEN")
    print("========================================")
    print(all_path)
    print(fp_path)
    print(fn_path)


if __name__ == "__main__":
    main()