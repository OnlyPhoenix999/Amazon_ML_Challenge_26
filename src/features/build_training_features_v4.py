from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "experiments" / "training_pairs" / "training_pairs_v4.tsv"
OUTPUT_PATH = ROOT / "data" / "processed" / "training_features_v4.tsv"

CHUNK_SIZE = 10000

sys.path.insert(0, str(ROOT))

from src.features.similarity_features import build_pair_features


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

ID_COLUMNS = [
    "entity_id",
    "target_entity_id",
    "label",
]


def normalize_source(value: str) -> str:
    value = str(value).strip().lower()

    if value in {"source2", "s2"}:
        return "S2"

    if value in {"source3", "s3"}:
        return "S3"

    raise ValueError(f"Unexpected target_source value: {value!r}")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    total_rows = 0
    first_write = True

    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Chunk : {CHUNK_SIZE:,}")
    print()

    for chunk_number, df in enumerate(
        pd.read_csv(
            INPUT_PATH,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        df = df.fillna("")

        rows = []

        for row in df.itertuples(index=False):
            source = normalize_source(row.target_source)

            features = build_pair_features(
                row.business_name,
                row.target_business_name,
                row.business_address,
                row.target_business_address,
                row.country,
                row.target_country,
                source,
            )

            output_row = {
                "entity_id": row.entity_id,
                "target_entity_id": row.target_entity_id,
                **features,
                "label": int(row.label),
            }

            rows.append(output_row)

        feature_df = pd.DataFrame(
            rows,
            columns=ID_COLUMNS[:-1] + FEATURE_COLUMNS + ["label"],
        )

        feature_df.to_csv(
            OUTPUT_PATH,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False
        total_rows += len(feature_df)

        print(
            f"Chunk {chunk_number:>3} | "
            f"rows={len(feature_df):>6,} | "
            f"total={total_rows:>9,}"
        )

    print()
    print("Feature generation complete.")
    print(f"Rows:    {total_rows:,}")
    print(f"Features: {len(FEATURE_COLUMNS):,}")
    print(f"Columns: {len(FEATURE_COLUMNS) + 3:,}")
    print(f"Output:  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
