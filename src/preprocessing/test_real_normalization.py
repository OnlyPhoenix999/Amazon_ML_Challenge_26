from pathlib import Path
import pandas as pd

from normalization import normalize_record


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN = ROOT / "data" / "raw" / "dataset" / "train"

OUTPUT = (
    ROOT
    / "experiments"
    / "ground_truth_inspection"
    / "normalization_real_data_sample.tsv"
)


# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 20
RANDOM_STATE = 42


# ============================================================
# PROCESS
# ============================================================

all_results = []


for source in ["source1", "source2", "source3"]:

    path = TRAIN / f"train_{source}.tsv"

    print(f"\nReading {source}...")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=5000,
    )

    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )

    for row in sample.itertuples(index=False):

        result = normalize_record(
            row.business_name,
            row.business_address,
            row.country,
        )

        all_results.append({
            "source": source,
            "entity_id": row.entity_id,
            "country": row.country,

            "raw_name":
                row.business_name,

            "normalized_name":
                result["business_name_normalized"],

            "name_without_suffix":
                result["business_name_without_suffix"],

            "legal_suffix":
                result["legal_suffix"],

            "name_script":
                result["name_script"],

            "raw_address":
                row.business_address,

            "normalized_address":
                result["business_address_normalized"],

            "address_script":
                result["address_script"],
        })


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame(all_results)

results.to_csv(
    OUTPUT,
    sep="\t",
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("REAL-DATA NORMALIZATION SANITY CHECK")
print("=" * 70)

print(
    f"Records checked: {len(results):,}"
)

print(
    f"Output: {OUTPUT}"
)

print("\nSample:\n")

print(
    results[
        [
            "source",
            "country",
            "raw_name",
            "normalized_name",
            "legal_suffix",
            "raw_address",
            "normalized_address",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("SCRIPT DISTRIBUTION")
print("=" * 70)

print(
    results.groupby(
        ["source", "name_script"]
    ).size().to_string()
)

print("\nDone.")