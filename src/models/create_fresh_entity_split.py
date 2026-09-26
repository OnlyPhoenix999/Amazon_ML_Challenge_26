from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "training_features_v4_augmented.tsv"
OUT_DIR = ROOT / "experiments" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026


def main():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        dtype={"entity_id": str, "target_entity_id": str, "label": int},
        usecols=["entity_id"],
    ).drop_duplicates()

    entities = df["entity_id"].to_frame()

    # First: 300 holdout / 1700 remaining
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=300,
        random_state=SEED,
    )

    remaining_idx, holdout_idx = next(
        first.split(entities, groups=entities["entity_id"])
    )

    remaining = entities.iloc[remaining_idx]
    holdout = entities.iloc[holdout_idx]

    # Then: 300 development / 1400 training
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=300,
        random_state=SEED + 1,
    )

    train_idx, dev_idx = next(
        second.split(remaining, groups=remaining["entity_id"])
    )

    train = remaining.iloc[train_idx]
    dev = remaining.iloc[dev_idx]

    assert len(train) == 1400
    assert len(dev) == 300
    assert len(holdout) == 300

    train["entity_id"].to_csv(
        OUT_DIR / "train_entities.tsv",
        sep="\t",
        index=False,
        header=True,
    )

    dev["entity_id"].to_csv(
        OUT_DIR / "dev_entities.tsv",
        sep="\t",
        index=False,
        header=True,
    )

    holdout["entity_id"].to_csv(
        OUT_DIR / "holdout_entities.tsv",
        sep="\t",
        index=False,
        header=True,
    )

    print("Fresh Source1 split created:")
    print(f"Train:    {len(train):,}")
    print(f"Dev:      {len(dev):,}")
    print(f"Holdout:  {len(holdout):,}")

    print()
    print("Overlap checks:")
    print("train n dev     =", len(set(train.entity_id) & set(dev.entity_id)))
    print("train n holdout =", len(set(train.entity_id) & set(holdout.entity_id)))
    print("dev n holdout   =", len(set(dev.entity_id) & set(holdout.entity_id)))


if __name__ == "__main__":
    main()
