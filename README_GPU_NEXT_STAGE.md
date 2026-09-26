# GPU Entity-Resolution Next Stage

## What we are combining

We keep the strongest parts of the two approaches:

1. Existing project normalization.
2. Frozen ALL_V4 blocking.
3. 39-feature entity-resolution representation.
4. Hard-negative training.
5. Source1-grouped validation.
6. Official entity-level Macro F0.5 threshold tuning.
7. RTX 3060 GPU for the gradient-boosted model.
8. CatBoost as a controlled A/B model, not an automatic replacement.
9. Optional GPU blocking is a benchmark only until its keys are proven identical to ALL_V4.

## Important blocker rule

Do NOT replace ALL_V4 with the teammate's earlier GPU blocking script without verification.

The earlier GPU benchmark uses the old five-key blocker. The current production blocker is:

- country + name_first_v2
- country + name_prefix2_v2
- country + address_number_v2
- country + address_first_v4

The generator in this bundle imports `build_keys()` directly from the frozen V4 benchmark script.

## Why the GPU is used here

Unicode normalization and RapidFuzz string matching remain CPU-side.

The RTX 3060 is used where the workload is naturally numerical:

- gradient-boosted model training
- model prediction
- later SHAP/model analysis
- optional GPU data-frame acceleration only after correctness is established

XGBoost's current GPU API uses `tree_method="hist"` and `device="cuda"`.
CatBoost uses `task_type="GPU"` and `devices="0"`.

## Pipeline

```text
raw TSV
  |
  v
normalization.py
  |
  v
ALL_V4
  |
  v
hard-negative training pairs
  |
  v
39 pair features
  |
  +--> XGBoost GPU
  |
  +--> CatBoost GPU
  |
  v
entity-level Macro F0.5
  |
  v
threshold comparison
```

## First experiment

Use a 100-entity sample first:

```powershell
python src/preprocessing/generate_training_pairs_v4.py `
  --sample-size 100 `
  --negatives-per-entity 50 `
  --chunk-size 100000
```

Build features:

```powershell
python src/features/pair_features.py `
  --input experiments/training_pairs/training_pairs_v4.tsv `
  --output experiments/training_pairs/features_v4.tsv `
  --chunksize 25000
```

Train XGBoost:

```powershell
python src/models/train_xgb_gpu.py `
  --features experiments/training_pairs/features_v4.tsv `
  --ground-truth data/raw/dataset/train/train_ground_truth.tsv `
  --source1-sample experiments/training_pairs/source1_training_sample.tsv
```

Train CatBoost on the same feature file:

```powershell
python src/models/train_catboost_gpu.py `
  --features experiments/training_pairs/features_v4.tsv `
  --ground-truth data/raw/dataset/train/train_ground_truth.tsv `
  --source1-sample experiments/training_pairs/source1_training_sample.tsv
```

## Then scale

Once the 100-S1 run is correct:

```powershell
python src/preprocessing/generate_training_pairs_v4.py `
  --sample-size 2000 `
  --negatives-per-entity 100 `
  --chunk-size 100000
```

Then regenerate features and train both models on exactly the same feature table.

## Fair comparison rule

XGBoost and CatBoost must use:

- identical candidate pairs
- identical labels
- identical 39 features
- identical Source1-grouped split
- identical threshold grid
- identical Macro F0.5 evaluator

Only the model changes.

## What not to change yet

Do not simultaneously change:

- ALL_V4
- normalization.py
- feature definitions
- candidate sampling
- train/validation split
- model

Otherwise an A/B comparison is not interpretable.

## Later experiments

Only after the baseline is stable:

1. XGBoost vs CatBoost.
2. Hard-negative ratio.
3. Character n-gram features.
4. Feature ablation.
5. One global threshold vs S2/S3 thresholds.
6. Classifier vs ranking objective.
7. Cross-script evaluation slices.
8. Larger training S1 population.
