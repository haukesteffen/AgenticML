# Plan: Per-Family Feature Selection via Null Importance (Revised)

## Context

This repo (`AgenticML`) is an autoresearch harness iterating on `solution.py`
for Kaggle Playground tabular comps. Current comp is **playground-series-s6e4**
— multiclass classification of `Irrigation_Need`, scored on `balanced_accuracy`.
Train ~630k rows, test ~270k rows. Features: 11 numeric + 8 categorical, no
missingness.

Model families:
- **GBDT**: `exp/xgb`, `exp/lightgbm`, `exp/lightgbm2`, `exp/catboost`
- **Neural**: `exp/mlp`, `exp/mlp2`, `exp/realmlp`
- **Linear**: `exp/linear`

Prior context: `notebooks/adversarial_validation.ipynb` established that train
and test are i.i.d., so drops must be *noise-driven*, not shift-driven.

## Why this is a revision

A first pass at this analysis
(`notebooks/feature_selection_null_importance.ipynb`, currently on disk)
produced results that cannot be trusted:

1. **Linear result broken.** It used `|coef_|.mean(axis=0)` as the importance
   measure. Under L2-regularized `LogisticRegression`, null-label runs shrink
   coefficients strongly, while real-label runs grow them. Consequence: every
   feature scores `null_pct ≈ 1.0` — the measure can't distinguish signal
   from noise. All 19 features came out as "keep," which is the measure's
   failure mode, not a real finding.
2. **GBDT result hit the gain-based null-importance artifact.** With 6
   dominant features, LightGBM concentrates its attention on those under real
   labels and barely splits on the other 13. Under shuffled labels, no feature
   is predictive, so LGBM spreads splits roughly evenly, giving the weak
   features *more* gain than they get with real labels. Many features ended
   up with `real_gain < null_gain`, labeling them "drop" for the wrong
   reason.
3. **Cross-family reconciliation was meaningless** given the broken linear.
4. **Minor leakage**: `OneHotEncoder` and `StandardScaler` were fit on the
   full 80k subsample before the train/val split.
5. **Split noise in MLP null distribution**: each null iteration used a fresh
   `train_test_split`, so the null distribution absorbed split variance on
   top of label-shuffle variance, inflating `null_max`.

Only the MLP result was defensible, and even that may be under-trained
(`max_iter=50`, early stopping).

The fix: use a **single, unified permutation-importance methodology across
all three families** on a **fixed train/val split** with encoders fit
**inside** the training split. Measure importance as the drop in
`balanced_accuracy` (the competition metric) when a feature's columns are
permuted in validation.

## Unified methodology

For each family:

1. Fit a preprocessing pipeline (family-specific) on the training split only.
   Encode both train and val with it.
2. Fit the family's representative model on `(X_tr_enc, y_tr)`.
3. Score baseline: `balanced_accuracy_score(y_va, model.predict(X_va_enc))`.
4. For each original feature, permute *all* of its expanded columns together
   (respecting the OHE `col_group`), re-score, record the drop.
5. Real importance vector: 19 values, one per feature.
6. Null iterations: shuffle `y` (the training label vector) at the start of
   each iteration. Re-fit the model on the shuffled labels, re-score
   baseline, re-compute permutation drops. **Keep the train/val row indices
   and preprocessing pipeline fixed** across real and all null runs, so the
   only source of variance is the label shuffle (plus any intrinsic model
   nondeterminism via the iteration's seed).

For each feature, summarize:
- `null_pct`: percentile of real-importance within that feature's null
  distribution.
- `null_ratio`: `real / max(null)`.
- label: `keep` (`null_pct ≥ 0.99`), `borderline` (`0.95 ≤ null_pct < 0.99`),
  `drop` (`null_pct < 0.95`).

Because all three families now use the same metric (balanced-accuracy drop
under permutation), cross-family comparison becomes meaningful.

## Harness constraints

- Notebook is exploratory; does not replace `solution.py`.
- Work on the current branch (`eda`). Do not switch or modify any `exp/*`
  branch.
- Do not modify `solution.py`, `harness/`, `config.yaml`, or `.env`.
- Read `data/train.csv` only.

## Notebook location and file handling

- Overwrite `notebooks/feature_selection_null_importance.ipynb` with the
  revised notebook. The prior file is uncommitted and produces unreliable
  results; keeping it around would invite confusion. Regenerate
  `notebooks/feature_selection_results.json` and
  `notebooks/feature_selection_table.csv` in place.
- Use the existing `uv`-managed environment. Deps: `pandas`, `numpy`,
  `scikit-learn`, `lightgbm`, `matplotlib`. Do not add new dependencies.

## Tractability

Permutation importance adds a prediction-pass-per-feature per iteration,
which is modest overhead on 16k validation rows.

- Subsample train to **80k rows**, stratified on `y`, `random_state=42`.
- Fixed 80/20 train/val split on the subsample (64k train / 16k val),
  `random_state=42`, stratified.
- `N_NULL = 50` iterations per family.
- Soft per-family time budget: 20 minutes. If exceeded, cut `N_NULL` to 30
  and note this in the summary cell.
- MLP: bump `max_iter` to 200 (with `early_stopping=True`,
  `validation_fraction=0.1`) so the "no signal" verdict on weak features
  is from a model that actually converged — not one that stopped at 50
  iterations.

## Notebook structure

### 1. Setup
- Load `data/train.csv`, drop `id`, factorize target via
  `pd.factorize(..., sort=True)` (matches the harness's encoding).
- Stratified subsample to 80k rows.
- Identify `num_cols` / `cat_cols`.
- Split once, globally: `X_tr_raw, X_va_raw, y_tr, y_va` via stratified
  `train_test_split(..., test_size=0.2, random_state=42)`. This split is
  reused across all three families and all null iterations.

### 2. Shared scaffolding
- `permutation_importance_group(predict_fn, X_va_enc, y_va, col_group,
  feature_names, perm_seed)`:
  - Baseline `balanced_accuracy_score`.
  - For each feature, permute all columns in `col_group[feature]` together
    (single shared permutation across the group), re-predict, re-score.
  - Return vector of `(baseline - permuted_score)` per feature.
- `null_importance_study(fit_and_predict, X_tr, X_va, y_tr, y_va,
  col_group, feature_names, n_null, seed, time_budget, label)`:
  - Fit once on real `y_tr`, measure real permutation importance.
  - Loop `n_null` iterations: shuffle `y_tr` (per-iter seed), re-fit,
    measure permutation importance using the same `X_va`.
  - Return `(imp_real, imp_null_matrix, n_null_done)`.
- `summarize(imp_real, imp_null, feature_names)` → per-feature table
  (same as before, with `null_pct` computed robustly even when the null
  distribution is flat-zero).

### 3. Family 1 — GBDT (LightGBM)
- Preprocessing: `astype('category')` on categorical columns, fit on
  `X_tr_raw` only (i.e., establish the category set from train), apply to
  val. No scaling.
- `col_group`: one column per feature (native categoricals; no OHE
  expansion).
- Model: `LGBMClassifier(n_estimators=200, learning_rate=0.05,
  num_leaves=31, random_state=<iter_seed>, n_jobs=-1, verbose=-1)`,
  fit with `categorical_feature=cat_cols`.
- Run study, display table, save `gbdt_drops`.

### 4. Family 2 — Neural (MLP)
- Preprocessing: fit `OneHotEncoder(drop='first', handle_unknown='ignore')`
  and `StandardScaler()` on `X_tr_raw` *only*; apply both to train and val.
- `col_group`: maps each original feature to its expanded column indices.
- Model: `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
  early_stopping=True, validation_fraction=0.1, random_state=<iter_seed>)`.
- Run study, display table, save `mlp_drops`.

### 5. Family 3 — Linear (logistic regression)
- Preprocessing: same OHE + scaler pipeline as §4 (refit on this family's
  train so the pipeline object is independent; same settings).
- Same `col_group` shape as §4.
- Model: `LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
  solver='lbfgs', random_state=<iter_seed>)`.
- Run study, display table, save `linear_drops`.

### 6. Cross-family reconciliation
- Combined table: feature × {gbdt, mlp, linear} with `null_pct` and
  `label`.
- `globally_safe_drops` = drop in all three.
- `{family}_only_drops` for each.
- Heatmap over `null_pct` with values annotated.

### 7. Save artifacts
- Overwrite `notebooks/feature_selection_results.json` with the same
  schema as before, including `methodology: "permutation_importance_balanced_accuracy"`
  in the metadata block.
- Overwrite `notebooks/feature_selection_table.csv`.

### 8. Summary cell
Final markdown cell, self-contained, answers:
1. How many features are globally safe to drop? Which ones?
2. Per family, how many additional drops are warranted?
3. Three paste-ready `HYPOTHESIS` candidates, one per family.
4. Asymmetries across families (if the unified methodology still surfaces
   any — now they would be real disagreements, not measurement artifacts).
5. A one-line confidence read: compare the 6 "always keep" features to the
   previous (broken) run's 6 survivors. If they match, the qualitative
   signal is robust; if they differ, call it out.

## Deliverables

1. `notebooks/feature_selection_null_importance.ipynb` — revised,
   executed, outputs saved.
2. `notebooks/feature_selection_results.json` — per-family drop lists +
   metadata.
3. `notebooks/feature_selection_table.csv` — per-feature × per-family
   table.
4. Three paste-ready `HYPOTHESIS` strings in the final markdown cell.

## Non-goals

- Do **not** modify `solution.py`, any `exp/*` branch, `harness/`, or
  `config.yaml`.
- Do **not** run the harness.
- Do **not** implement the drops — that's a follow-up one-axis attempt per
  family.
- Do **not** tune model hyperparameters further. The goal is importance
  ranking with a reasonable baseline, not a championship model.
- Do **not** add CatBoost, RealMLP, LightGBM2, or MLP2 as separate
  studies.
- Do **not** fall back to `|coef_|` or gain-based importance for any
  family — they were the problem last time.
