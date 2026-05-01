# AgenticML

AgenticML is an autoresearch-style harness for iteratively improving tabular
ML models on Kaggle Playground competitions through diverse base models and
honest stacking.

The harness has two roles, signaled by which file you're editing:

- `solution.py` — base-model lane. The agent hill-climbs on plain 5-fold CV.
- `ensemble.py` — L2 stacking phase. The agent hill-climbs on L2's
  outer-OOF score from honest nested CV over promoted lanes.

Coordination artifacts live on `main`:

- `WIKI.md` — competition knowledge base. Edits are agent-proposed,
  human-approved.
- `LANES.md` — index of (recipe × family) lanes with status. Edits are
  agent-proposed, human-approved.
- `features/` — feature recipes. Each recipe is a self-contained file with a
  `build(train_df, test_df, target, id_column)` function. Outputs are cached.

## What you edit

- `solution.py` — base-model experiments. Declare `HYPOTHESIS`, `RECIPE`, and
  optionally `FAMILY` at module scope.
- `ensemble.py` — L2 ensemble experiments. Declare `HYPOTHESIS` and `SOURCES`
  (a list of promoted lane names).

Read the docstring of the file you're editing for the full contract.

## Harness lifecycle

### `python -m harness run`

1. **Commit** — the harness commits your changes to exactly one experiment
   file (`solution.py` or `ensemble.py`, plus `NOTES.md` if present) using
   `HYPOTHESIS` as the commit message.
2. **Smoke test** — runs `fit_predict` on a small data subset.
3. **Full CV** — runs all folds of cross-validation. Each fold has a hard
   timeout (`budget.fold_seconds`).
4. **Score comparison** — compares mean CV score against the best
   `improved` run on this branch.
   - **Improved**: the commit stays.
   - **Regressed** / failure: the harness discards the experiment file
     change and resets the branch.

The run finishes by printing:
```
RESULT status=improved score=0.8342 best=0.8310 folds=5 elapsed=347s loc=120
```

### `python -m harness promote`

Runs nested 5×5 CV on the current `solution.py`, logs honest L1 OOF
artifacts to the `promoted` MLflow experiment, and tags them with the lane
name `<RECIPE>__<FAMILY>`. **No improvement gate, no git commit.** Triggered
by the human when a lane is exhausted; ensembling phase then consumes those
artifacts as `SOURCES`.

### `python -m harness status [--experiment promoted] [--lane NAME]`

Default: current branch's runs. With `--experiment promoted`: every
promoted lane's best score. With `--lane NAME`: history of one lane.

## Time budget

Each fold has a hard wall-clock timeout of `budget.fold_seconds` seconds.
Plan your model complexity so a single fold completes well within this
limit. The smoke test has its own separate timeout (`budget.smoke_seconds`).

## Rules

- The **only** commands that execute experiment files are `python -m
  harness run` and `python -m harness promote`. Never import or run
  `solution.py` / `ensemble.py` directly.
- After running the harness, **wait for the process to finish** and read the
  `RESULT` (or `PROMOTED`) line.
- `HYPOTHESIS` must be a plain string literal at module scope.
- Do not import or call `mlflow` — the harness handles all logging.
- Do not touch: `config.yaml`, `harness/`, `data/`, `.env`.
- `WIKI.md` and `LANES.md` are agent-proposed / human-approved. Do not edit
  them autonomously.
- `features/` files belong to the architectural phase, not the iteration
  loop — discuss new recipes with the human before adding.
- Prefer simpler solutions.

## Change discipline

**Each attempt makes exactly one change.**

The axes are:
1. Preprocessing (imputation, scaling, outlier handling)
2. Feature engineering (a derived feature, transform, encoding — fold-aware
   only; deterministic transforms belong in a recipe)
3. Hyperparameters (tune the current estimator)
4. Ensembling / stacking (combine multiple estimators)

If your `HYPOTHESIS` cannot be written as a single clause naming one axis,
you are changing too much — split it into separate attempts.

## Workflow

1. `python -m harness status` — see recent attempts and the current best
   score.
2. Read the current experiment file (`solution.py` or `ensemble.py`).
3. Pick **one** axis to change. Edit exactly one experiment file — update
   `fit_predict` for that one axis only, set `HYPOTHESIS` to a single clause
   naming the axis and the change.
4. `python -m harness run` — wait for the `RESULT` line.
5. Read the output, then iterate.
