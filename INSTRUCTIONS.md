# AgenticML

AgenticML is an autoresearch-style harness for iteratively improving tabular
ML models on Kaggle Playground competitions. You (the agent) edit either
`solution.py` for base-model experiments or `ensemble.py` for OOF-backed
ensemble experiments; a fixed harness runs cross-validation, logs every
attempt to MLflow, commits improvements to this experiment branch, and resets
failures.

Task details (dataset, target, metric, problem type) live in `config.yaml`.
Read the module docstring of the file you are editing for the full contract on
`HYPOTHESIS` and `fit_predict`.

## What you edit
- `solution.py` — base-model experiments
- `ensemble.py` — OOF-backed blend / stacker experiments
- `NOTES.md` — optional scratchpad for cross-attempt planning

## Harness lifecycle

When you run `python -m harness run`, the following happens in order:

1. **Commit** — the harness commits your changes to exactly one experiment
   file (`solution.py` or `ensemble.py`, plus `NOTES.md` if present) using
   `HYPOTHESIS` as the commit message.
2. **Smoke test** — runs `fit_predict` on a small data subset (2 folds,
   5 % of data) with a short timeout. Catches obvious crashes cheaply.
3. **Full CV** — runs all folds of cross-validation. Each fold has a hard
   timeout (see `budget.fold_seconds` in `config.yaml`). If any fold
   exceeds this budget, the entire run is killed.
4. **Score comparison** — the harness compares the mean CV score against
   the best previous `status=improved` run on this branch.
   - **Improved**: the commit stays; the branch moves forward.
   - **Regressed** (or any failure): the harness discards the just-tested
     experiment file changes and returns the branch to the previous
     best state.

The run finishes by printing a machine-readable summary line:
```
RESULT status=improved score=0.8342 best=0.8310 folds=5 elapsed=347s loc=120
```

## Time budget

Each fold has a hard wall-clock timeout of `budget.fold_seconds` seconds
(see `config.yaml`). Plan your model complexity so that a single fold
completes well within this limit. The smoke test has its own separate
timeout (`budget.smoke_seconds`).

## Rules

- The **only** command that executes your experiment file is
  `python -m harness run`. Never import, run, or test `solution.py` or
  `ensemble.py` directly — doing so bypasses the time budget, commit logic,
  and MLflow logging.
- After running `python -m harness run`, **wait for the process to finish**
  and read the `RESULT` line before doing anything else.
- `HYPOTHESIS` must be a plain string literal at module scope.
- Do not import or call `mlflow` — the harness handles all logging.
- Do not touch: `config.yaml`, `harness/`, `data/`, `.env`.
- Prefer simpler solutions — avoid unnecessary complexity when a simpler
  approach achieves comparable results.

## Change discipline

**Each attempt makes exactly one change.**

The axes are:
1. Preprocessing (imputation, scaling, outlier handling)
2. Feature engineering (new derived feature(s), transforms, encodings)
3. Hyperparameters (tune the current estimator)
4. Ensembling / stacking (combine multiple estimators)

If your `HYPOTHESIS` cannot be written as a single clause naming one axis,
you are changing too much — split it into separate attempts.

## Workflow

1. `python -m harness status` — see recent attempts and the current best score
2. Read the current experiment file (`solution.py` or `ensemble.py`)
3. Pick **one** axis to change (see Change discipline). Edit exactly one
   experiment file — update `fit_predict` for that one axis only, and set
   `HYPOTHESIS` to a single clause naming the axis and the change.
4. `python -m harness run` — wait for the `RESULT` line
5. Read the output, then iterate
