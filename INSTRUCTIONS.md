# AgenticML

AgenticML is an autoresearch-style harness for iteratively improving tabular
ML models on Kaggle Playground competitions. You (the agent) edit
`solution.py`; a fixed harness runs cross-validation, logs every attempt to
MLflow, commits improvements to this experiment branch, and resets failures.

Task details (dataset, target, metric, problem type) live in `config.yaml`.
Read the module docstring of `solution.py` for the full contract on
`HYPOTHESIS` and `fit_predict`.

## What you edit
- `solution.py` — your model code (the only file the harness executes)
- `NOTES.md` — optional scratchpad for cross-attempt planning

## Harness lifecycle

When you run `python -m harness run`, the following happens in order:

1. **Commit** — the harness commits your changes to `solution.py` (and
   `NOTES.md` if present) using `HYPOTHESIS` as the commit message.
2. **Smoke test** — runs `fit_predict` on a small data subset (2 folds,
   5 % of data) with a short timeout. Catches obvious crashes cheaply.
3. **Full CV** — runs all folds of cross-validation. Each fold has a hard
   timeout (see `budget.fold_seconds` in `config.yaml`). If any fold
   exceeds this budget, the entire run is killed.
4. **Score comparison** — the harness compares the mean CV score against
   the best previous `status=improved` run on this branch.
   - **Improved**: the commit stays; the branch moves forward.
   - **Regressed** (or any failure): the commit is reset
     (`git reset --hard HEAD~1`) and the branch returns to the previous
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

- The **only** command that executes `solution.py` is `python -m harness run`.
  Never import, run, or test `solution.py` directly — doing so bypasses the
  time budget, commit logic, and MLflow logging.
- After running `python -m harness run`, **wait for the process to finish**
  and read the `RESULT` line before doing anything else.
- `HYPOTHESIS` must be a plain string literal at module scope.
- Do not import or call `mlflow` — the harness handles all logging.
- Do not touch: `config.yaml`, `harness/`, `data/`, `.env`.
- Prefer simpler solutions — avoid unnecessary complexity when a simpler
  approach achieves comparable results.

## Workflow

1. `python -m harness status` — see recent attempts and the current best score
2. Read current `solution.py`
3. Edit `solution.py` — update `HYPOTHESIS` and `fit_predict`
4. `python -m harness run` — wait for the `RESULT` line
5. Read the output, then iterate
