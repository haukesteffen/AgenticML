# AgenticML

AgenticML is an autoresearch-style harness for iteratively improving tabular
ML models on Kaggle Playground competitions. You (the agent) edit
`solution.py`; a fixed harness runs cross-validation, logs every attempt to
MLflow, commits improvements to this experiment branch, and resets failures.
Each experiment lives on its own `exp/<name>` branch — attempts accumulate as
a linear history of improvements on that branch.

Task details (dataset, target, metric, problem type) live in `config.yaml`.
Read the module docstring of `solution.py` for the full harness contract.

## What you can edit
- `solution.py` — your model code
- `NOTES.md` — optional scratchpad for cross-attempt planning

## Workflow
1. `python -m harness status` — recent attempts on this branch
2. Read current `solution.py`
3. Edit `solution.py` — update `HYPOTHESIS` and `fit_predict`
4. `python -m harness run` — commits, smokes, runs full CV, logs MLflow, resets on failure

## Rules
- `HYPOTHESIS` must be a plain string literal at module scope
- Don't import or call mlflow — the harness handles it automatically
- Don't touch: `config.yaml`, `harness/`, `data/`, `.env`
