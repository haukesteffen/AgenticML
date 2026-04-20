# AgenticML

Autoresearch-style harness for iteratively improving tabular ML models on Kaggle Playground competitions. An agent edits `solution.py` for base-model experiments or `ensemble.py` for OOF-backed ensemble experiments; a fixed harness runs cross-validation, logs every attempt to MLflow, and auto-commits improvements.

## Setup

```bash
uv venv && uv sync
cp .env.example .env   # fill in MLflow credentials
```

## Dataset

Drop your competition's `train.csv` into `data/` and update `config.yaml` with the target column, problem type, and metric.

## Workflow

1. Create an experiment branch:
   ```bash
   git checkout -b exp/<experiment-name>
   ```

2. Edit exactly one experiment file:
   - `solution.py` for a base-model experiment
   - `ensemble.py` for an OOF-backed blend / stacker experiment

3. Run the harness:
   ```bash
   uv run python -m harness run
   ```
   This commits the changed experiment file, runs a smoke test, then full CV. If the score improves, the commit stays. Otherwise the branch resets to the previous best.

4. Check experiment history:
   ```bash
   uv run python -m harness status
   ```

## MLflow

Each `exp/<name>` branch gets its own MLflow experiment (`{prefix}_{slug}_{branch}`). Parent runs track the attempt-level summary (`hypothesis`, `mean_score`, `std_score`, `status`, `experiment_kind`) plus step-indexed `fold_score` history for each CV fold. The parent run also stores the edited experiment file snapshot (`solution.py` or `ensemble.py`). Base-model CV runs log `oof_predictions.npy`, `test_predictions.npy`, and `predictions_manifest.json` — the full-train refit materializes test predictions at CV time so submissions are a no-refit artifact download. Ensemble runs log `sources.json`, `oof_predictions.npy`, `test_predictions.npy`, and `predictions_manifest.json`.

## Ensembling

`ensemble.py` defines a `SOURCES` list that selects logged runs by branch or explicit MLflow run id. The harness downloads each source run's `oof_predictions.npy`, validates compatibility, and passes the resulting meta-feature table into `fit_predict`. That makes simple blends, hill-climbing, and stackers all fit the same branch + harness + MLflow workflow.

`harness submit` currently supports only base-model runs. Ensemble submission lands in a follow-up rollout that extends the ensemble CV worker to materialize its own `test_predictions.npy`, after which the same one-path submit flow applies.

## Agent instructions

See `INSTRUCTIONS.md` for the agent-facing primer and the module docstrings in `solution.py` and `ensemble.py` for the full contracts.
