# AgenticML

Autoresearch-style harness for iteratively improving tabular ML models on Kaggle Playground competitions. An agent edits `solution.py`; a fixed harness runs cross-validation, logs every attempt to MLflow, and auto-commits improvements.

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

2. Edit `solution.py` — update `HYPOTHESIS` and `fit_predict`.

3. Run the harness:
   ```bash
   uv run python -m harness run
   ```
   This commits your changes, runs a smoke test, then full CV. If the score improves, the commit stays. Otherwise the branch resets to the previous best.

4. Check experiment history:
   ```bash
   uv run python -m harness status
   ```

## MLflow

Each `exp/<name>` branch gets its own MLflow experiment (`{prefix}_{branch}`). Parent runs track the attempt-level summary (hypothesis, mean score, status). Nested child runs contain per-fold autologged hyperparameters.

## Agent instructions

See `INSTRUCTIONS.md` for the agent-facing primer and `solution.py`'s module docstring for the full contract.
