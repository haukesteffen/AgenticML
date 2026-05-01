"""
AgenticML ensemble (phase e) module.

This file is what you edit for L2 ensemble experiments. The harness runs L2
nested CV against a list of promoted lanes: each outer fold's L2-training
features come from the lanes' ``oof_inner[:, fold]`` and the L2-validation
features come from the lanes' ``oof_outer``. Final test predictions come from
refitting L2 on the full L1 outer-OOF table.

Contract
--------
Define exactly three things at module scope:

  HYPOTHESIS : str
      One-line plain string literal describing the change in this attempt.
      Used as the git commit message and MLflow tag.

  SOURCES : list[str]
      The lane names whose promoted artifacts feed L2. Each entry must match
      the ``lane`` tag of a run in the ``promoted`` MLflow experiment (e.g.
      ``"v1_raw__LGBMClassifier"``). Discoverable via:

          python -m harness status --experiment promoted

  fit_predict(X_train, y_train, X_val) -> np.ndarray
      Train your L2 model on the meta-features and return predictions for
      ``X_val``. The harness calls this once per outer fold and once more on
      the full table for the test prediction.

Meta-feature columns
--------------------
Multiclass sources contribute one column per class:
  ``<lane>__class_0``, ``<lane>__class_1``, ...

Binary-classification and regression sources contribute one column:
  ``<lane>__pred``

Rules
-----
- Do not import or call mlflow — the harness owns artifact resolution.
- Do not touch anything under ``harness/``, ``data/``, ``.env``, or ``config.yaml``.
- Use only the meta-features provided via ``X_train`` / ``X_val``.
- HYPOTHESIS must be a plain string literal at module scope.
"""

import numpy as np
import pandas as pd

HYPOTHESIS = "equal-weight average over selected promoted lanes"

SOURCES: list[str] = [
    # "v1_raw__LGBMClassifier",
    # "v3_target_enc__CatBoostClassifier",
]


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Average each source's prediction columns with equal weights."""
    if not SOURCES:
        raise ValueError("SOURCES is empty. Add at least one promoted lane name.")

    aliases: list[str] = []
    for column in X_val.columns:
        alias = column.split("__", 1)[0]
        if alias not in aliases:
            aliases.append(alias)

    blocks = [
        X_val[[c for c in X_val.columns if c.startswith(f"{alias}__")]].to_numpy()
        for alias in aliases
    ]
    preds = np.mean(blocks, axis=0)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0]
    return preds
