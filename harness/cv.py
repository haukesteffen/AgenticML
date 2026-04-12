from __future__ import annotations

from sklearn.model_selection import KFold, StratifiedKFold

from harness.config import CVConfig


def build_cv(
    problem_type: str,
    cv_config: CVConfig | None = None,
    *,
    n_splits: int | None = None,
    shuffle: bool | None = None,
    seed: int | None = None,
) -> KFold | StratifiedKFold:
    n = n_splits if n_splits is not None else cv_config.n_splits
    sh = shuffle if shuffle is not None else cv_config.shuffle
    sd = seed if seed is not None else cv_config.seed

    if problem_type == "regression":
        return KFold(n_splits=n, shuffle=sh, random_state=sd)
    return StratifiedKFold(n_splits=n, shuffle=sh, random_state=sd)
