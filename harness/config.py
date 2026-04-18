from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class DatasetConfig:
    train_path: str
    target: str
    id_column: str
    problem_type: str  # regression | binary_classification | multiclass_classification

    def __post_init__(self):
        valid = ("regression", "binary_classification", "multiclass_classification")
        if self.problem_type not in valid:
            raise ValueError(f"problem_type must be one of {valid}, got {self.problem_type!r}")


@dataclass(frozen=True)
class MetricConfig:
    name: str
    direction: str = field(init=False)

    MAXIMIZE = {"roc_auc", "accuracy", "balanced_accuracy"}
    MINIMIZE = {"logloss", "rmse", "mae"}

    def __post_init__(self):
        all_known = self.MAXIMIZE | self.MINIMIZE
        if self.name not in all_known:
            raise ValueError(f"metric.name must be one of {sorted(all_known)}, got {self.name!r}")
        object.__setattr__(self, "direction", "maximize" if self.name in self.MAXIMIZE else "minimize")


@dataclass(frozen=True)
class CVConfig:
    n_splits: int
    shuffle: bool
    seed: int


@dataclass(frozen=True)
class BudgetConfig:
    smoke_seconds: int
    fold_seconds: int


@dataclass(frozen=True)
class SmokeConfig:
    data_fraction: float
    n_splits: int


@dataclass(frozen=True)
class MLflowConfig:
    experiment_prefix: str
    competition_slug: str
    submissions_suffix: str = "submissions"
    kaggle_competition: str | None = None

    @property
    def kaggle_slug(self) -> str:
        return self.kaggle_competition or self.competition_slug


@dataclass(frozen=True)
class HarnessConfig:
    dataset: DatasetConfig
    metric: MetricConfig
    cv: CVConfig
    budget: BudgetConfig
    smoke: SmokeConfig
    mlflow: MLflowConfig
    project_root: Path

    @staticmethod
    def load(config_path: str = "config.yaml") -> HarnessConfig:
        project_root = Path(config_path).resolve().parent

        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        return HarnessConfig(
            dataset=DatasetConfig(**raw["dataset"]),
            metric=MetricConfig(name=raw["metric"]["name"]),
            cv=CVConfig(**raw["cv"]),
            budget=BudgetConfig(**raw["budget"]),
            smoke=SmokeConfig(**raw["smoke"]),
            mlflow=MLflowConfig(
                experiment_prefix=raw["mlflow"]["experiment_prefix"],
                competition_slug=raw["mlflow"]["competition_slug"],
                submissions_suffix=raw["mlflow"].get("submissions_suffix", "submissions"),
                kaggle_competition=raw["mlflow"].get("kaggle_competition"),
            ),
            project_root=project_root,
        )
