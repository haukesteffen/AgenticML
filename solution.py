"""
AgenticML solution module.

This file is the only thing you (the agent) edit. The harness in ``harness/``
runs cross-validation over the dataset and calls ``fit_predict`` once per fold,
passing only the training and validation slices of that fold. You never see
the other folds, the test set, or the CV indices themselves.

Contract
--------
Define exactly two things at module scope:

  HYPOTHESIS : str
      A one-line plain string literal describing what this attempt tries.
      Used as the git commit message and MLflow tag. Must be a literal — it
      is read via ast.parse without executing the module.

  fit_predict(X_train, y_train, X_val) -> np.ndarray
      Train your model on (X_train, y_train) and return predictions on X_val.

Inputs
------
  X_train : pandas.DataFrame  — training fold features (id column already dropped)
  y_train : numpy.ndarray     — training fold targets (integer-encoded for
                                 classification via pd.factorize, float for regression)
  X_val   : pandas.DataFrame  — validation fold features (id column already dropped)

Return shape
------------
  regression                  -> 1D array of shape (len(X_val),)
  binary_classification       -> 1D array of shape (len(X_val),), probability
                                 of the positive class
  multiclass_classification   -> 2D array of shape (len(X_val), n_classes),
                                 per-class probabilities with columns in
                                 ascending class-index order (matching pd.factorize
                                 with sort=True)

What the harness captures automatically
---------------------------------------
``mlflow.autolog()`` is enabled in the worker before ``fit_predict`` runs.
Any sklearn / xgboost / lightgbm estimator you call ``.fit()`` on has its
hyperparameters logged automatically into a per-fold nested MLflow run.
You do NOT need to import mlflow or log anything yourself.

Rules
-----
- Do not import or call mlflow — the harness owns logging.
- Do not touch anything under ``harness/``, ``data/``, ``.env``, or ``config.yaml``.
- Do not read test data — you only have what arrives via function arguments.
- Feature engineering must be done inside ``fit_predict`` so it runs on the
  training fold only (no cross-fold leakage).
- HYPOTHESIS must be a plain string literal at module scope.

Cookbook
-------
sklearn Pipeline (works for all 3 problem types)::

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_val)

XGBoost (multiclass)::

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        objective="multi:softprob", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)

Optuna-wrapped XGBoost (best params land in autolog via the final fit)::

    import optuna
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
        m = XGBClassifier(**params, n_jobs=-1)
        return cross_val_score(m, X_train, y_train, cv=3,
                               scoring="balanced_accuracy").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best = XGBClassifier(**study.best_params, n_jobs=-1)
    best.fit(X_train, y_train)
    return best.predict_proba(X_val)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HYPOTHESIS = "balanced logistic regression with engineered stress interactions, quantile bins, and categorical crosses"


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["water_input_total"] = out["Rainfall_mm"] + out["Previous_Irrigation_mm"]
    out["water_input_gap"] = out["Rainfall_mm"] - out["Previous_Irrigation_mm"]
    out["irrigation_per_area"] = out["Previous_Irrigation_mm"] / (out["Field_Area_hectare"] + 1.0)
    out["rainfall_per_area"] = out["Rainfall_mm"] / (out["Field_Area_hectare"] + 1.0)
    out["evaporation_stress"] = (
        out["Temperature_C"] * out["Sunlight_Hours"] * (out["Wind_Speed_kmh"] + 1.0)
    ) / (out["Humidity"] + 5.0)
    out["moisture_retention"] = out["Soil_Moisture"] * out["Organic_Carbon"]
    out["conductivity_moisture"] = out["Electrical_Conductivity"] * out["Soil_Moisture"]
    out["temp_humidity_interaction"] = out["Temperature_C"] * out["Humidity"]
    out["ph_neutral_distance_sq"] = (out["Soil_pH"] - 7.0) ** 2
    out["soil_moisture_sq"] = out["Soil_Moisture"] ** 2
    out["temperature_sq"] = out["Temperature_C"] ** 2
    out["humidity_sq"] = out["Humidity"] ** 2
    out["log_rainfall"] = np.log1p(out["Rainfall_mm"])
    out["log_previous_irrigation"] = np.log1p(out["Previous_Irrigation_mm"])
    out["log_field_area"] = np.log1p(out["Field_Area_hectare"])
    out["log_electrical_conductivity"] = np.log1p(out["Electrical_Conductivity"])

    out["soil_crop"] = out["Soil_Type"].astype(str) + "__" + out["Crop_Type"].astype(str)
    out["season_region"] = out["Season"].astype(str) + "__" + out["Region"].astype(str)
    out["growth_irrigation"] = (
        out["Crop_Growth_Stage"].astype(str) + "__" + out["Irrigation_Type"].astype(str)
    )

    return out


def _add_quantile_bins(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    columns: list[str],
    q: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    val_out = val_df.copy()

    for col in columns:
        quantiles = np.unique(np.quantile(train_df[col], np.linspace(0.0, 1.0, q + 1)))
        if len(quantiles) <= 2:
            continue

        edges = quantiles.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        labels = [f"{col}_bin_{idx}" for idx in range(len(edges) - 1)]

        train_out[f"{col}_bin"] = pd.cut(
            train_df[col],
            bins=edges,
            labels=labels,
            include_lowest=True,
        ).astype("object")
        val_out[f"{col}_bin"] = pd.cut(
            val_df[col],
            bins=edges,
            labels=labels,
            include_lowest=True,
        ).astype("object")

    return train_out, val_out


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    X_train = _add_engineered_features(X_train)
    X_val = _add_engineered_features(X_val)
    X_train, X_val = _add_quantile_bins(
        X_train,
        X_val,
        columns=["Soil_Moisture", "Rainfall_mm", "Previous_Irrigation_mm", "Temperature_C"],
        q=5,
    )

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])

    pipe = Pipeline([
        ("preprocess", preprocessor),
        (
            "model",
            LogisticRegression(
                C=0.35,
                class_weight="balanced",
                max_iter=1500,
                n_jobs=-1,
                penalty="l2",
            ),
        ),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_val)
