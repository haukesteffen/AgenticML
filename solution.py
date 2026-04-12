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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HYPOTHESIS = "ordinal logistic regression with infrequent-category bucketing and richer water-stress ratio features"


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
    out["temp_moisture_interaction"] = out["Temperature_C"] * out["Soil_Moisture"]
    out["ph_neutral_distance_sq"] = (out["Soil_pH"] - 7.0) ** 2
    out["soil_moisture_sq"] = out["Soil_Moisture"] ** 2
    out["temperature_sq"] = out["Temperature_C"] ** 2
    out["humidity_sq"] = out["Humidity"] ** 2
    out["log_rainfall"] = np.log1p(out["Rainfall_mm"])
    out["log_previous_irrigation"] = np.log1p(out["Previous_Irrigation_mm"])
    out["log_field_area"] = np.log1p(out["Field_Area_hectare"])
    out["log_electrical_conductivity"] = np.log1p(out["Electrical_Conductivity"])
    out["rainfall_deficit"] = out["Sunlight_Hours"] * out["Temperature_C"] - out["Rainfall_mm"]
    out["humidity_gap"] = out["Humidity"] - out["Soil_Moisture"]
    out["water_buffer"] = out["Organic_Carbon"] * out["Soil_Moisture"] / (out["Electrical_Conductivity"] + 1.0)
    out["water_supply_ratio"] = (out["Rainfall_mm"] + out["Previous_Irrigation_mm"] + 1.0) / (
        out["Sunlight_Hours"] * out["Temperature_C"] + out["Wind_Speed_kmh"] + 1.0
    )
    out["dryness_index"] = (
        out["Temperature_C"] * (100.0 - out["Humidity"]) * (out["Sunlight_Hours"] + 1.0)
    ) / (out["Soil_Moisture"] + out["Rainfall_mm"] + out["Previous_Irrigation_mm"] + 5.0)
    out["salinity_stress"] = out["Electrical_Conductivity"] / (
        out["Soil_Moisture"] + out["Organic_Carbon"] + 1.0
    )
    out["ph_conductivity_interaction"] = out["Soil_pH"] * out["Electrical_Conductivity"]
    out["moisture_temperature_ratio"] = (out["Soil_Moisture"] + 1.0) / (out["Temperature_C"] + 1.0)
    out["rain_to_sunlight_ratio"] = (out["Rainfall_mm"] + 1.0) / (out["Sunlight_Hours"] + 1.0)
    out["irrigation_to_rain_ratio"] = (out["Previous_Irrigation_mm"] + 1.0) / (out["Rainfall_mm"] + 1.0)

    out["soil_crop"] = out["Soil_Type"].astype(str) + "__" + out["Crop_Type"].astype(str)
    out["season_region"] = out["Season"].astype(str) + "__" + out["Region"].astype(str)
    out["growth_irrigation"] = (
        out["Crop_Growth_Stage"].astype(str) + "__" + out["Irrigation_Type"].astype(str)
    )
    out["water_mulch"] = out["Water_Source"].astype(str) + "__" + out["Mulching_Used"].astype(str)
    out["soil_irrigation"] = out["Soil_Type"].astype(str) + "__" + out["Irrigation_Type"].astype(str)
    out["season_crop"] = out["Season"].astype(str) + "__" + out["Crop_Type"].astype(str)
    out["region_water"] = out["Region"].astype(str) + "__" + out["Water_Source"].astype(str)
    out["soil_mulch"] = out["Soil_Type"].astype(str) + "__" + out["Mulching_Used"].astype(str)

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


def _fit_binary_logistic(X_train, y_train, X_val, c_value: float, positive_weight: float) -> np.ndarray:
    model = LogisticRegression(
        C=c_value,
        class_weight={0: 1.0, 1: positive_weight},
        max_iter=1200,
        penalty="l2",
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


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
        columns=[
            "Soil_Moisture",
            "Rainfall_mm",
            "Previous_Irrigation_mm",
            "Temperature_C",
            "Humidity",
            "Electrical_Conductivity",
        ],
        q=6,
    )

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), numeric_cols),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=100,
                sparse_output=True,
            ),
            categorical_cols,
        ),
    ])
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)

    class_counts = np.bincount(y_train)
    high_idx = int(np.argmin(class_counts))
    low_idx = int(np.argmax(class_counts))
    medium_idx = int(next(idx for idx in range(len(class_counts)) if idx not in {high_idx, low_idx}))

    y_ge_medium = (y_train != low_idx).astype(int)
    y_ge_high = (y_train == high_idx).astype(int)

    p_ge_medium = _fit_binary_logistic(
        X_train_enc,
        y_ge_medium,
        X_val_enc,
        c_value=0.18,
        positive_weight=2.2,
    )
    p_ge_high = _fit_binary_logistic(
        X_train_enc,
        y_ge_high,
        X_val_enc,
        c_value=0.42,
        positive_weight=8.8,
    )

    p_ge_high = np.minimum(p_ge_high, p_ge_medium)
    probs = np.zeros((len(X_val), len(class_counts)), dtype=float)
    probs[:, low_idx] = 1.0 - p_ge_medium
    probs[:, medium_idx] = np.clip(p_ge_medium - p_ge_high, 0.0, 1.0)
    probs[:, high_idx] = p_ge_high

    row_sums = probs.sum(axis=1, keepdims=True)
    probs /= row_sums
    return probs
