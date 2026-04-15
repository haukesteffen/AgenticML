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
from catboost import CatBoostClassifier, CatBoostRegressor

HYPOTHESIS = "engineered drought-stress flags and low-cardinality categorical crosses with balanced CatBoost"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    X_train_prepared = X_train.copy()
    X_val_prepared = X_val.copy()

    def add_features(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()

        low_moisture = (prepared["Soil_Moisture"] < 26.5).astype(np.int8)
        low_rainfall = (prepared["Rainfall_mm"] < 700.0).astype(np.int8)
        high_temp = (prepared["Temperature_C"] > 30.0).astype(np.int8)
        high_wind = (prepared["Wind_Speed_kmh"] > 10.5).astype(np.int8)
        no_mulch = prepared["Mulching_Used"].eq("No").astype(np.int8)
        active_stage = prepared["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"]).astype(np.int8)

        # Surface the sharp stress thresholds that appear to drive the label.
        prepared["low_moisture_flag"] = low_moisture
        prepared["low_rainfall_flag"] = low_rainfall
        prepared["high_temp_flag"] = high_temp
        prepared["high_wind_flag"] = high_wind
        prepared["no_mulch_flag"] = no_mulch
        prepared["active_stage_flag"] = active_stage
        prepared["stress_count"] = (
            low_moisture
            + low_rainfall
            + high_temp
            + high_wind
            + no_mulch
            + active_stage
        )
        prepared["stress_band"] = prepared["stress_count"].astype(str)

        prepared["soil_ph_neutral_gap"] = (prepared["Soil_pH"] - 6.5).abs()
        prepared["evaporation_stress"] = (
            prepared["Temperature_C"]
            * prepared["Sunlight_Hours"]
            * prepared["Wind_Speed_kmh"]
            / (prepared["Humidity"] + 1.0)
        )
        prepared["moisture_rain_ratio"] = prepared["Soil_Moisture"] / (prepared["Rainfall_mm"] + 1.0)
        prepared["irrigation_area_ratio"] = (
            prepared["Previous_Irrigation_mm"] / (prepared["Field_Area_hectare"] + 0.25)
        )

        prepared["crop_stage"] = (
            prepared["Crop_Type"].astype(str) + "__" + prepared["Crop_Growth_Stage"].astype(str)
        )
        prepared["season_irrigation"] = (
            prepared["Season"].astype(str) + "__" + prepared["Irrigation_Type"].astype(str)
        )
        prepared["water_mulch"] = (
            prepared["Water_Source"].astype(str) + "__" + prepared["Mulching_Used"].astype(str)
        )

        return prepared

    X_train_prepared = add_features(X_train_prepared)
    X_val_prepared = add_features(X_val_prepared)

    categorical_cols = X_train_prepared.select_dtypes(exclude=[np.number]).columns.tolist()

    if categorical_cols:
        for frame in (X_train_prepared, X_val_prepared):
            frame[categorical_cols] = frame[categorical_cols].astype(str).fillna("__missing__")

    common_params = {
        "allow_writing_files": False,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }

    if np.issubdtype(y_train.dtype, np.floating):
        model = CatBoostRegressor(loss_function="RMSE", **common_params)
        model.fit(X_train_prepared, y_train, cat_features=categorical_cols)
        return model.predict(X_val_prepared)

    n_classes = np.unique(y_train).size
    if n_classes == 2:
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=400,
            depth=6,
            learning_rate=0.06,
            one_hot_max_size=64,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            l2_leaf_reg=6,
            random_strength=0.5,
            auto_class_weights="Balanced",
            **common_params,
        )
        model.fit(X_train_prepared, y_train, cat_features=categorical_cols)
        return model.predict_proba(X_val_prepared)[:, 1]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=400,
        depth=6,
        learning_rate=0.06,
        one_hot_max_size=64,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        l2_leaf_reg=6,
        random_strength=0.5,
        auto_class_weights="Balanced",
        **common_params,
    )
    model.fit(X_train_prepared, y_train, cat_features=categorical_cols)
    return model.predict_proba(X_val_prepared)
