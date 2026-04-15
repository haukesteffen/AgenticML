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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

HYPOTHESIS = "calibrated CatBoost with triple stress crosses and an evaporation-to-water ratio"


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    y_train = np.asarray(y_train)
    X_train_prepared = X_train.copy()
    X_val_prepared = X_val.copy()

    def add_features(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()

        low_moisture = (prepared["Soil_Moisture"] < 26.5).astype(np.int8)
        very_low_moisture = (prepared["Soil_Moisture"] < 20.5).astype(np.int8)
        low_rainfall = (prepared["Rainfall_mm"] < 700.0).astype(np.int8)
        extreme_low_rainfall = (prepared["Rainfall_mm"] < 350.0).astype(np.int8)
        high_temp = (prepared["Temperature_C"] > 30.0).astype(np.int8)
        extreme_heat = (prepared["Temperature_C"] > 35.0).astype(np.int8)
        high_wind = (prepared["Wind_Speed_kmh"] > 10.5).astype(np.int8)
        strong_wind = (prepared["Wind_Speed_kmh"] > 14.5).astype(np.int8)
        no_mulch = prepared["Mulching_Used"].eq("No").astype(np.int8)
        active_stage = prepared["Crop_Growth_Stage"].isin(["Vegetative", "Flowering"]).astype(np.int8)

        # Surface the sharp stress thresholds that appear to drive the label.
        prepared["low_moisture_flag"] = low_moisture
        prepared["very_low_moisture_flag"] = very_low_moisture
        prepared["low_rainfall_flag"] = low_rainfall
        prepared["extreme_low_rainfall_flag"] = extreme_low_rainfall
        prepared["high_temp_flag"] = high_temp
        prepared["extreme_heat_flag"] = extreme_heat
        prepared["high_wind_flag"] = high_wind
        prepared["strong_wind_flag"] = strong_wind
        prepared["no_mulch_flag"] = no_mulch
        prepared["active_stage_flag"] = active_stage
        prepared["stress_count"] = (
            low_moisture
            + very_low_moisture
            + low_rainfall
            + extreme_low_rainfall
            + high_temp
            + extreme_heat
            + high_wind
            + strong_wind
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
        prepared["water_balance"] = (
            prepared["Soil_Moisture"]
            + 0.015 * prepared["Rainfall_mm"]
            + 0.08 * prepared["Previous_Irrigation_mm"]
        ) / (1.0 + 0.04 * prepared["Temperature_C"] + 0.03 * prepared["Wind_Speed_kmh"])
        prepared["evaporation_balance_ratio"] = prepared["evaporation_stress"] / (
            prepared["water_balance"] + 1.0
        )
        prepared["canopy_demand"] = (
            active_stage
            * prepared["Temperature_C"]
            * prepared["Sunlight_Hours"]
            / (prepared["Humidity"] + 1.0)
        )
        prepared["dryness_gap"] = (
            (26.5 - prepared["Soil_Moisture"]).clip(lower=0)
            + 0.03 * (700.0 - prepared["Rainfall_mm"]).clip(lower=0)
        )

        prepared["moisture_band"] = pd.cut(
            prepared["Soil_Moisture"],
            bins=[-np.inf, 14.0, 20.5, 26.5, 40.0, np.inf],
            labels=["vlow", "low", "dry", "ok", "wet"],
        ).astype(str)
        prepared["temp_band"] = pd.cut(
            prepared["Temperature_C"],
            bins=[-np.inf, 21.0, 30.0, 35.0, np.inf],
            labels=["cool", "warm", "hot", "xhot"],
        ).astype(str)
        prepared["wind_band"] = pd.cut(
            prepared["Wind_Speed_kmh"],
            bins=[-np.inf, 8.5, 10.5, 14.5, np.inf],
            labels=["calm", "breeze", "windy", "harsh"],
        ).astype(str)
        prepared["rain_band"] = pd.cut(
            prepared["Rainfall_mm"],
            bins=[-np.inf, 350.0, 700.0, 1200.0, 1800.0, np.inf],
            labels=["xdry", "dry", "mid", "wet", "xwet"],
        ).astype(str)

        prepared["crop_stage"] = (
            prepared["Crop_Type"].astype(str) + "__" + prepared["Crop_Growth_Stage"].astype(str)
        )
        prepared["season_irrigation"] = (
            prepared["Season"].astype(str) + "__" + prepared["Irrigation_Type"].astype(str)
        )
        prepared["water_mulch"] = (
            prepared["Water_Source"].astype(str) + "__" + prepared["Mulching_Used"].astype(str)
        )
        prepared["stage_mulch"] = (
            prepared["Crop_Growth_Stage"].astype(str) + "__" + prepared["Mulching_Used"].astype(str)
        )
        prepared["stage_mulch_irrigation"] = (
            prepared["stage_mulch"] + "__" + prepared["Irrigation_Type"].astype(str)
        )
        prepared["stage_irrigation"] = (
            prepared["Crop_Growth_Stage"].astype(str) + "__" + prepared["Irrigation_Type"].astype(str)
        )
        prepared["crop_stage_mulch"] = (
            prepared["crop_stage"] + "__" + prepared["Mulching_Used"].astype(str)
        )
        prepared["risk_regime"] = (
            prepared["stage_mulch"]
            + "__"
            + prepared["moisture_band"]
            + "__"
            + prepared["temp_band"]
            + "__"
            + prepared["wind_band"]
        )
        prepared["dryness_signature"] = (
            prepared["stage_mulch_irrigation"] + "__" + prepared["moisture_band"] + "__" + prepared["rain_band"]
        )

        return prepared

    X_train_prepared = add_features(X_train_prepared)
    X_val_prepared = add_features(X_val_prepared)

    categorical_cols = X_train_prepared.select_dtypes(exclude=[np.number]).columns.tolist()

    if categorical_cols:
        for frame in (X_train_prepared, X_val_prepared):
            frame[categorical_cols] = frame[categorical_cols].fillna("__missing__").astype(str)

    common_params = {
        "allow_writing_files": False,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
    }

    if np.issubdtype(y_train.dtype, np.floating):
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=550,
            depth=7,
            learning_rate=0.05,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            l2_leaf_reg=7,
            random_strength=0.75,
            **common_params,
        )
        model.fit(X_train_prepared, y_train, cat_features=categorical_cols)
        return model.predict(X_val_prepared)

    n_classes = np.unique(y_train).size
    class_counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(np.float64)
    class_weights = class_counts.mean() / np.maximum(class_counts, 1.0)
    class_weights /= class_weights.mean()
    minority_idx = int(np.argmin(class_counts))
    majority_idx = int(np.argmax(class_counts))
    class_weights[minority_idx] *= 1.10
    class_weights[majority_idx] *= 0.98

    if n_classes == 2:
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=550,
            depth=7,
            learning_rate=0.05,
            one_hot_max_size=32,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            l2_leaf_reg=7,
            random_strength=0.75,
            class_weights=class_weights.tolist(),
            **common_params,
        )
        model.fit(X_train_prepared, y_train, cat_features=categorical_cols)
        return model.predict_proba(X_val_prepared)[:, 1]

    def build_multiclass_model() -> CatBoostClassifier:
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=550,
            depth=7,
            learning_rate=0.05,
            one_hot_max_size=32,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            l2_leaf_reg=7,
            random_strength=0.75,
            class_weights=class_weights.tolist(),
            **common_params,
        )

    best_multipliers = np.ones(n_classes, dtype=np.float64)
    remaining_indices = [idx for idx in range(n_classes) if idx not in (minority_idx, majority_idx)]
    if len(X_train_prepared) >= 20_000:
        fit_idx, calib_idx = train_test_split(
            np.arange(len(X_train_prepared)),
            test_size=0.12,
            stratify=y_train,
            random_state=42,
        )
        calibration_model = build_multiclass_model()
        calibration_model.fit(
            X_train_prepared.iloc[fit_idx],
            y_train[fit_idx],
            cat_features=categorical_cols,
        )
        calibration_probs = calibration_model.predict_proba(X_train_prepared.iloc[calib_idx])

        best_score = -np.inf
        for minority_mult in [1.00, 1.08, 1.16, 1.24, 1.32]:
            for mid_mult in [0.97, 1.00, 1.03, 1.06]:
                for majority_mult in [0.97, 1.00, 1.03]:
                    candidate = np.ones(n_classes, dtype=np.float64)
                    candidate[minority_idx] = minority_mult
                    candidate[majority_idx] = majority_mult
                    for idx in remaining_indices:
                        candidate[idx] = mid_mult
                    score = balanced_accuracy_score(
                        y_train[calib_idx],
                        np.argmax(calibration_probs * candidate, axis=1),
                    )
                    if score > best_score:
                        best_score = score
                        best_multipliers = candidate

    model = build_multiclass_model()
    model.fit(X_train_prepared, y_train, cat_features=categorical_cols)
    probabilities = model.predict_proba(X_val_prepared) * best_multipliers
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities
