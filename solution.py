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

HYPOTHESIS = "calibrated CatBoost with holdout-picked tree count and a severe-stress minority boost"


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
        prepared["humidity_band"] = pd.cut(
            prepared["Humidity"],
            bins=[-np.inf, 42.0, 55.0, 69.0, 82.0, np.inf],
            labels=["dryair", "light", "mid", "humid", "saturated"],
        ).astype(str)
        prepared["irrigation_band"] = pd.cut(
            prepared["Previous_Irrigation_mm"],
            bins=[-np.inf, 16.0, 37.0, 61.0, 89.0, 110.0, np.inf],
            labels=["trace", "light", "medium", "heavy", "excess", "max"],
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

    def build_multiclass_model(iterations: int, use_best_model: bool = False) -> CatBoostClassifier:
        params = {
            "loss_function": "MultiClass",
            "iterations": iterations,
            "depth": 7,
            "learning_rate": 0.05,
            "one_hot_max_size": 32,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.85,
            "l2_leaf_reg": 7,
            "random_strength": 0.75,
            "class_weights": class_weights.tolist(),
            "use_best_model": use_best_model,
            **common_params,
        }
        if use_best_model:
            params["od_type"] = "Iter"
            params["od_wait"] = 60
        return CatBoostClassifier(
            **params,
        )

    best_multipliers = np.ones(n_classes, dtype=np.float64)
    best_iterations = 550
    stress_threshold = None
    stress_minority_boost = 1.0
    remaining_indices = [idx for idx in range(n_classes) if idx not in (minority_idx, majority_idx)]
    if len(X_train_prepared) >= 20_000:
        fit_idx, calib_idx = train_test_split(
            np.arange(len(X_train_prepared)),
            test_size=0.12,
            stratify=y_train,
            random_state=42,
        )
        calibration_model = build_multiclass_model(iterations=900, use_best_model=True)
        calibration_model.fit(
            X_train_prepared.iloc[fit_idx],
            y_train[fit_idx],
            eval_set=(X_train_prepared.iloc[calib_idx], y_train[calib_idx]),
            cat_features=categorical_cols,
        )
        best_iterations = int(np.clip(calibration_model.tree_count_, 400, 900))
        calibration_probs = calibration_model.predict_proba(X_train_prepared.iloc[calib_idx])
        calibration_frame = X_train_prepared.iloc[calib_idx]
        stress_counts = calibration_frame["stress_count"].to_numpy()
        severe_masks = {
            None: np.zeros(len(calibration_frame), dtype=bool),
            5: stress_counts >= 5,
            6: stress_counts >= 6,
            7: stress_counts >= 7,
            50: (
                calibration_frame["very_low_moisture_flag"].to_numpy().astype(bool)
                & calibration_frame["high_temp_flag"].to_numpy().astype(bool)
            ),
            51: (
                calibration_frame["low_moisture_flag"].to_numpy().astype(bool)
                & calibration_frame["high_temp_flag"].to_numpy().astype(bool)
                & calibration_frame["high_wind_flag"].to_numpy().astype(bool)
            ),
        }

        def score_multipliers(
            multipliers: np.ndarray,
            threshold: int | None = None,
            minority_boost: float = 1.0,
        ) -> float:
            adjusted_probs = calibration_probs * multipliers
            if threshold in severe_masks:
                mask = severe_masks[threshold]
                if mask.any() and minority_boost != 1.0:
                    adjusted_probs = adjusted_probs.copy()
                    adjusted_probs[mask, minority_idx] *= minority_boost
            adjusted_probs /= adjusted_probs.sum(axis=1, keepdims=True)
            return balanced_accuracy_score(
                y_train[calib_idx],
                np.argmax(adjusted_probs, axis=1),
            )

        best_score = -np.inf
        for minority_mult in [1.00, 1.08, 1.16, 1.24, 1.32, 1.40]:
            for mid_mult in [0.94, 0.97, 1.00, 1.03, 1.06]:
                for majority_mult in [0.94, 0.97, 1.00, 1.02, 1.05]:
                    candidate = np.ones(n_classes, dtype=np.float64)
                    candidate[minority_idx] = minority_mult
                    candidate[majority_idx] = majority_mult
                    for idx in remaining_indices:
                        candidate[idx] = mid_mult
                    for threshold in severe_masks:
                        for minority_boost in [1.00, 1.04, 1.08, 1.12, 1.16]:
                            score = score_multipliers(candidate, threshold, minority_boost)
                            if score > best_score:
                                best_score = score
                                best_multipliers = candidate
                                stress_threshold = threshold
                                stress_minority_boost = minority_boost
        for step in [0.04, 0.02, 0.01]:
            improved = True
            while improved:
                improved = False
                for idx in range(n_classes):
                    for scale in [1.0 - step, 1.0 + step]:
                        candidate = best_multipliers.copy()
                        candidate[idx] = np.clip(candidate[idx] * scale, 0.85, 1.50)
                        score = score_multipliers(
                            candidate,
                            stress_threshold,
                            stress_minority_boost,
                        )
                        if score > best_score:
                            best_score = score
                            best_multipliers = candidate
                            improved = True
                for threshold in severe_masks:
                    for minority_boost in [0.98, 1.00, 1.02, 1.04]:
                        candidate_boost = np.clip(
                            stress_minority_boost * minority_boost,
                            1.0,
                            1.25,
                        )
                        score = score_multipliers(
                            best_multipliers,
                            threshold,
                            candidate_boost,
                        )
                        if score > best_score:
                            best_score = score
                            stress_threshold = threshold
                            stress_minority_boost = candidate_boost
                            improved = True

    model = build_multiclass_model(iterations=best_iterations)
    model.fit(X_train_prepared, y_train, cat_features=categorical_cols)
    probabilities = model.predict_proba(X_val_prepared) * best_multipliers
    if stress_threshold is not None and stress_minority_boost != 1.0:
        stress_counts = X_val_prepared["stress_count"].to_numpy()
        if stress_threshold == 5:
            severe_mask = stress_counts >= 5
        elif stress_threshold == 6:
            severe_mask = stress_counts >= 6
        elif stress_threshold == 7:
            severe_mask = stress_counts >= 7
        elif stress_threshold == 50:
            severe_mask = (
                X_val_prepared["very_low_moisture_flag"].to_numpy().astype(bool)
                & X_val_prepared["high_temp_flag"].to_numpy().astype(bool)
            )
        else:
            severe_mask = (
                X_val_prepared["low_moisture_flag"].to_numpy().astype(bool)
                & X_val_prepared["high_temp_flag"].to_numpy().astype(bool)
                & X_val_prepared["high_wind_flag"].to_numpy().astype(bool)
            )
        if severe_mask.any():
            probabilities[severe_mask, minority_idx] *= stress_minority_boost
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities
