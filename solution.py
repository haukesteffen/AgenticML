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
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

HYPOTHESIS = "ovr LightGBM with calibrated class offsets and a slightly further final refit uplift after early stopping"


def _engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()

    numeric_groups = {
        "water_inputs": ["Rainfall_mm", "Previous_Irrigation_mm"],
        "dryness": ["Temperature_C", "Sunlight_Hours", "Wind_Speed_kmh", "Humidity"],
        "soil": ["Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity", "Soil_pH"],
        "field": ["Field_Area_hectare"],
    }
    if all(col in engineered.columns for cols in numeric_groups.values() for col in cols):
        rainfall = engineered["Rainfall_mm"]
        irrigation = engineered["Previous_Irrigation_mm"]
        temperature = engineered["Temperature_C"]
        sunlight = engineered["Sunlight_Hours"]
        wind = engineered["Wind_Speed_kmh"]
        humidity = engineered["Humidity"]
        moisture = engineered["Soil_Moisture"]
        organic_carbon = engineered["Organic_Carbon"]
        conductivity = engineered["Electrical_Conductivity"]
        soil_ph = engineered["Soil_pH"]
        field_area = engineered["Field_Area_hectare"]

        engineered["Water_Input_Total"] = rainfall + irrigation
        engineered["Atmospheric_Demand"] = temperature * sunlight * wind / (humidity + 1.0)
        engineered["Soil_Water_Buffer"] = moisture * (organic_carbon + 1.0)
        engineered["Dryness_Stress"] = engineered["Atmospheric_Demand"] / (
            engineered["Soil_Water_Buffer"] + rainfall / 20.0 + irrigation + 1.0
        )
        engineered["Water_Per_Area"] = engineered["Water_Input_Total"] / (field_area + 0.5)
        engineered["Salinity_PH_Interaction"] = conductivity * soil_ph

    category_pairs = [
        ("Crop_Type", "Season"),
        ("Crop_Growth_Stage", "Season"),
        ("Soil_Type", "Region"),
        ("Irrigation_Type", "Water_Source"),
    ]
    for left, right in category_pairs:
        if left in engineered.columns and right in engineered.columns:
            engineered[f"{left}__{right}"] = (
                engineered[left].astype(str) + "__" + engineered[right].astype(str)
            )

    return engineered


def _align_categories(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X_train_prepared = _engineer_features(X_train)
    X_val_prepared = _engineer_features(X_val)

    categorical_cols = X_train_prepared.select_dtypes(
        include=["object", "category", "string"],
    ).columns.tolist()
    for col in categorical_cols:
        train_as_category = X_train_prepared[col].astype("category")
        X_train_prepared[col] = train_as_category
        X_val_prepared[col] = pd.Categorical(
            X_val_prepared[col],
            categories=train_as_category.cat.categories,
        )

    return X_train_prepared, X_val_prepared, categorical_cols


def _ovr_raw_scores(
    models: list[LGBMClassifier],
    X: pd.DataFrame,
) -> np.ndarray:
    return np.column_stack([model.predict_proba(X)[:, 1] for model in models])


def _normalized_ovr_proba(raw_scores: np.ndarray, class_biases: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw_scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    logits = logits + class_biases
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _fit_binary_lgbm(
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    random_state: int,
) -> LGBMClassifier:
    pos_count = int(y_fit.sum())
    neg_count = int(len(y_fit) - pos_count)
    pos_weight = 1.0 if pos_count == 0 else min(max(np.sqrt(neg_count / pos_count), 1.0), 6.0)
    sample_weight = np.where(y_fit == 1, pos_weight, 1.0)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1800,
        learning_rate=0.035,
        num_leaves=96,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=2.0,
        min_split_gain=0.01,
        max_bin=255,
        cat_smooth=20,
        cat_l2=10,
        n_jobs=-1,
        random_state=random_state,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(
        X_fit,
        y_fit,
        sample_weight=sample_weight,
        eval_set=[(X_eval, y_eval)],
        eval_metric="binary_logloss",
        callbacks=[early_stopping(100, verbose=False)],
    )
    return model


def _search_class_biases(raw_scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    candidate_steps = np.array([-1.0, -0.7, -0.45, -0.25, -0.1, 0.0, 0.1, 0.25, 0.45, 0.7, 1.0])
    class_biases = np.zeros(raw_scores.shape[1], dtype=float)
    best_score = balanced_accuracy_score(y_true, raw_scores.argmax(axis=1))

    for _ in range(3):
        improved = False
        for class_idx in range(raw_scores.shape[1]):
            best_local_bias = class_biases[class_idx]
            for step in candidate_steps:
                trial_biases = class_biases.copy()
                trial_biases[class_idx] += step
                trial_pred = _normalized_ovr_proba(raw_scores, trial_biases).argmax(axis=1)
                trial_score = balanced_accuracy_score(y_true, trial_pred)
                if trial_score > best_score + 1e-6:
                    best_score = trial_score
                    best_local_bias = trial_biases[class_idx]
                    improved = True
            class_biases[class_idx] = best_local_bias
        if not improved:
            break

    return class_biases


def _final_iteration_count(best_iteration: int, calibration_fraction: float) -> int:
    # Early stopping happens on a reduced fit subset; use a modest uplift when refitting on all rows.
    uplift = 1.0 + min(0.16, calibration_fraction + 0.01)
    return max(1, int(np.ceil(best_iteration * uplift)))


def fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
) -> np.ndarray:
    """Train a model on (X_train, y_train) and return predictions on X_val."""
    X_train_prepared, X_val_prepared, _ = _align_categories(X_train, X_val)

    common_params = {
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": -1,
    }

    if np.issubdtype(np.asarray(y_train).dtype, np.floating):
        model = LGBMRegressor(**common_params)
        model.fit(X_train_prepared, y_train)
        return model.predict(X_val_prepared)

    y_train_array = np.asarray(y_train)
    classes = np.unique(y_train_array)
    class_counts = np.bincount(y_train_array.astype(int), minlength=int(classes.max()) + 1)
    can_stratify = np.all(class_counts[class_counts > 0] >= 2)
    calibration_fraction = 0.12 if len(y_train_array) >= 50_000 else 0.2

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_prepared,
        y_train_array,
        test_size=calibration_fraction,
        random_state=42,
        stratify=y_train_array if can_stratify else None,
    )

    calibration_models: list[LGBMClassifier] = []
    calibration_scores = []
    for class_id in classes:
        binary_y_fit = (y_fit == class_id).astype(int)
        binary_y_cal = (y_cal == class_id).astype(int)
        model = _fit_binary_lgbm(
            X_fit,
            binary_y_fit,
            X_cal,
            binary_y_cal,
            random_state=42 + int(class_id),
        )
        calibration_models.append(model)
        calibration_scores.append(model.predict_proba(X_cal)[:, 1])

    calibration_raw_scores = np.column_stack(calibration_scores)
    class_biases = _search_class_biases(calibration_raw_scores, y_cal)
    best_iterations = [
        max(1, int(model.best_iteration_ or model.n_estimators))
        for model in calibration_models
    ]

    final_models: list[LGBMClassifier] = []
    for class_id, best_iteration in zip(classes, best_iterations, strict=False):
        final_n_estimators = _final_iteration_count(best_iteration, calibration_fraction)
        binary_target = (y_train_array == class_id).astype(int)
        final_model = LGBMClassifier(
            objective="binary",
            n_estimators=final_n_estimators,
            learning_rate=0.035,
            num_leaves=96,
            min_child_samples=120,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=2.0,
            min_split_gain=0.01,
            max_bin=255,
            cat_smooth=20,
            cat_l2=10,
            n_jobs=-1,
            random_state=42 + int(class_id),
            verbosity=-1,
            force_col_wise=True,
        )
        pos_count = int(binary_target.sum())
        neg_count = int(len(binary_target) - pos_count)
        pos_weight = 1.0 if pos_count == 0 else min(max(np.sqrt(neg_count / pos_count), 1.0), 6.0)
        sample_weight = np.where(binary_target == 1, pos_weight, 1.0)
        final_model.fit(X_train_prepared, binary_target, sample_weight=sample_weight)
        final_models.append(final_model)

    raw_val_scores = _ovr_raw_scores(final_models, X_val_prepared)
    proba = _normalized_ovr_proba(raw_val_scores, class_biases)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    return proba
