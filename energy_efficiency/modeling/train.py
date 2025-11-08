"""
Entrenamiento y validación de modelos – Equipo 8.
Incluye pipelines para Linear Regression, Random Forest y Gradient Boosting.
"""
from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .features import build_preprocessor
from .config import RANDOM_STATE, TEST_SIZE, TARGET_COLS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT


@dataclass
class TrainResult:
    models: Dict[str, Pipeline]
    cv_summary: Dict[str, Dict[str, float]]  # {model: {r2, rmse, mae}}


def _wrap(estimator):
    # Asegura multi-salida cuando el estimador no la soporta nativamente
    return estimator if hasattr(estimator, "n_outputs_") else MultiOutputRegressor(estimator)


def train_models(df: pd.DataFrame) -> TrainResult:
    """
    Entrena 3 modelos con preprocesamiento uniforme y retorna
    pipelines + resumen de CV.
    """
    y = df[TARGET_COLS].copy()
    X = df.drop(columns=TARGET_COLS).copy()

    pre = build_preprocessor(list(X.columns))
    models = {
        "LinearRegression": _wrap(LinearRegression()),
        "RandomForest": _wrap(RandomForestRegressor(n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1)),
        "GradientBoosting": _wrap(GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        )),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_summary: Dict[str, Dict[str, float]] = {}
    trained: Dict[str, Pipeline] = {}

    for name, est in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", est)])

        with mlflow.start_run(run_name=name):
            pipe.fit(X_tr, y_tr)
            trained[name] = pipe

            # Holdout metrics
            y_hat = pipe.predict(X_te)
            r2 = r2_score(y_te, y_hat, multioutput="uniform_average")
            mae = mean_absolute_error(y_te, y_hat, multioutput="uniform_average")
            rmse = mean_squared_error(y_te, y_hat, multioutput="uniform_average", squared=False)

            # CV metrics (promedio)
            r2_cv = cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean()
            mae_cv = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()
            rmse_cv = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error").mean()

            cv_summary[name] = {"r2": float(r2_cv), "mae": float(mae_cv), "rmse": float(rmse_cv)}

            # Log MLflow
            mlflow.log_params({"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "n_features": X.shape[1]})
            mlflow.log_metrics({"holdout_r2": r2, "holdout_mae": mae, "holdout_rmse": rmse,
                                "cv_r2": r2_cv, "cv_mae": mae_cv, "cv_rmse": rmse_cv})
            mlflow.sklearn.log_model(pipe, artifact_path="model")

    return TrainResult(models=trained, cv_summary=cv_summary)
