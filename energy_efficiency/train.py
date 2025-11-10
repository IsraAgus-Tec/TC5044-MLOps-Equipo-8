# energy_efficiency/modeling/train.py
# Team 8 — Minimal trainer (LR/RF/GB) with shared preprocessing + MLflow

from __future__ import annotations

from typing import Dict, Tuple, Optional
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_validate

from ..config import (
    TARGET_COLS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
)

class Team8Trainer:
    """Entrena LR/RF/GB con un preprocesamiento numérico simple y registra CV en MLflow."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: Optional[list[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.df = df.copy()
        self.target_cols = target_cols or list(TARGET_COLS)
        self.test_size = test_size
        self.random_state = random_state
        self.feature_cols = [c for c in self.df.columns if c not in self.target_cols]

        self.X_train = self.X_test = self.Y_train = self.Y_test = None
        self.models: Dict[str, Pipeline] = {}
        self.cv_report: Dict[str, Dict[str, float]] = {}

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def split_data(self) -> None:
        X = self.df[self.feature_cols]
        Y = self.df[self.target_cols]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state
        )

    def _preprocessor(self) -> ColumnTransformer:
        cast = FunctionTransformer(lambda f: f.apply(pd.to_numeric, errors="coerce"), validate=False)
        num = Pipeline([("cast", cast), ("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        return ColumnTransformer([("num", num, self.feature_cols)], remainder="drop")

    def _models(self) -> Dict[str, object]:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=500, max_depth=10, random_state=self.random_state, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=250, learning_rate=0.1, max_depth=3, random_state=self.random_state
            ),
        }

    @staticmethod
    def _wrap(est):
        # Envuelve si el estimador no soporta multi-output nativamente
        if getattr(est, "_get_tags", lambda: {})().get("multioutput", False):
            return est
        return MultiOutputRegressor(est)

    def train(self, cv_folds: int = 5) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        if any(v is None for v in (self.X_train, self.Y_train)):
            raise RuntimeError("Debes ejecutar split_data() antes de train().")

        pre = self._preprocessor()
        scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

        for name, est in self._models().items():
            with mlflow.start_run(run_name=f"team8_{name}"):
                pipe = Pipeline([("pre", pre), ("model", self._wrap(est))])
                pipe.fit(self.X_train, self.Y_train)
                self.models[name] = pipe

                cv = cross_validate(pipe, self.X_train, self.Y_train, cv=cv_folds, scoring=scoring, n_jobs=-1)
                summary = {
                    "r2": float(cv["test_r2"].mean()),
                    "rmse": float(abs(cv["test_rmse"].mean())),
                    "mae": float(abs(cv["test_mae"].mean())),
                }
                self.cv_report[name] = summary
                for m, v in summary.items():
                    mlflow.log_metric(f"cv_{m}_mean", v)

                mlflow.log_params(
                    {"cv_folds": cv_folds, "test_size": self.test_size,
                     "random_state": self.random_state, "n_features": len(self.feature_cols)}
                )
                mlflow.set_tags({"team": "Equipo 8", "module": "trainer"})
                mlflow.sklearn.log_model(pipe, artifact_path="model")

        return self.models, self.cv_report
